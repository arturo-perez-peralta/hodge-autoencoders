import json
import logging
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
import logging_loki
import mlflow
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file
import torch

from src.hodge_sae.datasets import ActivationDataLoader
from src.hodge_sae.models import BaseSparseAutoEncoder

ROOT_DIR = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def configure_loki(loki_url: str | None, model_name: str) -> None:
    """Adds loki."""
    if not loki_url:
        return
    if any(isinstance(h, logging_loki.LokiHandler) for h in logger.handlers):
        return
    handler = logging_loki.LokiHandler(
        url=loki_url,
        tags={"app": "hodge-sae", "model": model_name},
        version="1",
    )
    logger.addHandler(handler)


def flatten_dict(d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """Flatten nested dicts for MLFlow."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


def save_model(sae: BaseSparseAutoEncoder, cfg_dict: dict[str, Any], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file(sae.state_dict(), save_dir / "model.safetensors")
    (save_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    logger.info("Model checkpoint saved to %s", save_dir)


def run_training(
    sae: BaseSparseAutoEncoder,
    data_loader: ActivationDataLoader,
    cfg: DictConfig,
) -> float:
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.train.lr)
    samples_seen = 0
    step = 0
    epoch = 0
    last_metrics: dict[str, float] = {}

    try:
        while step < cfg.train.max_steps:
            epoch += 1
            steps_before = step

            for batch in data_loader:
                x = batch.activations
                sae_output = sae(x)
                loss_out = sae.compute_loss(x, sae_output)

                optimizer.zero_grad()
                loss_out.loss.backward()
                optimizer.step()
                sae.normalize_decoder_weights()

                samples_seen += x.shape[0]
                step += 1
                last_metrics = loss_out.metrics

                if step % cfg.train.log_every == 0 or step == cfg.train.max_steps:
                    mlflow.log_metrics(
                        {**last_metrics, "train/samples_seen": samples_seen, "train/epoch": epoch},
                        step=step,
                    )
                    summary = " | ".join(
                        f"{k.split('/')[-1]}={v:.4f}"
                        for k, v in last_metrics.items()
                        if k in ("loss/mse", "loss/total", "metrics/l0", "metrics/l0_hole")
                    )
                    logger.info(
                        "Step %d/%d | Epoch %d | Samples %s | %s",
                        step, cfg.train.max_steps, epoch, f"{samples_seen:,}", summary,
                    )

                if step >= cfg.train.max_steps:
                    break

            if step == steps_before:
                raise RuntimeError("DataLoader yielded 0 batches; aborting to avoid infinite loop.")

        logger.info("Training finished cleanly at step %d (%s samples).", step, f"{samples_seen:,}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted at step %d. Saving checkpoint...", step)


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> float:
    torch.manual_seed(cfg.seed)
    configure_loki(cfg.get("loki_url"), model_name=cfg.model.get("name", "unknown"))

    loader: ActivationDataLoader = instantiate(cfg.data, device=cfg.device)
    model: BaseSparseAutoEncoder = instantiate(cfg.model, input_dim=loader.d_in).to(cfg.device)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    if mlflow.get_experiment_by_name(cfg.mlflow.experiment_name) is None:
        mlflow.create_experiment(
            cfg.mlflow.experiment_name,
            artifact_location=cfg.mlflow.artifact_location,
        )
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    with mlflow.start_run(run_name=cfg.get("run_name")) as run:
        mlflow.log_params(flatten_dict({**cfg_dict, "d_in": loader.d_in}))
        mlflow.log_dict(cfg_dict, "hydra_config.yaml")

        try:
            final_mse = run_training(model, loader, cfg)
        finally:
            run_tag = cfg.get("run_name") or model.__class__.__name__
            save_dir = ROOT_DIR / "storage" / "models" / f"{run_tag}_{run.info.run_id[:8]}"
            save_model(model, cfg_dict, save_dir)
            mlflow.log_param("model_path", str(save_dir))
            if cfg.mlflow.get("register_model", False):
                mlflow.log_artifacts(str(save_dir), artifact_path="model")
                mlflow.register_model(
                    model_uri=f"runs:/{run.info.run_id}/model",
                    name=cfg.mlflow.registered_model_name,
                )

    return final_mse


if __name__ == "__main__":
    main()