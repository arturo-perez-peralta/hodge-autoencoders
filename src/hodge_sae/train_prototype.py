import logging
import os

import json
from pathlib import Path
from safetensors.torch import save_file

import logging_loki
import mlflow
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

from src.hodge_sae.model.hodge_sae import (
    HodgeSparseAutoEncoder,
    TwoDimensionalHodgeSparseAutoEncoder,
)
from src.hodge_sae.model.sae import SparseAutoEncoder

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("hodge-sae-experiments")

logger = logging.getLogger("hodge-trainer")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

loki_url = os.getenv("LOKI_URL")
if loki_url:
    loki_handler = logging_loki.LokiHandler(
        url=loki_url,
        tags={"app": "hodge-sae", "model": "hodge2d"},
        version="1",
    )
    logger.addHandler(loki_handler)

config = {
    "train": {
        "total_tokens": 50_000_000,
        "batch_size": 32,
        "seq_length": 128,
        "buffer_batches": 32,
        "sae_batch_size": 4096,
        "lr": 3e-4,
        "dataset": "monology/pile-uncopyrighted",
        "model_name": "gelu-1l",
        "hook_name": "blocks.0.mlp.hook_post",
    },
    "model": {
        "type": "hodge2d",
        "expansion_factors": [6, 3, 1],
        "reg_chain": 1e-3,
        "reg_harm": 1e-3,
        "reg_boundary": 1e-3,
        "l1_feat": [1e-3, 1e-3],
        "l1_hole": 1e-3,
    },
}

ROOT_DIR = Path(__file__).resolve().parent.parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_REGISTRY = {
    "sae": SparseAutoEncoder,
    "hodge2d": TwoDimensionalHodgeSparseAutoEncoder,
    "hodge": HodgeSparseAutoEncoder,
}


def build_model(model_cfg: dict, input_dim: int) -> torch.nn.Module:
    cfg = model_cfg.copy()
    model_type = cfg.pop("type")
    factors = cfg.pop("expansion_factors")

    if model_type == "hodge2d":
        cfg["latent_dim_0"] = input_dim * factors[0]
        cfg["latent_dim_1"] = input_dim * factors[1]
        cfg["latent_dim_2"] = input_dim * factors[2]
    elif model_type == "hodge":
        cfg["latent_dims_list"] = [input_dim * f for f in factors]
    elif model_type == "sae":
        cfg["latent_dim"] = input_dim * factors[0]

    return MODEL_REGISTRY[model_type](input_dim=input_dim, **cfg)


def get_token_batch(model, data_iter, batch_size, seq_len):
    texts = [next(data_iter)["text"] for _ in range(batch_size)]
    tokens = model.tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding="max_length",
    )["input_ids"].to(DEVICE)
    return tokens

def save_model(sae: torch.nn.Module, cfg: dict, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file(sae.state_dict(), save_dir / "model.safetensors")
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"Modelo guardado en {save_dir}")

def main(cfg: dict):
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    model = HookedTransformer.from_pretrained(train_cfg["model_name"], device=DEVICE)
    model.eval()
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    d_in = model.cfg.d_mlp
    sae = build_model(model_cfg, input_dim=d_in).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=train_cfg["lr"])

    dataset = load_dataset(train_cfg["dataset"], split="train", streaming=True)
    data_iter = iter(dataset)
    tokens_seen = 0

    run_name = (
        f"{model_cfg['type']}_ch{model_cfg['reg_chain']}"
        f"_hm{model_cfg['reg_harm']}_hl{model_cfg['l1_hole']}"
        f"_{train_cfg['model_name']}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({**train_cfg, **model_cfg, "d_in": d_in})
        try:
            while tokens_seen < train_cfg["total_tokens"]:
                buffer = []
                with torch.no_grad():
                    for _ in range(train_cfg["buffer_batches"]):
                        tokens = get_token_batch(
                            model, data_iter, train_cfg["batch_size"], train_cfg["seq_length"]
                        )
                        _, cache = model.run_with_cache(
                            tokens,
                            names_filter=lambda name: name == train_cfg["hook_name"],
                            stop_at_layer=1,
                        )
                        acts = cache[train_cfg["hook_name"]].reshape(-1, d_in)
                        buffer.append(acts)

                buffer = torch.cat(buffer, dim=0)
                perm = torch.randperm(buffer.shape[0], device=DEVICE)
                buffer = buffer[perm]

                for i in range(0, buffer.shape[0], train_cfg["sae_batch_size"]):
                    batch = buffer[i : i + train_cfg["sae_batch_size"]]

                    loss, metrics, x_hat, features = sae.compute_loss(batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    sae.normalize_decoder_weights()
                    tokens_seen += batch.shape[0]

                mlflow.log_metrics(metrics, step=tokens_seen)
                logger.info(
                    f"Tokens: {tokens_seen:,} | "
                    f"MSE: {metrics['loss/mse']:.4f} | "
                    f"L0: {metrics['metrics/l0']:.1f} | "
                    f"Holes L0: {metrics['metrics/l0_hole']:.1f}"
                )

        except KeyboardInterrupt:
            logger.info("Entrenamiento interrumpido manualmente. Guardando checkpoint...")
        finally:
            save_dir = ROOT_DIR / "storage" / "models" / f"{run_name}_{run.info.run_id[:8]}"
            save_model(sae, cfg, save_dir)
            mlflow.log_artifacts(str(save_dir), artifact_path="model")


if __name__ == "__main__":
    main(config)