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

import torch
from torch import nn
import torch.nn.functional as F

class TwoDimensionalHodgeSparseAutoEncoder(nn.Module):

    def __init__(
            self,
            input_dim: int, 
            latent_dim_0: int,
            latent_dim_1: int,
            latent_dim_2: int,
            reg_chain: float,
            reg_harm: float,
            reg_boundary: float,
            l1_feat: list[float],
            l1_hole: float
            ):
        super().__init__() 
        self.input_dim = input_dim
        
        self.latent_dim_0 = latent_dim_0
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2

        self.reg_chain = reg_chain
        self.reg_harm = reg_harm
        self.reg_boundary = reg_boundary
        self.l1_feat = l1_feat
        self.l1_hole = l1_hole

        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        self.encoders = nn.ModuleDict({
            '0': nn.Linear(input_dim, latent_dim_0),
            '1': nn.Linear(input_dim, latent_dim_1),
            '2': nn.Linear(input_dim, latent_dim_2),
        })
        self.decoders = nn.ModuleDict({
            '0': nn.Linear(latent_dim_0, input_dim, bias=False),
            '1': nn.Linear(latent_dim_1, input_dim, bias=False),
            '2': nn.Linear(latent_dim_2, input_dim, bias=False),
        })
        self.boundaries = nn.ModuleDict({
            '0': nn.Linear(latent_dim_0, latent_dim_1, bias=False),
            '2': nn.Linear(latent_dim_2, latent_dim_1, bias=False),
        })
        self.normalize_decoder_weights()

    def forward(self, x: torch.Tensor):

        x_cent = x - self.b_dec
        
        f0 = F.relu(self.encoders['0'](x_cent))
        f2 = F.relu(self.encoders['2'](x_cent))
        
        holes = F.relu(self.encoders['1'](x_cent))
        
        f1 = holes + self.boundaries['0'](f0) + self.boundaries['2'](f2)

        features = [f0, f1, f2]

        x_hat = (
            self.decoders['0'](f0)
            + self.decoders['1'](f1)
            + self.decoders['2'](f2)
            + self.b_dec
        )

        return x_hat, features, holes

    def compute_topological_losses(self, holes: torch.Tensor):
        # \partial_1
        d1 = self.boundaries['0'].weight.T
        # \partial_2
        d2 = self.boundaries['2'].weight

        # 1. Boundary law: \partial_1 * \partial_2 = 0
        chain_loss = torch.norm(d1 @ d2, p='fro') ** 2

        # 2. Harmonic orthogonality: \partial_1 * holes = 0  y  \partial_2^T * holes = 0
        # holes: (batch, S_1)
        harm_down = torch.mean(torch.sum((holes @ d1.T) ** 2, dim=-1))
        harm_up = torch.mean(torch.sum((holes @ d2) ** 2, dim=-1))
        harmonic_loss = harm_down + harm_up

        return chain_loss, harmonic_loss

    def compute_loss(self, x: torch.Tensor):
        x_hat, features, holes = self(x)
        chain_loss, harm_loss = self.compute_topological_losses(holes)

        mse_loss = F.mse_loss(x_hat, x)

        f0, f2 = features[0], features[2]

        l1_f0_raw = f0.abs().sum(dim=-1).mean()
        l1_f2_raw = f2.abs().sum(dim=-1).mean()
        l1_hole_raw = holes.abs().sum(dim=-1).mean()

        l1_z_loss = self.l1_feat[0] * l1_f0_raw + self.l1_feat[1] * l1_f2_raw
        l1_hole_loss = self.l1_hole * l1_hole_raw
        boundary_l1 = sum(b.weight.abs().mean() for b in self.boundaries.values())

        loss = (
            mse_loss
            + self.reg_chain * chain_loss
            + self.reg_harm * harm_loss
            + l1_z_loss
            + l1_hole_loss
            + self.reg_boundary * boundary_l1
        )

        l0_f0 = (f0 > 0).float().sum(dim=-1).mean().item()
        l0_f2 = (f2 > 0).float().sum(dim=-1).mean().item()
        l0_hole = (holes > 0).float().sum(dim=-1).mean().item()

        metrics = {
            "loss/total": loss.item(),
            "loss/mse": mse_loss.item(),
            "loss/chain": chain_loss.item(),
            "loss/harm": harm_loss.item(),
            "loss/l1_z": l1_z_loss.item(),
            "loss/l1_hole": l1_hole_loss.item(),
            "loss/l1_boundary": boundary_l1.item(),
            "metrics/l0": l0_f0 + l0_f2 + l0_hole,
            "metrics/l0_z": l0_f0 + l0_f2,
            "metrics/l0_hole": l0_hole,
            "l0/z_dim_0": l0_f0,
            "l0/z_dim_2": l0_f2,
            "l0/hole_dim_1": l0_hole,
            "l1/z_dim_0": l1_f0_raw.item(),
            "l1/z_dim_2": l1_f2_raw.item(),
            "l1/hole_dim_1": l1_hole_raw.item(),
        }

        feat = {"features": features, "holes": holes, "z": [f0, f2]}

        return loss, metrics, x_hat, feat

    @torch.no_grad()
    def normalize_decoder_weights(self):
        for _, decoder in self.decoders.items():
            decoder.weight.data = F.normalize(decoder.weight.data, p=2, dim=0)
        for boundary in self.boundaries.values():
            boundary.weight.data = F.normalize(boundary.weight.data, p=2, dim=0)



class HodgeSparseAutoEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            latent_dims_list: list[int],
            reg_chain: float,
            reg_harm: float,
            l1_feat: list[float],
            l1_hole: list[float],
            reg_boundary: float
            ):
        super().__init__()
        assert len(latent_dims_list) >= 2, "Need at least two latent dimensions."

        self.input_dim = input_dim
        self.latent_dims_list = latent_dims_list
        self.latent_dim_total = sum(latent_dims_list)
        self.n_latent = len(latent_dims_list)

        self.reg_chain = reg_chain
        self.reg_harm = reg_harm
        self.l1_feat = l1_feat
        self.l1_hole = l1_hole
        self.reg_boundary = reg_boundary

        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        self.z_encoders = nn.ModuleList([
            nn.Linear(input_dim, dim) for dim in latent_dims_list
        ])
        self.harm_encoders = nn.ModuleList([
            nn.Linear(input_dim, dim) for dim in latent_dims_list
        ])
        self.decoders = nn.ModuleList([
            nn.Linear(dim, input_dim, bias=False) for dim in latent_dims_list
        ])

        # Boundary operator
        self.boundaries = nn.ParameterList([
            nn.Parameter(torch.randn(latent_dims_list[k], latent_dims_list[k + 1]) * 0.02)
            for k in range(self.n_latent - 1)
        ])

        self.normalize_decoder_weights()

    def apply_dirac(self, v_list: list[torch.Tensor]) -> list[torch.Tensor]:
        out = [torch.zeros_like(v) for v in v_list]
        for k, d in enumerate(self.boundaries):
            out[k] = out[k] + F.linear(v_list[k + 1], d)
            out[k + 1] = out[k + 1] + F.linear(v_list[k], d.T)
        return out

    def forward(self, x: torch.Tensor):
        x_cent = x - self.b_dec

        z = [F.relu(enc(x_cent)) for enc in self.z_encoders]
        holes = [F.relu(enc(x_cent)) for enc in self.harm_encoders]

        bz = self.apply_dirac(z)
        features = [bz_k + h_k for bz_k, h_k in zip(bz, holes)]

        x_hat = sum(dec(f_k) for dec, f_k in zip(self.decoders, features)) + self.b_dec

        return x_hat, features, holes, z

    def compute_topological_losses(self, holes: list[torch.Tensor]):
        if len(self.boundaries) > 1:
            chain_loss = sum(
                torch.norm(self.boundaries[k] @ self.boundaries[k + 1], p='fro') ** 2
                for k in range(len(self.boundaries) - 1)
            )
        else:
            chain_loss = torch.tensor(0.0, device=holes[0].device)

        b_holes = self.apply_dirac(holes)
        harmonic_loss = sum(
            torch.mean(torch.sum(bh_k ** 2, dim=-1))
            for bh_k in b_holes
        )

        return chain_loss, harmonic_loss

    def compute_loss(self, x: torch.Tensor):

        x_hat, features, holes, z = self(x)
        chain_loss, harm_loss = self.compute_topological_losses(holes)

        mse_loss = F.mse_loss(x_hat, x)

        l1_z_raw = [z_k.abs().sum(dim=-1).mean() for z_k in z]
        l1_hole_raw = [h_k.abs().sum(dim=-1).mean() for h_k in holes]

        l1_z_loss = sum(self.l1_feat[k] * l1_z_raw[k] for k in range(self.n_latent))
        l1_hole_loss = sum(self.l1_hole[k] * l1_hole_raw[k] for k in range(self.n_latent))
        boundary_l1 = sum(d.abs().mean() for d in self.boundaries)

        loss = (
            mse_loss
            + self.reg_chain * chain_loss
            + self.reg_harm * harm_loss
            + l1_z_loss
            + l1_hole_loss
            + self.reg_boundary * boundary_l1
        )

        l0_z = [(z_k > 0).float().sum(dim=-1).mean().item() for z_k in z]
        l0_hole = [(h_k > 0).float().sum(dim=-1).mean().item() for h_k in holes]

        metrics = {
            "loss/total": loss.item(),
            "loss/mse": mse_loss.item(),
            "loss/chain": chain_loss.item(),
            "loss/harm": harm_loss.item(),
            "loss/l1_z": l1_z_loss.item(),
            "loss/l1_hole": l1_hole_loss.item(),
            "loss/l1_boundary": boundary_l1.item(),
            "metrics/l0": sum(l0_z) + sum(l0_hole),
            "metrics/l0_z": sum(l0_z),
            "metrics/l0_hole": sum(l0_hole),
        }

        for k in range(self.n_latent):
            metrics[f"l0/z_dim_{k}"] = l0_z[k]
            metrics[f"l0/hole_dim_{k}"] = l0_hole[k]
            metrics[f"l1/z_dim_{k}"] = l1_z_raw[k].item()
            metrics[f"l1/hole_dim_{k}"] = l1_hole_raw[k].item()

        feat = {"features": features, "holes": holes, "z": z}

        return loss, metrics, x_hat, feat

    @torch.no_grad()
    def normalize_decoder_weights(self):
        for decoder in self.decoders:
            decoder.weight.data = F.normalize(decoder.weight.data, p=2, dim=0)
        for boundary in self.boundaries:
            boundary.data = F.normalize(boundary.data, p=2, dim=0)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
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
        "l1_hole": 1e-4,
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
            mlflow.log_param("model_path", str(save_dir))


if __name__ == "__main__":
    main(config)