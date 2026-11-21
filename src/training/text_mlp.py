import os

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

import mlflow
import logging
import logging_loki

from src.model.hodge_sae import TwoDimensionalHodgeSparseAutoEncoder, HodgeSparseAutoEncoder
from src.model.sae import SparseAutoEncoder

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

# --- Configuración ---

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
        "l1_feat": [1e-3, 1e-3],
        "l1_hole": 1e-3,
    },
}

model = HookedTransformer.from_pretrained(config['model']['type'], device=DEVICE)
model.eval()
D_IN = model.cfg.d_mlp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
run_name = f"{config['model']['type']}_ch{config['model']['reg_chain']}_hm{config['model']['reg_harm']}_hl{config['model']['reg_hole']}_dt{config['train']['dataset']}_layer{config['train']['model_name']}"

def get_token_batch(data_iter, batch_size, seq_len):
    texts = [next(data_iter)["text"] for _ in range(batch_size)]
    tokens = model.tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding="max_length",
    )["input_ids"].to(DEVICE)
    return tokens

MODEL_REGISTRY = {
    "sae": StandardSAE,
    "hodge2d": TwoDimensionalHodgeSparseAutoEncoder,
    "hodge": HodgeSparseAutoEncoder,
}

def build_model(model_cfg: dict, input_dim: int) -> torch.nn.Module:
    cfg = model_cfg.copy()
    model_type = cfg.pop("type")
    return MODEL_REGISTRY[model_type](input_dim=input_dim, **cfg)

def main(cfg: dict):

    sae = build_model(config["model"], input_dim=D_IN).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)



    dataset = load_dataset(DATASET, split="train", streaming=True)

    data_iter = iter(dataset)
    tokens = get_token_batch(data_iter, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH)
    tokens_seen = 0

    while tokens_seen < TOTAL_TOKENS:
        # 1. Llenar el buffer acumulando activaciones sin gradientes
        buffer = []
        with torch.no_grad():
            for _ in range(BUFFER_BATCHES):
                tokens = get_token_batch(data_iter, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH)
                _, cache = MODEL.run_with_cache(
                    tokens,
                    names_filter=lambda name: name == HOOK_NAME,
                    stop_at_layer=1,
                )
                # Aplanar [batch, seq, d_mlp] -> [batch * seq, d_mlp]
                acts = cache[HOOK_NAME].reshape(-1, D_IN)
                buffer.append(acts)

        # Concatenar y mezclar (shuffle) todos los vectores del buffer
        buffer = torch.cat(buffer, dim=0)
        perm = torch.randperm(buffer.shape[0], device=DEVICE)
        buffer = buffer[perm]

        # 2. Entrenar el SAE recorriendo el buffer mezclado
        for i in range(0, buffer.shape[0], SAE_BATCH_SIZE):
            batch = buffer[i : i + SAE_BATCH_SIZE]

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