import os

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

import mlflow
import logging
import logging_loki

from src.model.hodge_sae import TwoDimensionalHodgeSparseAutoEncoder

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("hodge-sae-experiments")

# --- Configuración ---
TOTAL_TOKENS = 50_000_000   
BATCH_SIZE = 32         
SEQ_LENGTH = 128               
BUFFER_BATCHES = 32          
SAE_BATCH_SIZE = 4096        
L1_COEFF = 1e-3               

DATASET = "monology/pile-uncopyrighted"
MODEL = "gelu-1l"
HOOK_NAME = "blocks.0.mlp.hook_post"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = HookedTransformer.from_pretrained(MODEL, device=DEVICE)
MODEL.eval()

LAYER = 1
D_IN = MODEL.cfg.d_mlp
D_SAE = D_IN * 10

D_SAE_0 = D_IN * 6
D_SAE_1 = D_IN * 3
D_SAE_2 = D_IN * 1

REG_CHAIN = 1e-3
REG_HARM = 1e-3
REG_FEAT = [1e-3, 1e-3, 1e-3]
REG_HOLE = 1e-3


sae = TwoDimensionalHodgeSparseAutoEncoder(
    input_dim = D_IN,
    latent_dim_0 = D_SAE_0,
    latent_dim_1 = D_SAE_1,
    latent_dim_2 = D_SAE_2,
    reg_chain = REG_CHAIN,
    reg_harm = REG_HARM,
    l1_feat = REG_FEAT,
    l1_hole = REG_HOLE,
).to(DEVICE)
optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)



dataset = load_dataset(DATASET, split="train", streaming=True)


def get_token_batch(data_iter, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH):
    texts = [next(data_iter)["text"] for _ in range(batch_size)]
    tokens = MODEL.tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding="max_length",
    )["input_ids"].to(DEVICE)
    return tokens

data_iter = iter(dataset)
tokens = get_token_batch(data_iter, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH)

hook_name = "blocks.0.mlp.hook_post"


tokens_seen = 0

while tokens_seen < TOTAL_TOKENS:
    # 1. Llenar el buffer acumulando activaciones sin gradientes
    buffer = []
    with torch.no_grad():
        for _ in range(BUFFER_BATCHES):
            tokens = get_token_batch(data_iter, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH)
            _, cache = MODEL.run_with_cache(
                tokens,
                names_filter=lambda name: name == hook_name,
                stop_at_layer=1,
            )
            # Aplanar [batch, seq, d_mlp] -> [batch * seq, d_mlp]
            acts = cache[hook_name].reshape(-1, D_IN)
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

    # Log rápido cada vez que se vacía un buffer
    print(f"Tokens: {tokens_seen:,} | MSE: {metrics['loss/mse']:.4f} | L0: {metrics['metrics/l0']:.1f}")