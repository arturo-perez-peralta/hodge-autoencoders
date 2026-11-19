import torch
from torch import nn
import torch.nn.functional as F

class SparseAutoEncoder(nn.Module):

    def __init__(self, input_dim: int, dict_size: int, l1_pen: float = 1e-3):
        super().__init__() 
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim)
        self.l1_pen = l1_pen
        self.normalize_decoder_weights()

    def forward(self, x: torch.Tensor):
        features = F.relu(self.encoder(x - self.decoder.bias))
        x_hat = self.decoder(features)
        return x_hat, features

    def loss(self, x: torch.Tensor):
        x_hat, features = self(x)

        mse_loss = F.mse_loss(x_hat, x)
        l1_loss = self.l1_pen * features.abs().sum(dim=-1).mean()

        loss = mse_loss + l1_loss

        metrics = {
            "loss/total": loss.item(),
            "loss/mse": mse_loss.item(),
            "loss/l1": l1_loss.item(),
            "metrics/l0": (features > 0).float().sum(dim=-1).mean().item(),
        }

        feat = {'features': features}

        return loss, metrics, x_hat, feat

    @torch.no_grad()
    def normalize_decoder_weights(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

