from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor, nn



@dataclass
class SAEOutput:
    x_hat: Float[Tensor, "batch d_in"] 
    features: Tensor | list[Tensor]
    aux: dict[str, Any] = field(default_factory = dict)

@dataclass
class LossOutput:
    loss: Float[Tensor, ""]
    metrics: dict[str, float]

class BaseSparseAutoEncoder(ABC, nn.Module):
    input_dim: int

    @abstractmethod
    def forward(self, x: Float[Tensor, "batch d_in"]) -> SAEOutput:
        ...

    @abstractmethod
    def compute_loss(
        self,
        x: Float[Tensor, "batch d_in"],
        x_hat: Float[Tensor, "batch d_in"],
        output: SAEOutput
    ) -> LossOutput:
        ...