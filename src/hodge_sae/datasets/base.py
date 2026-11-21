from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from jaxtyping import Float
from torch import Tensor



@dataclass(frozen=True)
class ActivationBatch:
    acts: Float[Tensor, "batch d_in"]
    metadata: dict[str, Any] | None = None


class ActivationDataLoader(Protocol):
    @property
    def d_in(self) -> int: ...

    def __iter__(self) -> Iterator[ActivationBatch]: ...