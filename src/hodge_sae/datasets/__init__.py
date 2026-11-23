from .base import ActivationBatch, ActivationDataLoader
from .text import TextDataLoader, PileDataLoader
from .cache import CachedActivationDataLoader
from .arithmetic import ArithmeticDataLoader


DATASET_REGISTRY: dict[str, type[ActivationDataLoader]] = {
    "pile": PileDataLoader,
    "arithmetic": ArithmeticDataLoader,
    "text": TextDataLoader,
    "cache": CachedActivationDataLoader,
}

def build_dataloader(data_cfg: dict, device: str) -> ActivationDataLoader:
    cfg = data_cfg.copy()
    dataset_type = cfg.pop("type")

    if dataset_type not in DATASET_REGISTRY:
        valid = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Dataset '{dataset_type}' does not exist. Options: {valid}")

    return DATASET_REGISTRY[dataset_type](device=device, **cfg)