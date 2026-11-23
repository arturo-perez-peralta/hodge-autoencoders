from .base import BaseSparseAutoEncoder
from .hodge_sae import HodgeSparseAutoEncoder, TwoDimensionalHodgeSparseAutoEncoder
from .sae import SparseAutoEncoder


MODEL_REGISTRY: dict[str, type[BaseSparseAutoEncoder]] = {
    "hodge": HodgeSparseAutoEncoder,
    "hodge2d": TwoDimensionalHodgeSparseAutoEncoder,
    "sae": SparseAutoEncoder,
}

def build_model(model_cfg: dict, device: str) -> BaseSparseAutoEncoder:
    cfg = model_cfg.copy()
    model_type = cfg.pop("type")

    if model_type not in MODEL_REGISTRY:
        valid = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_type}' does not exist. Options: {valid}")

    return MODEL_REGISTRY[model_type](device=device, **cfg)