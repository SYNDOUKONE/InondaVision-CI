from .unet import UNet
from .deeplabv3 import DeepLabV3Plus
from .segformer import SegFormerFlood  # noqa: F401  (import optionnel)

MODEL_REGISTRY = {
    "unet": UNet,
    "deeplabv3": DeepLabV3Plus,
    "segformer": SegFormerFlood,
}


def build_model(name: str, in_channels: int = 2, num_classes: int = 3, **kwargs):
    try:
        cls = MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Modèle inconnu: {name!r}. Choix disponibles: {sorted(MODEL_REGISTRY)}"
        ) from None
    return cls(in_channels=in_channels, num_classes=num_classes, **kwargs)
