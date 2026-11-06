import torch.nn as nn
from peft import LoraConfig, get_peft_model
from .vit_dinov3 import vit_dinov3


class ViTWithLoRA(nn.Module):
    """Wrapper for DINOv3 ViT with LoRA adapters.

    FIXED: Now inherits from nn.Module so .train()/.eval()/.to() propagate correctly!
    """

    def __init__(self, checkpoint_path, lora_r=16, device="cuda"):
        super().__init__()

        self.vit = vit_dinov3(
            checkpoint_path=checkpoint_path,
            device=device
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=32,
            target_modules=["qkv", "proj"],
            lora_dropout=0.1,
            bias="none"
        )

        self.vit.dinov3 = get_peft_model(
            self.vit.dinov3,
            lora_config
        )

    def encode(self, frames):
        """Encode frames to embeddings."""
        return self.vit.encode(frames)

    @property
    def vision_encoder(self):
        return self.vit.dinov3

    @property
    def dtype(self):
        return self.vit.dtype

    @property
    def device(self):
        return self.vit.device

    def get_trainable_parameters(self):
        return [p for p in self.vit.dinov3.parameters() if p.requires_grad]