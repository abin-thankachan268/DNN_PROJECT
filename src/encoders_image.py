"""
CLIP encoder helpers for image and text embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


@dataclass
class ClipEncoder:
    model: CLIPModel
    processor: CLIPProcessor
    device: torch.device

    @classmethod
    def from_pretrained(cls, model_name: str, cache_dir: Path, device: torch.device):
        model = CLIPModel.from_pretrained(model_name, cache_dir=str(cache_dir), use_safetensors=True).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name, cache_dir=str(cache_dir))
        for p in model.parameters():
            p.requires_grad = False
        return cls(model=model, processor=processor, device=device)

    @property
    def mean(self):
        return torch.tensor([0.48145466, 0.45782750, 0.40821073], device=self.device).view(1, 3, 1, 1)

    @property
    def std(self):
        return torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def image_emb(self, img_bchw_01: torch.Tensor) -> torch.Tensor:
        x = (img_bchw_01.to(self.device) - self.mean) / self.std
        e = self.model.get_image_features(pixel_values=x)
        return F.normalize(e, dim=-1)

    @torch.no_grad()
    def text_emb(self, text_list: Iterable[str]) -> torch.Tensor:
        inputs = self.processor(text=list(text_list), return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        e = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        return F.normalize(e, dim=-1)
