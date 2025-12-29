"""
Generator and discriminator models for CLIP-aligned captioning.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders_text import PromptConfig, tokenize_batch


class SharedEncoder(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(512 + 512, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, ctx_images_bkchw, ctx_caps_list: List[List[str]], clip_encoder, max_ctx_words: int):
        bsz, k, c, h, w = ctx_images_bkchw.shape
        imgs = ctx_images_bkchw.reshape(bsz * k, c, h, w)

        with torch.no_grad():
            e_img = clip_encoder.image_emb(imgs)

        flat_caps = []
        for caps in ctx_caps_list:
            for ccap in caps:
                words = ccap.split()
                flat_caps.append(" ".join(words[:max_ctx_words]))

        with torch.no_grad():
            e_txt = clip_encoder.text_emb(flat_caps)

        h_fuse = self.fuse(torch.cat([e_img, e_txt], dim=-1)).reshape(bsz, k, -1)
        h_fuse = self.temporal(h_fuse)
        return h_fuse.mean(dim=1)


class PrefixGPT2(nn.Module):
    def __init__(self, gpt_model, d_z: int = 256, n_prefix: int = 8):
        super().__init__()
        self.gpt = gpt_model
        self.n_prefix = n_prefix
        self.z_to_prefix = nn.Linear(d_z, n_prefix * self.gpt.config.n_embd)

    def forward(self, input_ids, attention_mask, labels, z, device: torch.device):
        bsz, _ = input_ids.shape
        tok_emb = self.gpt.transformer.wte(input_ids.to(device))
        prefix = self.z_to_prefix(z).view(bsz, self.n_prefix, -1)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)

        prefix_mask = torch.ones((bsz, self.n_prefix), dtype=attention_mask.dtype, device=device)
        attn = torch.cat([prefix_mask, attention_mask.to(device)], dim=1)

        prefix_labels = torch.full((bsz, self.n_prefix), -100, dtype=labels.dtype, device=device)
        lab = torch.cat([prefix_labels, labels.to(device)], dim=1)

        out = self.gpt(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=lab,
            use_cache=False,
        )
        return out.loss


class GeneratorG(nn.Module):
    def __init__(
        self,
        gpt_model,
        clip_encoder,
        tokenizer,
        prompt_cfg: PromptConfig,
        d_model: int = 256,
        n_prefix: int = 8,
    ):
        super().__init__()
        self.encoder = SharedEncoder(d_model=d_model)
        self.textdec = PrefixGPT2(gpt_model, d_z=d_model, n_prefix=n_prefix)
        self.img_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 512))
        self.clip = clip_encoder
        self.tokenizer = tokenizer
        self.prompt_cfg = prompt_cfg

    def forward(self, batch, device: torch.device):
        ctx_images = batch["ctx_images"].to(device)
        tgt_image = batch["tgt_image"].to(device)

        z = self.encoder(ctx_images, batch["ctx_captions"], self.clip, self.prompt_cfg.max_ctx_words)

        enc = tokenize_batch(self.tokenizer, batch["ctx_captions"], batch["tgt_caption"], self.prompt_cfg)
        text_loss = self.textdec(enc["input_ids"], enc["attention_mask"], enc["labels"], z, device)

        with torch.no_grad():
            tgt_img_emb = self.clip.image_emb(tgt_image)

        pred_img_emb = F.normalize(self.img_head(z), dim=-1)
        img_loss = F.mse_loss(pred_img_emb, tgt_img_emb)

        with torch.no_grad():
            gt_txt_emb = self.clip.text_emb(batch["tgt_caption"])
        align_loss = (1.0 - (pred_img_emb * gt_txt_emb).sum(dim=-1)).mean()

        return text_loss, img_loss, align_loss, pred_img_emb, z


class DPair(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, img_emb_512, txt_emb_512):
        x = torch.cat([img_emb_512, txt_emb_512], dim=-1)
        return self.net(x).squeeze(-1)
