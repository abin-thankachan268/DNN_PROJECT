"""
Dataset helpers for StoryReasoning K+1 captioning with CLIP/GPT models.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json
import os
import random
import re

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
import datasets

_GDI = re.compile(r"<gdi\s+image(\d+)>", re.IGNORECASE)
_GD_TAGS = re.compile(r"</?gd[ioal]\b[^>]*>", re.IGNORECASE)
_ANY_TAGS = re.compile(r"<[^>]+>")


@dataclass
class CacheDirs:
    data_dir: Path
    hf_home: Path
    hf_datasets_cache: Path
    hf_hub_cache: Path
    transformers_cache: Path
    torch_home: Path


def setup_cache_dirs(project_dir: Path) -> CacheDirs:
    data_dir = project_dir / "data"
    hf_home = data_dir / "hf_home"
    hf_datasets_cache = data_dir / "hf_datasets"
    hf_hub_cache = data_dir / "hf_hub"
    transformers_cache = data_dir / "transformers"
    torch_home = data_dir / "torch"

    for p in [data_dir, hf_home, hf_datasets_cache, hf_hub_cache, transformers_cache, torch_home]:
        p.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["XDG_CACHE_HOME"] = str(data_dir / "xdg_cache")

    return CacheDirs(
        data_dir=data_dir,
        hf_home=hf_home,
        hf_datasets_cache=hf_datasets_cache,
        hf_hub_cache=hf_hub_cache,
        transformers_cache=transformers_cache,
        torch_home=torch_home,
    )


def strip_grounding_tags(text: str) -> str:
    if text is None:
        return ""
    text = _GD_TAGS.sub("", text)
    text = _ANY_TAGS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_story_into_frame_segments(story_text: str) -> Dict[int, str]:
    if not story_text:
        return {}
    matches = list(_GDI.finditer(story_text))
    if not matches:
        return {}
    segs: Dict[int, str] = {}
    for i, match in enumerate(matches):
        frame_idx = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(story_text)
        segs[frame_idx] = story_text[start:end].strip()
    return segs


def get_frame_captions(example: Dict) -> List[str]:
    frame_count = int(example.get("frame_count", 0))
    segs = split_story_into_frame_segments(example.get("story", ""))
    captions = []
    for i in range(1, frame_count + 1):
        captions.append(strip_grounding_tags(segs.get(i, "")))
    return captions


def load_storyreasoning(cache_dir: Path):
    datasets.config.HF_DATASETS_CACHE = str(cache_dir)
    ds = load_dataset("daniel3303/StoryReasoning", cache_dir=str(cache_dir))
    return ds["train"], ds["test"]


def filter_k_plus_1(hf_ds, k: int):
    def has_k_plus_1(ex):
        return int(ex["frame_count"]) >= (k + 1)

    return hf_ds.filter(has_k_plus_1)


def build_splits(train_hf, k: int, seed: int, run_dir: Path) -> Tuple[List[int], List[int], List[int]]:
    idxs = list(range(len(train_hf)))
    story_ids: List[str] = []
    frame_counts: List[int] = []
    for i in idxs:
        ex = train_hf[i]
        story_ids.append(ex["story_id"])
        frame_counts.append(int(ex["frame_count"]))

    df_ids = pd.DataFrame({"idx": idxs, "story_id": story_ids, "frame_count": frame_counts})
    df_ids["len_bin"] = pd.cut(
        df_ids["frame_count"],
        bins=[0, 7, 11, 15, 100],
        labels=["5-7", "8-11", "12-15", "16+"],
    )

    df_ids = df_ids.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for _, group in df_ids.groupby("len_bin", observed=False):
        g = group["idx"].tolist()
        n = len(g)
        n_train = int(round(0.80 * n))
        n_val = int(round(0.10 * n))
        train_idx += g[:n_train]
        val_idx += g[n_train : n_train + n_val]
        test_idx += g[n_train + n_val :]

    split_path = run_dir / "splits.json"
    split_path.write_text(
        json.dumps({"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx, "K": k}, indent=2),
        encoding="utf-8",
    )
    return train_idx, val_idx, test_idx


class StoryKPlus1(Dataset):
    def __init__(self, hf_ds, indices: List[int], k: int = 4, img_tf=None):
        self.hf_ds = hf_ds
        self.indices = indices
        self.k = k
        self.img_tf = img_tf

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, object]:
        ex = self.hf_ds[self.indices[i]]
        caps = get_frame_captions(ex)
        imgs = ex["images"]

        ctx_imgs = imgs[: self.k]
        tgt_img = imgs[self.k]

        if self.img_tf is not None:
            ctx_imgs = torch.stack([self.img_tf(im.convert("RGB")) for im in ctx_imgs], dim=0)
            tgt_img = self.img_tf(tgt_img.convert("RGB"))

        return {
            "story_id": ex["story_id"],
            "frame_count": int(ex["frame_count"]),
            "ctx_images": ctx_imgs,
            "ctx_captions": caps[: self.k],
            "tgt_image": tgt_img,
            "tgt_caption": caps[self.k],
        }


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    ctx_images = torch.stack([b["ctx_images"] for b in batch], dim=0)
    tgt_image = torch.stack([b["tgt_image"] for b in batch], dim=0)
    return {
        "story_id": [b["story_id"] for b in batch],
        "frame_count": torch.tensor([b["frame_count"] for b in batch]),
        "ctx_images": ctx_images,
        "tgt_image": tgt_image,
        "ctx_captions": [b["ctx_captions"] for b in batch],
        "tgt_caption": [b["tgt_caption"] for b in batch],
    }


def build_dataloaders(
    train_hf,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    k: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    img_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    ds_train = StoryKPlus1(train_hf, train_idx, k=k, img_tf=img_tf)
    ds_val = StoryKPlus1(train_hf, val_idx, k=k, img_tf=img_tf)
    ds_test = StoryKPlus1(train_hf, test_idx, k=k, img_tf=img_tf)

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return dl_train, dl_val, dl_test
