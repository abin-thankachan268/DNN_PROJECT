"""
Prompt formatting and tokenization helpers for GPT-2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from transformers import AutoTokenizer


@dataclass
class PromptConfig:
    max_len: int = 256
    max_tgt_tok: int = 64
    max_ctx_words: int = 25


def build_tokenizer(model_name: str, cache_dir, pad_with_eos: bool = True):
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    if pad_with_eos and tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def compress_caption(caption: str, max_words: int) -> str:
    if not caption:
        return ""
    words = caption.split()
    return " ".join(words[:max_words])


def format_prompt(ctx_caps: List[str], max_ctx_words: int) -> str:
    lines = [f"{i+1}: {compress_caption(c, max_ctx_words)}" for i, c in enumerate(ctx_caps)]
    return "CTX\n" + "\n".join(lines) + "\nNEXT:"


def tokenize_batch(tok, ctx_caps_batch: Iterable[List[str]], tgt_caps_batch: Iterable[str], cfg: PromptConfig):
    input_ids_list = []
    labels_list = []

    for ctx_caps, tgt in zip(ctx_caps_batch, tgt_caps_batch):
        prompt = format_prompt(ctx_caps, cfg.max_ctx_words)
        tgt = tgt if isinstance(tgt, str) else ""

        t_ids = tok(tgt, truncation=True, max_length=cfg.max_tgt_tok, add_special_tokens=False)["input_ids"]
        if len(t_ids) == 0:
            t_ids = [tok.eos_token_id]

        prompt_budget = cfg.max_len - len(t_ids)
        if prompt_budget < 1:
            p_ids = []
            t_ids = t_ids[: cfg.max_len]
        else:
            p_ids = tok(prompt, truncation=True, max_length=prompt_budget, add_special_tokens=False)["input_ids"]

        ids = p_ids + t_ids
        labs = ([-100] * len(p_ids)) + t_ids

        input_ids_list.append(torch.tensor(ids, dtype=torch.long))
        labels_list.append(torch.tensor(labs, dtype=torch.long))

    max_b = max(x.size(0) for x in input_ids_list)
    input_ids = torch.full((len(input_ids_list), max_b), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(labels_list), max_b), -100, dtype=torch.long)
    attn_mask = torch.zeros((len(input_ids_list), max_b), dtype=torch.long)

    for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
        n = ids.size(0)
        input_ids[i, :n] = ids
        labels[i, :n] = lab
        attn_mask[i, :n] = 1

    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}
