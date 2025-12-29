"""
Evaluation helpers for CLIPScore and discriminator accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from .dataloader import build_dataloaders, build_splits, filter_k_plus_1, load_storyreasoning, setup_cache_dirs
from .encoders_image import ClipEncoder
from .encoders_text import PromptConfig, build_tokenizer, format_prompt
from .model_dual_decoders import DPair, GeneratorG


@torch.no_grad()
def generate_caption_one(G: GeneratorG, tok, ctx_images_1k, ctx_caps_1k, device, max_new_tokens: int = 48) -> str:
    G.eval()
    z = G.encoder(ctx_images_1k.to(device), [ctx_caps_1k], G.clip, G.prompt_cfg.max_ctx_words)

    prompt = format_prompt(ctx_caps_1k, G.prompt_cfg.max_ctx_words)
    inp = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    pid = inp["input_ids"]
    am = inp["attention_mask"]

    prefix = G.textdec.z_to_prefix(z).view(1, G.textdec.n_prefix, -1)
    tok_emb = G.textdec.gpt.transformer.wte(pid)
    inputs_embeds = torch.cat([prefix, tok_emb], dim=1)

    prefix_mask = torch.ones((1, G.textdec.n_prefix), dtype=am.dtype, device=device)
    am2 = torch.cat([prefix_mask, am], dim=1)

    out_ids = G.textdec.gpt.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=am2,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    full = tok.decode(out_ids[0], skip_special_tokens=True)
    pred = full.split("NEXT:", 1)[-1].strip() if "NEXT:" in full else full.replace(prompt, "", 1).strip()
    return pred if pred.strip() else " "


@torch.no_grad()
def clipscore_generated(G: GeneratorG, dl, device, max_new_tokens: int = 48, n_batches=None):
    G.eval()
    scores = []
    total = len(dl) if n_batches is None else min(n_batches, len(dl))
    it = iter(dl)
    pbar = tqdm(range(total), desc="CLIPScore(gen)")
    for _ in pbar:
        batch = next(it)
        bs = batch["tgt_image"].shape[0]
        tgt_img_emb = G.clip.image_emb(batch["tgt_image"].to(device))

        preds = []
        for i in range(bs):
            pred = generate_caption_one(G, G.tokenizer, batch["ctx_images"][i : i + 1], batch["ctx_captions"][i], device, max_new_tokens)
            preds.append(pred if pred.strip() else " ")

        pred_txt_emb = G.clip.text_emb(preds)
        cos = (tgt_img_emb * pred_txt_emb).sum(dim=-1).detach().cpu().numpy()
        scores.extend(cos.tolist())
        pbar.set_postfix(mean=float(np.mean(scores)))
    return float(np.mean(scores)), float(np.std(scores))


@torch.no_grad()
def discriminator_accuracy(G: GeneratorG, D: DPair, dl, device, n_batches: int = 100, max_items: int = 800):
    G.eval()
    D.eval()
    correct = 0
    total = 0

    it = iter(dl)
    steps = min(n_batches, len(dl))
    pbar = tqdm(range(steps), desc="D accuracy")
    for _ in pbar:
        batch = next(it)
        bs = batch["tgt_image"].shape[0]

        tgt_img_emb = G.clip.image_emb(batch["tgt_image"].to(device))
        real_txt_emb = G.clip.text_emb(batch["tgt_caption"])

        preds = []
        for i in range(bs):
            pred = generate_caption_one(G, G.tokenizer, batch["ctx_images"][i : i + 1], batch["ctx_captions"][i], device, max_new_tokens=32)
            preds.append(pred if pred.strip() else " ")
        fake_txt_emb = G.clip.text_emb(preds)

        d_real = D(tgt_img_emb, real_txt_emb).detach().cpu().numpy()
        d_fake = D(tgt_img_emb, fake_txt_emb).detach().cpu().numpy()

        correct += int((d_real > 0).sum())
        correct += int((d_fake < 0).sum())
        total += 2 * bs
        pbar.set_postfix(acc=correct / max(total, 1))
        if total >= max_items:
            break

    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    caches = setup_cache_dirs(project_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_hf, _ = load_storyreasoning(caches.hf_datasets_cache)
    train_ok = filter_k_plus_1(train_hf, args.k)
    run_dir = caches.data_dir / "runs" / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_idx, val_idx, test_idx = build_splits(train_ok, args.k, args.seed, run_dir)
    _, dl_val, dl_test = build_dataloaders(train_ok, train_idx, val_idx, test_idx, k=args.k, batch_size=args.batch)

    clip = ClipEncoder.from_pretrained("openai/clip-vit-base-patch32", caches.transformers_cache, device)
    tok = build_tokenizer("gpt2", caches.transformers_cache)
    gpt = AutoModelForCausalLM.from_pretrained(
        "gpt2", cache_dir=str(caches.transformers_cache), use_safetensors=True
    ).to(device)

    prompt_cfg = PromptConfig()
    G = GeneratorG(gpt, clip, tok, prompt_cfg, d_model=256, n_prefix=8).to(device)
    D = DPair().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "G" in ckpt:
        G.load_state_dict(ckpt["G"], strict=True)
    if "D" in ckpt:
        D.load_state_dict(ckpt["D"], strict=True)

    val_mean, val_std = clipscore_generated(G, dl_val, device, n_batches=None)
    test_mean, test_std = clipscore_generated(G, dl_test, device, n_batches=None)
    val_dacc = discriminator_accuracy(G, D, dl_val, device)
    test_dacc = discriminator_accuracy(G, D, dl_test, device)

    results = {
        "val_clipscore_gen_mean": val_mean,
        "val_clipscore_gen_std": val_std,
        "test_clipscore_gen_mean": test_mean,
        "test_clipscore_gen_std": test_std,
        "val_D_accuracy": val_dacc,
        "test_D_accuracy": test_dacc,
    }

    out_path = run_dir / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
