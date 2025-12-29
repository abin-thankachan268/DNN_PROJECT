"""
Training for CLIP-aligned captioning with warmup + adversarial fine-tune.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from .consistency_module import d_hinge_loss, g_hinge_loss
from .dataloader import build_dataloaders, build_splits, filter_k_plus_1, load_storyreasoning, setup_cache_dirs
from .encoders_image import ClipEncoder
from .encoders_text import PromptConfig, build_tokenizer, format_prompt
from .model_dual_decoders import DPair, GeneratorG


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


def train_warmup_epoch(
    G: GeneratorG,
    dl,
    g_opt,
    scaler: GradScaler,
    device,
    alpha_img: float,
    lambda_align: float,
    train: bool = True,
) -> Dict[str, float]:
    G.train(train)
    total = {"txt": 0.0, "img": 0.0, "align": 0.0, "tot": 0.0}
    n = 0
    pbar = tqdm(dl, desc=("warmup-train" if train else "warmup-val"))
    for batch in pbar:
        with autocast("cuda", enabled=(device.type == "cuda")):
            txt_loss, img_loss, align_loss, _, _ = G(batch, device)
            loss = txt_loss + alpha_img * img_loss + lambda_align * align_loss

        if train:
            g_opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(g_opt)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler.step(g_opt)
            scaler.update()

        bs = batch["tgt_image"].shape[0]
        total["txt"] += float(txt_loss.detach()) * bs
        total["img"] += float(img_loss.detach()) * bs
        total["align"] += float(align_loss.detach()) * bs
        total["tot"] += float(loss.detach()) * bs
        n += bs
        pbar.set_postfix(tot=total["tot"] / n, txt=total["txt"] / n, img=total["img"] / n, al=total["align"] / n)

    for k in total:
        total[k] /= max(n, 1)
    return total


def train_adv_epoch(
    G: GeneratorG,
    D: DPair,
    dl,
    g_opt,
    d_opt,
    scaler: GradScaler,
    device,
    alpha_img: float,
    lambda_align: float,
    beta_adv: float,
    train: bool = True,
) -> Dict[str, float]:
    G.train(train)
    D.train(train)

    meters = {"d": 0.0, "g_adv": 0.0, "txt": 0.0, "img": 0.0, "align": 0.0, "g_tot": 0.0}
    n = 0
    pbar = tqdm(dl, desc=("adv-train" if train else "adv-val"))

    for batch in pbar:
        bs = batch["tgt_image"].shape[0]
        ctx_images = batch["ctx_images"].to(device)
        tgt_image = batch["tgt_image"].to(device)

        with torch.no_grad():
            tgt_img_emb = G.clip.image_emb(tgt_image)
            real_txt_emb = G.clip.text_emb(batch["tgt_caption"])

        if train:
            preds = []
            for i in range(bs):
                pred = generate_caption_one(G, G.tokenizer, ctx_images[i : i + 1], batch["ctx_captions"][i], device)
                preds.append(pred)
            with torch.no_grad():
                fake_txt_emb = G.clip.text_emb(preds)

            d_real = D(tgt_img_emb, real_txt_emb)
            d_fake = D(tgt_img_emb, fake_txt_emb)
            d_loss = d_hinge_loss(d_real, d_fake)

            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            d_opt.step()
        else:
            d_loss = torch.tensor(0.0, device=device)

        with autocast("cuda", enabled=(device.type == "cuda")):
            txt_loss, img_loss, align_loss, _, _ = G(batch, device)

            preds = []
            for i in range(bs):
                pred = generate_caption_one(G, G.tokenizer, ctx_images[i : i + 1], batch["ctx_captions"][i], device)
                preds.append(pred)
            fake_txt_emb = G.clip.text_emb(preds)

            d_fake_for_g = D(tgt_img_emb.detach(), fake_txt_emb)
            g_adv = g_hinge_loss(d_fake_for_g)

            g_total = txt_loss + alpha_img * img_loss + lambda_align * align_loss + beta_adv * g_adv

        if train:
            g_opt.zero_grad(set_to_none=True)
            scaler.scale(g_total).backward()
            scaler.unscale_(g_opt)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler.step(g_opt)
            scaler.update()

        meters["d"] += float(d_loss.detach()) * bs
        meters["g_adv"] += float(g_adv.detach()) * bs
        meters["txt"] += float(txt_loss.detach()) * bs
        meters["img"] += float(img_loss.detach()) * bs
        meters["align"] += float(align_loss.detach()) * bs
        meters["g_tot"] += float(g_total.detach()) * bs
        n += bs

        pbar.set_postfix(d=meters["d"] / n, g_tot=meters["g_tot"] / n, g_adv=meters["g_adv"] / n)

    for k in meters:
        meters[k] /= max(n, 1)
    return meters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["warmup", "adv"], default="warmup")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--adv_epochs", type=int, default=2)
    parser.add_argument("--alpha_img", type=float, default=0.5)
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--beta_adv", type=float, default=0.2)
    parser.add_argument("--unfreeze_gpt", action="store_true", default=False)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    caches = setup_cache_dirs(project_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = caches.data_dir / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_hf, _ = load_storyreasoning(caches.hf_datasets_cache)
    train_ok = filter_k_plus_1(train_hf, args.k)
    train_idx, val_idx, test_idx = build_splits(train_ok, args.k, args.seed, run_dir)
    dl_train, dl_val, _ = build_dataloaders(
        train_ok,
        train_idx,
        val_idx,
        test_idx,
        k=args.k,
        batch_size=args.batch,
    )

    clip = ClipEncoder.from_pretrained("openai/clip-vit-base-patch32", caches.transformers_cache, device)
    tok = build_tokenizer("gpt2", caches.transformers_cache)
    gpt = AutoModelForCausalLM.from_pretrained(
        "gpt2", cache_dir=str(caches.transformers_cache), use_safetensors=True
    ).to(device)

    prompt_cfg = PromptConfig()
    G = GeneratorG(gpt, clip, tok, prompt_cfg, d_model=256, n_prefix=8).to(device)
    D = DPair().to(device)

    if not args.unfreeze_gpt:
        for p in G.textdec.gpt.parameters():
            p.requires_grad = False

    g_params = [p for p in G.parameters() if p.requires_grad]
    g_opt = torch.optim.AdamW(g_params, lr=2e-4)
    d_opt = torch.optim.AdamW(D.parameters(), lr=2e-4)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    history = []
    if args.mode == "warmup":
        for ep in range(1, args.warmup_epochs + 1):
            tr = train_warmup_epoch(G, dl_train, g_opt, scaler, device, args.alpha_img, args.lambda_align, train=True)
            va = train_warmup_epoch(G, dl_val, g_opt, scaler, device, args.alpha_img, args.lambda_align, train=False)
            print(f"Warmup Epoch {ep} | train: {tr} | val: {va}")
            history.append({"epoch": ep, "train": tr, "val": va})

        torch.save({"G": G.state_dict(), "history": history}, run_dir / "G_warmup.pt")
    else:
        for ep in range(1, args.adv_epochs + 1):
            tr = train_adv_epoch(
                G,
                D,
                dl_train,
                g_opt,
                d_opt,
                scaler,
                device,
                args.alpha_img,
                args.lambda_align,
                args.beta_adv,
                train=True,
            )
            va = train_adv_epoch(
                G,
                D,
                dl_val,
                g_opt,
                d_opt,
                scaler,
                device,
                args.alpha_img,
                args.lambda_align,
                args.beta_adv,
                train=False,
            )
            print(f"Adv Epoch {ep} | train: {tr} | val: {va}")
            history.append({"epoch": ep, "train": tr, "val": va})

        torch.save({"G": G.state_dict(), "D": D.state_dict(), "history": history}, run_dir / "GAN_final.pt")

    (run_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("Saved outputs to:", run_dir)


if __name__ == "__main__":
    main()
