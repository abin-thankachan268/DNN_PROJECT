# DNNLS StoryReasoning GAN (CLIP + GPT-2)

This project trains a CLIP-aligned caption generator that predicts the next frame caption from K context frames in the StoryReasoning dataset. It combines a shared image+text encoder, a GPT-2 prefix decoder, and a discriminator operating in CLIP embedding space for adversarial alignment.

## Key components

- **Dataset:** StoryReasoning (K+1 framing, story-level splits)
- **Encoders:** CLIP image/text embeddings (frozen)
- **Generator:** Transformer-based context encoder + GPT-2 prefix decoder
- **Discriminator:** Pairwise CLIP embedding classifier (hinge loss)
- **Losses:** text LM loss + image embedding loss + text-image alignment + optional adversarial loss

## Repo layout

- `src/dataloader.py` � dataset parsing, splits, and PyTorch loaders
- `src/encoders_image.py` � CLIP embedding utilities
- `src/encoders_text.py` � prompt formatting + tokenization
- `src/model_dual_decoders.py` � generator + discriminator modules
- `src/train.py` � warmup and adversarial training loops
- `src/eval_alignment.py` � CLIPScore + discriminator accuracy
- `src/download_dataset.py` � dataset fetch into local cache
- `notebooks/notebook.ipynb` � full experiment notebook
- `data/` � local caches, runs, and outputs

## Setup

Create and activate a venv, then install requirements:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install datasets transformers accelerate evaluate tqdm pandas numpy pillow matplotlib scikit-learn wordcloud
```

## Download dataset

```bash
python -m src.download_dataset
```

## Train (warmup)

```bash
python -m src.train --mode warmup --k 4 --batch 2 --warmup_epochs 2
```

## Train (adversarial)

```bash
python -m src.train --mode adv --k 4 --batch 2 --adv_epochs 2
```

## Evaluate

```bash
python -m src.eval_alignment --checkpoint data\runs\<run_id>\GAN_final.pt
```

## Notes

- All caches and outputs are stored under `data/`.
- Use `--unfreeze_gpt` in `src/train.py` if you want to fine-tune GPT-2.
