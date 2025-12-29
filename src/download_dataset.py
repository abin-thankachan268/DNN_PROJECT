"""
Download StoryReasoning dataset into local cache folders.
"""
from __future__ import annotations

from pathlib import Path

from .dataloader import load_storyreasoning, setup_cache_dirs


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    caches = setup_cache_dirs(project_dir)
    train_hf, test_hf = load_storyreasoning(caches.hf_datasets_cache)
    print("Downloaded dataset to:", caches.hf_datasets_cache)
    print("Train size:", len(train_hf), "Test size:", len(test_hf))


if __name__ == "__main__":
    main()
