from __future__ import annotations

from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import random
import re
import shutil

# -------- CONFIG --------
DATASET_NAME = "euirim/goodwiki"
SPLIT = "train"
OUTPUT_DIR = Path("data/goodwiki_markdown")
SAMPLE_DIR = Path("data/goodwiki_markdown_sample")
MAX_DOCS = None  # e.g. 500 for testing
SAMPLE_SIZE = 500  # adjust for testing
random.seed(42)
# ------------------------


def clean_filename(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name[:120]


def save_markdown(title: str, markdown: str, out_dir: Path) -> None:
    filename = clean_filename(title) + ".md"
    path = out_dir / filename

    # Markdown is already clean -> just wrap title
    content = f"# {title}\n\n{markdown.strip()}\n"
    path.write_text(content, encoding="utf-8")


def download_goodwiki() -> int:
    print(f"Loading dataset: {DATASET_NAME} ({SPLIT})")
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for item in tqdm(ds):
        title = item["title"]
        markdown = item["markdown"]

        if not markdown or not markdown.strip():
            continue

        save_markdown(title, markdown, OUTPUT_DIR)

        count += 1
        if MAX_DOCS and count >= MAX_DOCS:
            break

    print(f"Saved {count} markdown files to {OUTPUT_DIR}")
    return count


def sample_goodwiki() -> int:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    all_files = list(OUTPUT_DIR.glob("*.md"))
    if SAMPLE_SIZE > len(all_files):
        sample_size = len(all_files)
    else:
        sample_size = SAMPLE_SIZE

    sampled_files = random.sample(all_files, sample_size)
    for f in sampled_files:
        shutil.copy(f, SAMPLE_DIR / f.name)

    print(f"Sampled {sample_size} files to {SAMPLE_DIR}")
    return sample_size


def main() -> None:
    download_goodwiki()
    sample_goodwiki()


if __name__ == "__main__":
    main()
