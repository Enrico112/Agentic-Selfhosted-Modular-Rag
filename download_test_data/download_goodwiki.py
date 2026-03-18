from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import re

# -------- CONFIG --------
DATASET_NAME = "euirim/goodwiki"
SPLIT = "train"
OUTPUT_DIR = Path("data/goodwiki_markdown")
MAX_DOCS = None  # e.g. 500 for testing
# ------------------------

def clean_filename(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name[:120]

def save_markdown(title: str, markdown: str, out_dir: Path):
    filename = clean_filename(title) + ".md"
    path = out_dir / filename

    # Markdown is already clean → just wrap title
    content = f"# {title}\n\n{markdown.strip()}\n"
    path.write_text(content, encoding="utf-8")


def main():
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


if __name__ == "__main__":
    main()