from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from markdown_chunker import chunk_markdown


def main() -> None:
    sample_dir = Path("data/goodwiki_markdown_sample")
    if not sample_dir.exists():
        print("Sample directory not found:", sample_dir)
        return

    markdown_files = sorted(sample_dir.glob("*.md"))
    if not markdown_files:
        print("No markdown files found in:", sample_dir)
        return

    sample_file = markdown_files[0]
    chunks = chunk_markdown(sample_file, max_tokens=400, overlap_ratio=0.1, debug=True)

    print("\nFirst chunk preview:\n")
    print(chunks[0]["text"][:800])


if __name__ == "__main__":
    main()
