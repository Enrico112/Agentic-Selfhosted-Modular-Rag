from pathlib import Path
import random
import shutil

# -------- CONFIG --------
SOURCE_DIR = Path("data/goodwiki_markdown")
SAMPLE_DIR = Path("data/goodwiki_markdown_sample")
SAMPLE_SIZE = 500  # adjust for testing
random.seed(42)
# ------------------------

# Ensure output directory exists
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# List all markdown files
all_files = list(SOURCE_DIR.glob("*.md"))
if SAMPLE_SIZE > len(all_files):
    SAMPLE_SIZE = len(all_files)

# Randomly sample files
sampled_files = random.sample(all_files, SAMPLE_SIZE)

# Copy sampled files
for f in sampled_files:
    shutil.copy(f, SAMPLE_DIR / f.name)

print(f"Sampled {SAMPLE_SIZE} files to {SAMPLE_DIR}")