# =============================================================================
# setup_colab.py
# Run this FIRST on every new Colab session before anything else.
#
# What it does:
#   1. Mounts Google Drive
#   2. Clones / pulls latest code from GitHub
#   3. Downloads dataset files from Google Drive using gdown
#   4. Installs all dependencies
#   5. Verifies GPU is available
#   6. Verifies all files are in place
# =============================================================================

import os
import subprocess

# =============================================================================
# STEP 1 — Mount Google Drive
# =============================================================================
print("=" * 60)
print("STEP 1 — Mounting Google Drive")
print("=" * 60)

from google.colab import drive
drive.mount('/content/drive')
print("Google Drive mounted successfully")

# =============================================================================
# STEP 2 — Clone or Pull GitHub Repository
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2 — Setting up GitHub Repository")
print("=" * 60)

GITHUB_REPO = "https://github.com/abhijeet-anand75/wikipedia-title-generator.git"
PROJECT_DIR = "/content/wikipedia-title-generator"

if os.path.exists(PROJECT_DIR):
    # Repo already cloned — just pull latest changes
    print("Repository already exists — pulling latest changes...")
    os.chdir(PROJECT_DIR)
    os.system("git pull origin main")
else:
    # Fresh clone
    print("Cloning repository...")
    os.system(f"git clone {GITHUB_REPO}")
    os.chdir(PROJECT_DIR)

print(f"Working directory: {os.getcwd()}")
print("Files in project:")
print(os.listdir(PROJECT_DIR))

# =============================================================================
# STEP 3 — Install Dependencies
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3 — Installing Dependencies")
print("=" * 60)

os.system("pip install -q -r requirements.txt")
print("Dependencies installed")

# =============================================================================
# STEP 4 — Download Dataset Files from Google Drive
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4 — Downloading Dataset Files")
print("=" * 60)

os.system("pip install -q gdown")
import gdown

# File IDs from your Google Drive share links
files_to_download = {
    "train.csv"          : "1cYBtTR0d6Iv15giUQIySY7MfBZXVCEXC",
    "test.csv"           : "1cVHnB0Hz6wXUpzgVV3Z2iqfFWSTtb7UI",
    "glove.6B.300d.txt"  : "1cTO_lr8-ttiQbjb5Na7TXd51M6kgpxV_",
}

for filename, file_id in files_to_download.items():
    output_path = f"/content/{filename}"

    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"  {filename} already exists ({size:,} bytes) — skipping")
    else:
        print(f"  Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print(f"  {filename} downloaded successfully")

# =============================================================================
# STEP 5 — Verify GPU
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5 — Verifying GPU")
print("=" * 60)

import torch
if torch.cuda.is_available():
    gpu_name   = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU available  : YES")
    print(f"  GPU name       : {gpu_name}")
    print(f"  GPU memory     : {gpu_memory:.1f} GB")
else:
    print("  WARNING: No GPU detected!")
    print("  Go to Runtime → Change runtime type → GPU")

# =============================================================================
# STEP 6 — Verify All Files Present
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6 — Verifying All Files")
print("=" * 60)

required_files = [
    "/content/train.csv",
    "/content/test.csv",
    "/content/glove.6B.300d.txt",
    "/content/wikipedia-title-generator/config.py",
    "/content/wikipedia-title-generator/utils.py",
    "/content/wikipedia-title-generator/preprocess.py",
    "/content/wikipedia-title-generator/train_rnn.py",
    "/content/wikipedia-title-generator/train_transformer.py",
    "/content/wikipedia-title-generator/models/encoder.py",
    "/content/wikipedia-title-generator/models/decoder.py",
    "/content/wikipedia-title-generator/models/seq2seq.py",
    "/content/wikipedia-title-generator/data/dataset.py",
    "/content/wikipedia-title-generator/evaluation/metrics.py",
]

all_good = True
for filepath in required_files:
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"  ✓ {filepath.split('/')[-1]} ({size:,} bytes)")
    else:
        print(f"  ✗ MISSING: {filepath}")
        all_good = False

print()
if all_good:
    print("All files verified. Ready to run!")
else:
    print("Some files are missing. Check errors above.")

# =============================================================================
# STEP 7 — Add project to Python path
# =============================================================================
import sys
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
print(f"Project path added: {PROJECT_DIR}")

print("\n" + "=" * 60)
print("SETUP COMPLETE — You can now run:")
print("  !python preprocess.py")
print("  !python train_rnn.py")
print("  !python train_transformer.py")
print("=" * 60)