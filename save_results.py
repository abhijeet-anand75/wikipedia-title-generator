# =============================================================================
# save_results.py
# Run this after training to save all results to Google Drive.
# Prevents losing results when Colab session disconnects.
# =============================================================================

import os
import shutil
import time

DRIVE_BACKUP = "/content/drive/MyDrive/wikipedia-title-generator-results"
os.makedirs(DRIVE_BACKUP, exist_ok=True)

files_to_save = {
    "/content/results_all.json"  : "results_all.json",
    "/content/vocab.pkl"         : "vocab.pkl",
    "/content/best_rnn_model.pt" : "best_rnn_model.pt",
    "/content/preprocessed_train.csv"      : "preprocessed_train.csv",
    "/content/preprocessed_validation.csv" : "preprocessed_validation.csv",
    "/content/preprocessed_test.csv"       : "preprocessed_test.csv",
}

print("Saving results to Google Drive...")
print(f"Backup folder: {DRIVE_BACKUP}")

for src, filename in files_to_save.items():
    if os.path.exists(src):
        dst = os.path.join(DRIVE_BACKUP, filename)
        shutil.copy(src, dst)
        size = os.path.getsize(dst)
        print(f"  ✓ Saved: {filename} ({size:,} bytes)")
    else:
        print(f"  ✗ Not found: {filename} — skipping")

# Save T5 model folder if it exists
t5_src = "/content/t5-title-gen"
t5_dst = os.path.join(DRIVE_BACKUP, "t5-title-gen")
if os.path.exists(t5_src):
    shutil.copytree(t5_src, t5_dst, dirs_exist_ok=True)
    print(f"  ✓ Saved: t5-title-gen/")
else:
    print(f"  ✗ Not found: t5-title-gen/ — skipping")

print(f"\nAll results saved to Google Drive")
print(f"Location: {DRIVE_BACKUP}")