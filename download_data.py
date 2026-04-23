"""
CausalRec-Bench — Download Data and Models
Downloads all files from Hugging Face.

Usage:
    python download_data.py
"""

import os, sys, subprocess

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','huggingface_hub','--quiet'])
    from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "alihassan1437/causalrec-bench"
REPO_TYPE = "dataset"

print("=" * 60)
print("CausalRec-Bench — Download Data and Models")
print("=" * 60)
print(f"Source: https://huggingface.co/datasets/{REPO_ID}")
print()

os.makedirs('data', exist_ok=True)
os.makedirs('pretrained_models', exist_ok=True)

print("Fetching file list...")
all_files = list(list_repo_files(REPO_ID, repo_type=REPO_TYPE))
data_files = [f for f in all_files if f.startswith('data/')]
model_files = [f for f in all_files if f.startswith('pretrained_models/')]

print(f"  Data files:   {len(data_files)}")
print(f"  Model files:  {len(model_files)}")
print()

def download_file(repo_file):
    local_path = repo_file
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
        print(f"  SKIP {repo_file} (exists)")
        return
    print(f"  Downloading {repo_file}...", end='', flush=True)
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_file,
        repo_type=REPO_TYPE,
        local_dir='.'
    )
    size = os.path.getsize(local_path)/1024/1024
    print(f" {size:.0f} MB OK")

print("Downloading data files...")
for f in sorted(data_files):
    download_file(f)

print()
print("Downloading pretrained models...")
for f in sorted(model_files):
    download_file(f)

print()
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)
print()
print("Run evaluation:")
print("  python benchmark/run_evaluation.py")
