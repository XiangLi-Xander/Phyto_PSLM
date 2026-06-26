"""
Download all required model files for PhytoRNP_PSLM.

Usage:
    python scripts/download_models.py

This will download:
    1. ESM2-t33-650M model weights           → esm2/
    2. Benchmark features (esm_iupred)       → outputs/benchmarks/
    3. Benchmark RF model (handcrafted)      → outputs/benchmarks/
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ========= CONFIG =========
# After uploading files to your HuggingFace repo, set the ID here:
HF_REPO_ID = "XiangLi-Xander/Phyto_PSLM_assets"  # <-- CHANGE THIS
# ==========================

def download_hf_file(repo_id: str, filename: str, target_path: Path, description: str, size_hint: str = ""):
    """Download a single file from a HuggingFace repo."""
    if target_path.exists():
        print(f"[OK] {description} already exists at {target_path}")
        return True

    print(f"Downloading {description}...")
    if size_hint:
        print(f"  Size: {size_hint}")

    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_path.parent),
            local_dir_use_symlinks=False,
        )
        print(f"[OK] Downloaded to {target_path}")
        return True
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        return False


def download_esm2():
    """Download ESM2-t33-650M model from HuggingFace."""
    target_dir = REPO_ROOT / "esm2"
    target_dir.mkdir(exist_ok=True)

    if (target_dir / "model.safetensors").exists():
        print("[OK] ESM2 model already exists at esm2/model.safetensors")
        return True

    print("Downloading ESM2-t33-650M model from HuggingFace...")
    print("  Model: facebook/esm2_t33_650M_UR50D")
    print("  Size: ~2.5 GB, this may take a while...\n")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="facebook/esm2_t33_650M_UR50D",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print("[OK] ESM2 model downloaded successfully")
        return True
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to download ESM2 model: {e}")
        return False


def download_benchmark_assets():
    """Download benchmark files from user's HuggingFace repo."""
    assets = [
        ("esm_iupred_features.npz", "outputs/benchmarks/esm_iupred_features.npz", "ESM+IUPred features (422 MB)"),
        ("llps_rf_handcrafted.pkl", "outputs/benchmarks/llps_rf_handcrafted.pkl", "RF handcrafted model (126 MB)"),
    ]

    all_ok = True
    for filename, rel_path, desc in assets:
        ok = download_hf_file(HF_REPO_ID, filename, REPO_ROOT / rel_path, desc)
        if not ok:
            all_ok = False

    if not all_ok:
        print(f"\n[NOTE] To fix, upload files to HuggingFace:")
        print(f"  1. Create a repo at https://huggingface.co/new")
        print(f"  2. Upload the files from your local machine:")
        for filename, rel_path, desc in assets:
            src = REPO_ROOT / rel_path
            print(f"        {src}")
        print(f"  3. Set HF_REPO_ID in scripts/download_models.py")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Download model files for PhytoRNP_PSLM")
    parser.add_argument("--skip-esm2", action="store_true", help="Skip ESM2 model download")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmark assets download")
    args = parser.parse_args()

    print("=" * 60)
    print("  PhytoRNP_PSLM Model Downloader")
    print("=" * 60)

    if not args.skip_esm2:
        print("\n[1/3] ESM2-t33-650M model")
        print("-" * 40)
        download_esm2()

    if not args.skip_benchmarks:
        print("\n[2/3] Benchmark assets (from HuggingFace)")
        print("-" * 40)
        download_benchmark_assets()

    print("\n[3/3] Pre-trained model checkpoints")
    print("-" * 40)
    print("  The .pth model files are already included in the GitHub repo.")
    print("  They are located under outputs/models/, outputs/ablation/*/models/")

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
