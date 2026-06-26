"""
Download all required model files for PhytoRNP_PSLM.

Usage:
    python scripts/download_models.py

This will download:
    1. ESM2-t33-650M model weights → esm2/
    2. Pre-trained LLPS checkpoint → outputs/models/best_model.pth
"""

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def download_esm2():
    """Download ESM2-t33-650M model from HuggingFace."""
    target_dir = REPO_ROOT / "esm2"
    target_dir.mkdir(exist_ok=True)

    # Skip if already downloaded
    if (target_dir / "model.safetensors").exists():
        print("[OK] ESM2 model already exists at esm2/model.safetensors")
        return True

    print("Downloading ESM2-t33-650M model from HuggingFace...")
    print("  Model: facebook/esm2_t33_650M_UR50D")
    print(f"  Target: {target_dir}")
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


def download_pretrained_checkpoint(url: str = None):
    """Download the pre-trained LLPS model checkpoint."""
    target_dir = REPO_ROOT / "outputs" / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "best_model.pth"

    if target_path.exists():
        print(f"[OK] Pre-trained checkpoint already exists at {target_path}")
        return True

    if url is None:
        print("[SKIP] No URL provided for pre-trained checkpoint.")
        print("  To download the checkpoint, provide a URL:")
        print("    python scripts/download_models.py --checkpoint-url <URL>")
        print("  Or manually place best_model.pth at: outputs/models/best_model.pth")
        return False

    print(f"Downloading pre-trained checkpoint from {url}...")
    try:
        import requests
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(target_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}%", end="")
        print("\n[OK] Pre-trained checkpoint downloaded successfully")
        return True
    except ImportError:
        print("[ERROR] requests not installed. Run: pip install requests")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to download checkpoint: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download model files for PhytoRNP_PSLM")
    parser.add_argument("--checkpoint-url", type=str, default=None,
                        help="URL to the pre-trained best_model.pth (optional)")
    parser.add_argument("--skip-esm2", action="store_true",
                        help="Skip ESM2 model download")
    args = parser.parse_args()

    print("=" * 60)
    print("  PhytoRNP_PSLM Model Downloader")
    print("=" * 60)

    if not args.skip_esm2:
        print("\n[1/2] ESM2-t33-650M model")
        print("-" * 40)
        download_esm2()

    print("\n[2/2] Pre-trained LLPS checkpoint (best_model.pth)")
    print("-" * 40)
    download_pretrained_checkpoint(args.checkpoint_url)

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
