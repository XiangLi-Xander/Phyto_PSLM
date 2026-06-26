# Changelog

## 2026-06-26 — Sync local with GitHub

### Removed (deleted from repo)
- `ablation/01_no_esm2/src/` — model.py, train.py, utils.py
- `ablation/02_no_iupred/src/` — model.py, train.py, utils.py
- `ablation/03_onehot_esm/src/` — model.py, train.py, utils.py
- `outputs/genome_predict/gmax_llps_scores.csv`
- `outputs/genome_predict/gmax_score_distribution.png`
- `outputs/genome_predict/slycopersicum_llps_scores.csv`
- `outputs/genome_predict/slycopersicum_score_distribution.png`

### Added
- `scripts/download_models.py` — Automated downloader for ESM2 model and pre-trained checkpoints

### Modified
- `README.md` — Added model download section with script instructions; updated repo structure

### Large files (not tracked in git)
- `esm2/model.safetensors` (~2.5 GB) — ESM2 model weights, download via:
  ```bash
  python scripts/download_models.py
  # or
  huggingface-cli download facebook/esm2_t33_650M_UR50D --local-dir esm2
  ```
- `outputs/models/*.pth` — Pre-trained checkpoints, download separately from GitHub Releases
- `outputs/benchmarks/*.npz` / `*.pkl` — Benchmark features (generated at runtime)
