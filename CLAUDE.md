# CLAUDE.md

Guidance for AI-assisted work on this repository. See README.md for the full write-up and results.

## Project in one paragraph

Four-class classification of oral histopathology patches (OSCC, leukoplakia with dysplasia, leukoplakia without dysplasia, Normal) with ConvNeXt-Tiny. It also tests whether MedSAM segmentation before classification helps. Data:
- **NDB-UFES:** 3,763 patches of 512×512 with official folds.
- **Normal:** 290 images from Rahman et al., tiled into 2,863 patches of 512×512.

Both datasets are CC BY 4.0.

## Environment

- **Local laptop:** Windows, Git Bash, Intel i5 CPU, 7.7 GB RAM, **no NVIDIA GPU** (PyTorch is the CPU build). Use it for data preparation, small checks, Grad-CAM and figures. Do **not** run MedSAM or full training locally.
- **GPU work runs on Kaggle (T4)** through `notebooks/*.ipynb`. The notebooks `git clone` this repo, so **code changes must be pushed before they take effect on Kaggle**.
- **Data on Kaggle:** a private Kaggle Dataset containing `splits/` and `preprocessed_normalized/`, created by `scripts/data/make_kaggle_upload.py`.

## Commands

Always run scripts **as modules from the project root**:

```bash
python -m scripts.data.preprocess
python -m scripts.classification.train_classifier --smoke-test   # fast CPU check of the training code
python -m scripts.classification.gradcam --border-check
python -m scripts.reporting.make_readme_figures                  # regenerate docs/images/
```

Pipeline order:
1. `create_master_dataset`
2. `create_splits`
3. `preprocess`
4. `stain_normalize`
5. Then on Kaggle: `auto_segment` / `train_classifier`
6. Then locally: `gradcam`, then `make_readme_figures`

## Conventions

- **Paths:** never hard-code absolute paths. Import from `oral_cancer.config`:
  - `DATA_ROOT` is where inputs are read; `OUTPUT_ROOT` is where outputs are written. Both default to the project root; override with `ORAL_CANCER_DATA_ROOT` and `ORAL_CANCER_OUTPUT_ROOT`.
  - CSVs store **relative POSIX paths**. Use `resolve()` to read a CSV path and `relative()` to write one.
  - Create directories with `mkdir(parents=True, exist_ok=True)`.
- **Code style:** match the existing scripts. Use `# ====` section banners (PATHS / CONFIGURATION / ...), a module docstring explaining *why*, and argparse for options.
- **Splits are fixed:**
  - NDB-UFES uses the official folds: `folder` 1–4 train, 5 validation, 6 test.
  - Normal is split **by source image before tiling**.
  - Never re-split randomly by patch.
  - `folder` is a **CV fold, not a patient ID**; the patch metadata has no patient IDs.
- **Evaluation:** select models on validation macro-F1; evaluate the test set once. Report balanced accuracy and macro-F1, not just accuracy.
- **Generated data is git-ignored:** datasets, `splits/`, `preprocessed*/`, `segmentation/`, `runs/`, `outputs/` and model weights. Only `docs/images/` figures are committed (whitelisted in `.gitignore`).

## Known findings and pitfalls (do not rediscover)

- **MedSAM gives no localisation benefit.** Masks cover a median of 95–98% of each patch, because patches are already lesion crops. Classification uses whole patches. Never use class-dependent prompts, since that leaks the label.
- **Normal is a cross-dataset class.**
  - Colour-only AUC is 0.999 before and 0.974 after Macenko normalisation.
  - In greyscale, Normal F1 is still about 99%.
  - Grad-CAM puts about 65% of Normal heatmaps on the patch border, versus about 50% for the other classes. This happens for edge and interior tiles alike.
  - **Confirmed cause: sharpness.** Laplacian variance at 224px alone separates Normal with AUC 0.946; Normal is about 5x sharper. Brightness (0.83) and contrast (0.85) also differ.
  - **Fix:** `preprocess.py` blurs Normal tiles (`TILE_BLUR_SIGMA = 2.5`, calibrated with `check_sharpness --calibrate-blur`), which brings the sharpness AUC to about 0.50. Training adds random blur.
  - Treat Normal results as optimistic. The 3-class NDB-UFES result is the primary benchmark.
- **NDB-UFES folds are patch-level**, not patient-level, so results are optimistic. This includes published benchmarks; the main one is Maia et al. 2024 with DenseNet-121 at 91.9% balanced accuracy.
- **SAM/MedSAM at 1024px uses about 0.8 GB of GPU memory per image** in one attention tensor. Keep the batch size at 2 or less and use fp16 on a T4.
- **Tissue detection on the Normal images needs per-image optical density.** Their glass background has strong colour casts, which break fixed colour thresholds.

## Current status (2026-10-13)

**Done:**
- data pipeline, splits, tiling, stain normalisation, colour-bias check;
- MedSAM on the full dataset;
- ConvNeXt 4-class, 3-class and greyscale runs (single seed);
- Grad-CAM and the border check;
- README figures.

**Next:**
1. Retrain on the sharpness-matched data (re-upload the Kaggle dataset first) and compare Normal accuracy with the pre-fix runs.
2. Multi-seed runs (mean ± SD); a 384px experiment.
3. Inference script / demo; final report.

## Working agreements with the user

- **Git:** commit when asked. **Push only when explicitly asked.** Commits are authored by the user: **never add `Co-Authored-By: Claude` or any AI attribution** to commit messages or PR descriptions.
- **Compute:** ask before running anything heavy on the laptop. Quick checks are fine (smoke tests, Grad-CAM on a few images).
- **Licensing:** both datasets are CC BY 4.0. Credit both when showing images.
