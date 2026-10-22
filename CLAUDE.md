# CLAUDE.md

Guidance for AI-assisted work on this repository. See README.md for the full write-up and results.

## Project in one paragraph

Four-class classification of oral histopathology patches (OSCC, leukoplakia with dysplasia, leukoplakia without dysplasia, Normal) with ConvNeXt-Tiny. It also tests whether MedSAM segmentation before classification helps. Data:
- **NDB-UFES:** 3,763 patches of 512×512 with official folds.
- **Normal:** 290 images from Rahman et al., tiled into 2,863 patches of 512×512.
- **Whole-image pipeline:** the 237 NDB-UFES source images (2048×1536, 69 patients) plus 1,224 Rahman images (290 Normal, 934 OSCC in `dataset_3_rahman_oscc/`), 1,461 in total.

Both datasets are CC BY 4.0.

## Environment

- **Local laptop:** Windows, Git Bash, Intel i5 CPU, 7.7 GB RAM, **no NVIDIA GPU** (PyTorch is the CPU build). Use it for data preparation, small checks, Grad-CAM and figures. Do **not** run MedSAM or full training locally.
- **GPU work runs on Kaggle (T4)** through `notebooks/*.ipynb`. The notebooks `git clone` this repo, so **code changes must be pushed before they take effect on Kaggle**.
- **Data on Kaggle:** a private Kaggle Dataset containing `splits/` and `preprocessed_normalized/`, created by `scripts/data/make_kaggle_upload.py`.

## Commands

Shared code lives in the `oral_cancer/` package: `config`, `model` (ConvNeXt, transforms, GradCAM), `stain` (Macenko), `tissue` (tiling and tissue mask), `segmenter` (U-Net epithelium segmenter) and `predictor`. Scripts import from it; don't duplicate it.

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

Whole-image pipeline (segmentation → ROI → classification):
1. Locally: `create_whole_image_splits`, then `make_whole_image_upload`
2. On Kaggle, `notebooks/roi_classification_kaggle.ipynb`: `train_segmenter` → `segment_whole_images` → `tile_whole_images` → `train_classifier` (2 settings × 5 rounds, runs `roi_<all|roi>_round<k>`) → `summarize_roi`
3. Locally: `make_readme_figures --only epithelium_segmentation segmenter_dice roi_results roi_by_source`

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
  - Whole images use `splits/whole_image_folds.csv`: patient-level 5 folds (NDB grouped by patient; each Rahman image is its own group). Round k: test = fold k, validation = fold (k mod 5) + 1, train = the rest. The segmenter uses the same rounds, and each whole image is segmented by the round model that held out its fold.
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
  - **After retraining on sharpness-matched data:** Normal F1 is still 98.9% (98.3% in greyscale), and its Grad-CAM border share only dropped from about 65% to about 60%. The model still uses image-level source cues, most likely brightness (Normal ~155 vs ~141) and contrast. The top rows of Normal tiles are not brighter in the pixels, so the edge focus is a representation effect, not an artefact.
  - Decision (user): do not chase this further. The **4-class model is the headline** (92.2 ± 1.1% balanced accuracy). The comparison with published work uses the 4-class model's score on the three NDB-UFES classes (90.0 ± 1.7% vs 91.9%). **The README must not mention the separate 3-class model**, although its runs and code still exist.
  - Pre-fix runs are kept locally in `runs_before_blur/` (git-ignored) for before/after figures.
- **NDB-UFES folds are patch-level**, not patient-level, so results are optimistic. This includes published benchmarks; the main one is Maia et al. 2024 with DenseNet-121 at 91.9% balanced accuracy.
- **SAM/MedSAM at 1024px uses about 0.8 GB of GPU memory per image** in one attention tensor. Keep the batch size at 2 or less and use fp16 on a T4.
- **Tissue detection on the Normal images needs per-image optical density.** Their glass background has strong colour casts, which break fixed colour thresholds.
- **Epithelium masks** (`annotations/epithelium/masks/`, 50 whole images) are MedSAM-assisted annotations made by Claude, **not expert-verified**. Say so whenever Dice/IoU is reported.
- **Epithelium segmenter works; the ROI does not help classification.**
  - U-Net (ResNet-34) Dice 0.847 vs 0.798 all-tissue vs 0.779 whole image (10x: 0.81 / 0.77 / 0.71). Weakest on Rahman Normal at 10x.
  - Whole-image balanced accuracy: all-tissue tiles 71.7 ± 4.9%, ROI tiles 70.4 ± 1.4% (macro-F1 67.3 vs 64.3); the two agree on 92% of images.
  - **Source shortcut:** NDB images are never predicted Normal, Rahman images almost never leukoplakia, and 45% of NDB OSCC are called leukoplakia. NDB-only balanced accuracy is ~50–55% with patient-level folds, vs 90% on the patch folds.
  - Tiling blurs **all** Rahman tiles (Normal and OSCC), by source rather than class.
- **Magnification is not a shortcut:** about half of every class is 10x and half 40x (`check_magnification.py`).

## Current status (2026-10-22)

**Done:**
- data pipeline, splits, tiling, stain normalisation, colour-bias check;
- MedSAM on the full dataset;
- ConvNeXt 4-class, 3-class and greyscale runs, before and after sharpness matching;
- **multi-seed results** (seeds 1–3, sharpness-matched data): 3-class **89.7 ± 1.4%** balanced accuracy (primary; Maia 2024: 91.9%), 4-class 92.2 ± 1.1% (NDB-only 90.0 ± 1.7%, Normal F1 99.3%), greyscale 90.5 ± 1.4%. Seed spread is 1–2 points SD; single seed-42 runs are used only for the confusion-matrix, curve and Grad-CAM figures;
- Grad-CAM and the border check;
- README figures;
- prediction tool: `oral_cancer/predictor.py`, CLI `python -m scripts.inference.predict <image>`, web demo `streamlit run app/streamlit_app.py --server.address localhost`. Verified to reproduce the test-set predictions exactly;
- whole-image pipeline: patient-level folds, magnification check, 50 epithelium annotations (annotator: `streamlit run app/epithelium_annotator.py --server.address localhost`), U-Net segmenter, segmentation → ROI → classification comparison. Results are in the README; local copies in `runs/segmenter`, `runs/roi_*`.

**Next:**
1. Final report.

A 384px experiment was considered and **dropped by the user**: 224px is the standard input size of the pretrained model. Do not propose it again or list it as future work.

For any new training setting, use the multi-seed pattern: `notebooks/convnext_multiseed_kaggle.ipynb` with run folders `<setting>_seed<N>`, then `python -m scripts.classification.summarize_seeds`.

## Working agreements with the user

- **Git:** commit when asked. **Push only when explicitly asked.** Commits are authored by the user: **never add `Co-Authored-By: Claude` or any AI attribution** to commit messages or PR descriptions.
- **Compute:** ask before running anything heavy on the laptop. Quick checks are fine (smoke tests, Grad-CAM on a few images).
- **Licensing:** both datasets are CC BY 4.0. Credit both when showing images.
