# Presenting the Project to the Guide

A simple walkthrough to use while scrolling through the [README](../README.md). It follows the README's order. For each part: **what to show** and **what to say**.

---

## 1-minute summary (say this first)

"Our project classifies microscope images of oral tissue into four groups: **cancer (OSCC)**, **pre-cancer (leukoplakia with dysplasia)**, **leukoplakia without dysplasia**, and **normal tissue**. Our plan was two steps: first find the lesion with a segmentation model (**MedSAM**), then classify it with **ConvNeXt-Tiny**. We built and tested both. Our 4-class classifier reaches **about 92% balanced accuracy (92.2 ± 1.1%, averaged over 3 training runs)**. On the three classes that published papers use, it scores 90.0%, close to the best published result (91.9%). We also discovered that the Normal class, which comes from a different dataset, can be recognised from image-level differences between the two datasets instead of tissue. We investigated and documented this carefully. Finally, we built the full **segmentation → ROI → classification** pipeline on the whole microscope images: our epithelium segmenter works (**Dice 0.85**), but classifying only the segmented region did **not** improve accuracy, and the honest, patient-level whole-image scores are much lower than the patch scores."

---

## Section by section

### 1. Key results (top of README)
**Show:** the results table.
**Say:**
- "Our main result is the **4-class model: 92.2% ± 1.1% balanced accuracy**. Balanced accuracy means it's about 92% right *for each class on average*, which is fairer than plain accuracy when some classes have fewer images."
- "The ± is because we trained the model **3 times with different random starting points (seeds)** and took the average. Single runs varied by about 2 points, so the average is the reliable number."
- "Published papers only use the three NDB-UFES classes, because Normal is our addition. On those three classes our model scores 90.0%, and the best published result on the same official split is 91.9%. So we're at the published level."
- "Normal itself reaches 99%, but we show that this comes partly from differences between the two datasets, not only from the tissue. We explain that in the Grad-CAM part."
- "MedSAM on the patches didn't help, because the patches are already lesion. So we moved segmentation to the **whole images**, trained our own epithelium segmenter (Dice 0.85), and tested segmentation → ROI → classification there. The ROI gave no gain (70.4% vs 71.7%). I'll explain why."

### 2. Pipeline (flowchart)
**Show:** the flowchart.
**Say:** "These are the steps: get the data → split into train/validation/test → cut images into equal tiles → make the colours consistent → segmentation → classification → explain the predictions."

### 3. Datasets
**Show:** the dataset table and the sample-image grid.
**Say:**
- "We used two public datasets. The first one, **NDB-UFES**, has cancer and leukoplakia, 3,763 images from 69 patients. It has no normal tissue, so we took **normal images from a second dataset**."
- "Both are free to use for research (CC BY 4.0)."
- "We split the data into **train** (to learn), **validation** (to choose the best model) and **test** (used only once, at the end). For the first dataset we used the authors' **official split**, so our results are comparable with published papers."

### 4. Preprocessing
**Show:** the preprocessing figure.
**Say:**
- "The normal images were 4 times larger, so we **cut them into tiles** of the same size instead of shrinking them. Shrinking would make cells look smaller, and the model could tell the datasets apart just from cell size."
- "We **removed tiles that were mostly empty glass**."
- "We **standardised the stain colours** (Macenko normalisation), because the two hospitals' slides have different pink/purple shades."
- "We also **blurred the normal tiles slightly**, because they were much sharper than the other dataset. More on that later."

### 5. Segmentation (MedSAM)
**Show:** the mask-coverage chart and the example masks.
**Say:**
- "We ran MedSAM, a medical segmentation model, on all 6,626 images using Kaggle's free GPU."
- "**Finding:** MedSAM marked about **95–98% of every image** as the region of interest. In other words, it just selected the whole image."
- "**Reason:** the dataset images are already cropped from *inside* the lesion. There's no separate 'lesion region' to find; the whole image is lesion."
- "So we classify the whole image. This is an honest negative result: we tested the idea, and it doesn't add value for this kind of data."
- "A useful segmentation target is the **epithelium** (the surface tissue layer where dysplasia is judged). We did that on the whole images; that's section 8."

### 6. Classification (ConvNeXt-Tiny)
**Show:** the results table, confusion matrix and training curves.
**Say:**
- "We trained ConvNeXt-Tiny, a modern image model, starting from ImageNet-pretrained weights."
- "We trained two versions, each 3 times:
  - **4-class**: our main model;
  - **greyscale**: the same model without colour, to check whether colour was being used as a shortcut."
- **Confusion matrix:** "Most mistakes are between *neighbouring* stages, cancer vs pre-cancer or pre-cancer vs no dysplasia. It never confuses cancer with the 'no dysplasia' class. Even pathologists find these neighbouring stages hard."

### 7. Explainability (Grad-CAM): the important discovery
**Show:** the Grad-CAM images, then the border-check chart.
**Say:**
- "Grad-CAM shows **where the model looks**, as a heatmap. For cancer and leukoplakia it looks at tissue and cells, which is good."
- "For **Normal**, it was looking at the **edges of the image**, not the tissue. That was suspicious."
- "We tested it: **image sharpness alone**, one number per image, could tell Normal apart **95% of the time**. The Normal images were simply much sharper, because they came from a different hospital and camera setup."
- "So the model was partly recognising *'this is a sharp image'* instead of *'this is normal tissue'*. That's called a **shortcut**."
- "**Our fix:** we blurred the Normal images to match. Now sharpness can't tell them apart anymore (from 95% down to about 52%, which is chance level)."
- "**Then we retrained and checked again:** Normal stayed at 99%, and the heatmaps still looked at the edges. So the model still uses other image-wide differences, mainly that the Normal images are brighter."
- "**Our conclusion:** when you combine two datasets, the model can learn *which dataset* an image came from. Colour normalisation, tiling and blurring reduce this but can't fully remove it. That's why we report Normal's 99% with this caveat, and compare with published work only on the other three classes. Fixing it properly would need normal tissue from the same hospital as the other classes."

### 8. Whole images: segmentation → ROI → classification
**Show:** the segmenter Dice chart, the segmentation examples, the ROI results chart and the by-source table.
**Say:**
- "The full microscope images (before they were cut into patches) show more than the lesion: at 10x there is epithelium, connective tissue, muscle and empty glass. So segmentation makes sense there."
- "We used **1,461 whole images**: the 237 NDB-UFES originals plus the Rahman Normal and OSCC images. We split them **by patient**, so the same patient is never in training and test. That's stricter than the official patch split."
- **Annotation:** "We marked the epithelium on **50 images** using MedSAM boxes plus manual corrections. These masks are AI-assisted, not checked by a pathologist, so we say that next to every score."
- **Segmenter:** "We trained a U-Net to find the epithelium. On held-out images it reaches **Dice 0.85**, better than just taking all tissue (0.80) or the whole image (0.78). Dice is the overlap with the annotation; 1.0 would be perfect."
- **ROI vs all tissue:** "Then we classified each whole image twice: once from all tissue tiles, once only from the tiles the segmenter marked as epithelium. Result: **70.4% vs 71.7%** balanced accuracy. That difference is smaller than the variation between folds, so **the ROI doesn't help**."
- **Why:** "The by-source table shows it. The model mostly learns *which dataset* an image came from: NDB-UFES images are never called Normal, Rahman images are almost never called leukoplakia. Cutting away stroma can't remove that, because the cue is in the epithelium too."
- "And with a patient-level split, the three NDB-UFES classes drop to about **50–55%**, compared with 90% on the official patch split. That shows how much sharing patients between training and test inflates the patch results, ours and the published ones."

### 9. Limitations (be upfront; guides appreciate this)
**Say:**
- "The official dataset split is by image piece, not by patient, so the same patient can be in train and test. Published results have the same issue."
- "Normal comes from a different dataset, so some differences (brightness, contrast) remain."
- "Each model was trained 3 times with different random seeds. Results vary by 1–2 points between runs, so we report the average ± spread."
- "The epithelium masks are AI-assisted, not verified by a pathologist."
- "In the whole-image experiment, the classes come from different datasets, so the model can tell them apart by source."

### 10. Project structure and Getting started
**Say:** "The code is organised by stage: data, analysis, segmentation, classification, reporting. Everything can be re-run with one command per step, and the GPU steps run on Kaggle."

### 11. Live demo (prediction tool)
**Before the meeting**, start it in a terminal from the project folder:

```bash
streamlit run app/streamlit_app.py --server.address localhost
```

It opens in the browser at `http://localhost:8501`. The first start takes about 20 seconds.

**Show:**
1. **Try a test image** → pick an **OSCC** example: correct prediction, a confidence of about 90%+, and a heatmap on the tissue.
2. Pick a **"without dysplasia"** example: the hardest class. If it's wrong, say "this is the class even pathologists find hardest".
3. Pick a **Normal** example: the heatmap often sits at the tile's top edge, which is the shortcut we found.

**Say:**
- "The tool applies exactly the same steps as training: tiling, stain normalisation, then the model. We checked that it gives identical results to our test evaluation."
- "Large microscope images are cut into tiles, and the heatmap is stitched together."
- "It's a research prototype, not a diagnostic tool, and the page says so."

---

## Where we are: about 95% done

| Done ✅ | Still to do ⏳ |
|---|---|
| Literature survey, dataset and design | **Final report** / paper write-up |
| Data pipeline: splits, tiling, colour standardisation | |
| MedSAM segmentation on all images (finding documented) | |
| ConvNeXt classification (3 versions × 3 seeds, mean ± SD) | |
| Grad-CAM, shortcut found, blur fix tested, finding documented | |
| **Prediction tool**: web demo + command line (image → class + heatmap) | |
| Whole images: patient-level split, 50 epithelium annotations, U-Net segmenter (Dice 0.85) | |
| Segmentation → ROI → classification tested (no gain; source shortcut documented) | |

---

## Questions the guide may ask

- **"Why didn't segmentation work on the patches?"** The patches are already cut from inside the lesion, so the whole patch is the lesion. MedSAM had nothing smaller to isolate.
- **"So is there segmentation in the project?"** Yes. On the whole microscope images we annotated the epithelium on 50 images, trained a U-Net segmenter (Dice 0.85 on held-out patients), segmented all 1,461 images, and classified from the segmented regions.
- **"Then why didn't the ROI improve classification?"** The classifier mainly separates the two source datasets, and that cue is in the epithelium too. Removing stroma doesn't remove it. Within NDB-UFES, telling dysplasia from no dysplasia on whole images is close to chance with a patient-level split.
- **"Why are the whole-image numbers so much lower than 92%?"** Different, stricter test: whole images, and no patient appears in both training and test. The official patch split lets patches from one patient sit in both, which makes the patch scores (ours and published) optimistic.
- **"Who made the masks?"** They are AI-assisted (MedSAM plus manual corrections) and not checked by a pathologist. We state that next to every Dice score.
- **"Is 92% good?"** Yes. On the three classes that published papers use, we score 90.0%, close to the best published result (91.9%) with the same official split. And ours is an average over 3 runs, not a single best run.
- **"What does ± 1.1% mean?"** It's how much the result changes between training runs (standard deviation over 3 seeds). It shows the result is stable.
- **"Why compare on three classes and not four?"** Published papers on this dataset only have those three classes; Normal is our addition. Those three also all come from one dataset, so the Normal shortcut doesn't affect that comparison.
- **"How do you know the model isn't cheating?"** We checked with a colour test, a greyscale model, Grad-CAM heatmaps, and a sharpness test. That's how we found the Normal shortcut. We fixed sharpness and re-checked, and documented what remains.
- **"Why Kaggle?"** Our laptop has no GPU. Kaggle gives a free GPU, and the code runs there directly from our GitHub repo.
