# Presenting the Project to the Guide

A simple walkthrough to use while scrolling through the [README](../README.md). It follows the README's order. For each part: **what to show** and **what to say**.

---

## 1-minute summary (say this first)

"Our project classifies microscope images of oral tissue into four groups: **cancer (OSCC)**, **pre-cancer (leukoplakia with dysplasia)**, **leukoplakia without dysplasia**, and **normal tissue**. Our plan was two steps: first find the lesion with a segmentation model (**MedSAM**), then classify it with **ConvNeXt-Tiny**. We built and tested both. On the three classes from the main dataset, our classifier reaches **about 91% balanced accuracy**, close to the best published result (91.9%). We also discovered that the Normal class, which comes from a different dataset, can be recognised from image-level differences between the two datasets instead of tissue. We investigated and documented this carefully."

---

## Section by section

### 1. Key results (top of README)
**Show:** the results table.
**Say:**
- "Our main result is the **3-class model: 90.9% balanced accuracy**. Balanced accuracy means it's about 91% right *for each class on average*, which is fairer than plain accuracy when some classes have fewer images."
- "The best published result on the same dataset and the same official split is 91.9%, so we're at the published level."
- "We also have a 4-class model including Normal (88.6%, with Normal at 99%). But we show that Normal's 99% comes from differences between the two datasets, not from the tissue. That's why the 3-class result is our main one."
- "Segmentation didn't help on our data. I'll explain why."

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
- "A useful segmentation for the future would be the **epithelium** (the surface tissue layer where dysplasia is judged). But that needs hand-drawn masks to train."

### 6. Classification (ConvNeXt-Tiny)
**Show:** the results table, confusion matrix and training curves.
**Say:**
- "We trained ConvNeXt-Tiny, a modern image model, starting from ImageNet-pretrained weights."
- "We trained three versions:
  - **4-class**: our main model;
  - **3-class**: without Normal, to compare directly with papers;
  - **greyscale**: to check whether colour was being used as a shortcut."
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
- "**Our conclusion:** when you combine two datasets, the model can learn *which dataset* an image came from. Colour normalisation, tiling and blurring reduce this but can't fully remove it. That's why we treat the **3-class result** as our main result. Fixing it properly would need normal tissue from the same hospital as the other classes."

### 8. Limitations (be upfront; guides appreciate this)
**Say:**
- "The official dataset split is by image piece, not by patient, so the same patient can be in train and test. Published results have the same issue."
- "Normal comes from a different dataset, so some differences (brightness, contrast) remain."
- "Each model was trained once. For the final report we'll train each 3 times and report the average."

### 9. Project structure and Getting started
**Say:** "The code is organised by stage: data, analysis, segmentation, classification, reporting. Everything can be re-run with one command per step, and the GPU steps run on Kaggle."

---

## Where we are: about 75–80% done

| Done ✅ | Still to do ⏳ |
|---|---|
| Literature survey, dataset and design | Train each model **3 times** and report the average *(scores vary 3–5 points between single runs)* |
| Data pipeline: splits, tiling, colour standardisation | Try a **larger image size** (384px) for finer cell detail |
| MedSAM segmentation on all images (finding documented) | **Prediction tool**: upload an image, get class + heatmap |
| ConvNeXt classification (3 versions, retrained after the fix) | **Final report** / paper write-up |
| Grad-CAM, shortcut found, blur fix tested, finding documented | *Optional:* epithelium segmentation (needs hand-drawn masks) |

---

## Questions the guide may ask

- **"Why didn't segmentation work?"** The images are already cut from inside the lesion, so the whole image is the lesion. MedSAM had nothing smaller to isolate.
- **"Then why keep MedSAM in the project?"** We tested the hypothesis properly on all 6,626 images. A tested negative result is a valid research finding. Grad-CAM now does the "where is the problem" part.
- **"Is 91% good?"** Yes. It's at the level of the best published result on this dataset (91.9%), with the same official split.
- **"Why is your main result 3-class and not 4-class?"** The Normal images come from another dataset, and we proved the model partly recognises *which dataset* an image is from. The 3 main classes all come from one dataset, so that result is fair.
- **"How do you know the model isn't cheating?"** We checked with a colour test, a greyscale model, Grad-CAM heatmaps, and a sharpness test. That's how we found the Normal shortcut. We fixed sharpness and re-checked, and documented what remains.
- **"Why Kaggle?"** Our laptop has no GPU. Kaggle gives a free GPU, and the code runs there directly from our GitHub repo.
