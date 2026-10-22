# Script: explaining the project to our guide

A spoken script in simple words, from start to end. Read it in order, about 10–12 minutes. Lines in *italics* tell you what to show on screen.

---

## 1. What problem are we solving?

"Ma'am, oral cancer is usually diagnosed by a pathologist who looks at a thin slice of tissue under a microscope. The tissue is stained pink and purple (H&E stain), and the pathologist decides whether it is:

1. **OSCC**: oral squamous cell carcinoma, which is cancer;
2. **Leukoplakia with dysplasia**: a white patch where the cells have started to look abnormal, so it is pre-cancer;
3. **Leukoplakia without dysplasia**: a white patch, but the cells still look normal;
4. **Normal** tissue.

This takes an expert's time, and dysplasia in particular is hard to judge. Our project builds a computer model that looks at these microscope images and tells which of the four groups they belong to. We also tested whether first **finding the important region** (segmentation) and then classifying it gives a better result."

---

## 2. The data

*Show: the dataset table and the sample images in the README.*

"We used two free public datasets:

- **NDB-UFES** from Brazil: cancer and the two leukoplakia groups. It has 237 full microscope photos from 69 patients, and the authors also cut them into 3,763 small square pieces called **patches**.
- **Rahman et al.** from India: 290 photos of normal tissue and 934 photos of cancer.

Both datasets allow research use, as long as we credit them."

---

## 3. The workflow from start to end

"Our work happened in two parts."

### Part A: the patch pipeline (our main classifier)

"**Step 1: Split the data.** We divide the images into three groups:
- **training**, which the model learns from;
- **validation**, which we use to pick the best version of the model;
- **test**, which the model has never seen, used only once at the end for the final score.

For NDB-UFES we used the official split from the dataset authors, so we can compare with published papers.

**Step 2: Cut images into tiles.** The normal photos were 4 times bigger than the patches. If we shrank them, the cells would look smaller and the model could cheat by looking at cell size. So we **cut** them into tiles of the same size (512 × 512 pixels), and we threw away tiles that were mostly empty glass.

**Step 3: Make the colours the same.** The two datasets were stained in different hospitals, one pinker and one more purple. We used a method called **Macenko stain normalisation**, which converts every image to the same standard pink and purple.

**Step 4: Make the sharpness the same.** We found the normal images were about 5 times sharper than the others, so the model could recognise 'normal' just because the image is sharp. We blurred them slightly to match.

**Step 5: Train the classifier.** ConvNeXt-Tiny, explained in section 5.

**Step 6: Test and explain.** We measure the accuracy on the test set and use **Grad-CAM** heatmaps to see where the model is looking."

### Part B: the whole-image pipeline (segmentation → ROI → classification)

"**Step 1:** We took the 1,461 full microscope photos and split them **by patient**, so the same patient is never in both training and test. This is a stricter and fairer test.

**Step 2:** We marked the epithelium by hand, with help from MedSAM, on 50 photos.

**Step 3:** We trained a segmentation model to find the epithelium automatically.

**Step 4:** We used it on all 1,461 photos to get the **region of interest (ROI)**.

**Step 5:** We classified each photo twice, once using only the ROI and once using all the tissue, and compared the two."

---

## 4. How segmentation works

*Show: the epithelium segmentation figure (green = epithelium).*

### What we are looking for, and why

"Oral tissue has layers. On top is the **epithelium**, the lining layer of cells. Below it is the **connective tissue (stroma)**: fibres, blood vessels and muscle. Around the tissue there is empty glass.

We chose the **epithelium as the region of interest** because:
- oral cancer **starts in the epithelium**;
- dysplasia is **judged only in the epithelium**, by how abnormal its cells look and whether its layers are still in order;
- the stroma and the glass don't help the decision; they are just background.

So the idea was to show the classifier only the part a pathologist actually looks at."

### First attempt: MedSAM on the patches

"MedSAM is a well-known medical segmentation model. We ran it on all 6,626 patches. It marked about 95–98% of every patch as the region. That's because the patches were already cut from **inside** the lesion, so there was nothing smaller to find. That's why we moved to the full microscope photos, where the epithelium is only part of the picture."

### How our segmenter learns

"1. **Teaching examples.** On 50 full photos we marked which pixels are epithelium. We drew a box, MedSAM suggested an outline, and we corrected it. These marked pictures are called **masks**.
2. **The model.** We used a **U-Net**, the standard network for medical segmentation. The first half of a U-Net shrinks the image step by step to understand *what* is in it; for example, 'these are closely packed cells in layers', which is epithelium, versus 'these are loose pink fibres', which is stroma. The second half enlarges it back step by step to say *where* exactly, pixel by pixel.
3. **Training.** The model sees random pieces of the 50 photos, guesses which pixels are epithelium, compares its guess with our mask, and corrects itself. It repeats this thousands of times.
4. **Output.** For every pixel it gives a number from 0 to 1: how sure it is that the pixel is epithelium. Above 0.5 counts as epithelium. That gives the green area in the figure.
5. **From mask to ROI.** We cut each photo into tiles and keep the tiles that are **at least half epithelium**. Those tiles are the ROI."

### How good is it?

*Show: the segmenter Dice chart.*

"We measure segmentation with **Dice**: how much the model's area overlaps our marked area. 1.0 is a perfect match and 0 is no overlap. We always tested on patients the model had never seen.
- Our segmenter: **Dice 0.85**.
- Simply taking all the tissue: 0.80. Taking the whole photo: 0.78.

So the segmenter really learned to find the epithelium, and it does better than these simple methods. One honest point: our masks were made with AI help and not checked by a pathologist."

---

## 5. How classification works

### The model: ConvNeXt-Tiny

"Our classifier is **ConvNeXt-Tiny**, a modern image-recognition network.

1. **It starts with general knowledge.** It was first trained by its authors on **ImageNet**, more than a million everyday photos. From that it already knows how to see edges, shapes, textures and patterns.
2. **We teach it our task.** This is called **fine-tuning**. We show it thousands of our tiles together with the correct answer. Each time it guesses wrong, it adjusts itself a little. After many rounds it learns which patterns belong to which group.
3. **We make it robust.** During training we randomly flip, rotate, blur and change the colours of the tiles, so it doesn't memorise exact pictures. We also give more weight to the smaller classes so it doesn't ignore them."

### What it looks at to tell the classes apart

"Nobody tells the model the rules; it learns them from the examples. But the differences it can use are the ones pathologists use:
- **OSCC (cancer):** cells breaking out of the epithelium and invading below, irregular groups of cells, large dark irregular nuclei, sometimes round 'keratin pearls'.
- **With dysplasia:** the cells are still inside the epithelium, but they look abnormal: bigger and darker nuclei, crowding, and the neat layers getting disorganised.
- **Without dysplasia:** a thicker surface layer, but the cells are still regular and in order.
- **Normal:** regular, orderly layers.

The early layers of the network detect simple things like edges and dots. The middle layers detect nuclei and cell shapes. The deep layers detect bigger patterns like crowding and invasion."

### How it gives the answer

"For each tile the model outputs **four probabilities** that add up to 100%, for example OSCC 85%, with dysplasia 10%, without dysplasia 4%, normal 1%. The highest one is the answer. A full photo has many tiles, so we **average** the tiles' probabilities and take the highest."

### Checking where it looks: Grad-CAM

*Show: the Grad-CAM images.*

"Grad-CAM colours the part of the image that influenced the decision most. For cancer and leukoplakia, the model looks at the cells and tissue, which is good. For normal, it often looked at the image edges. That made us suspicious, and it is how we discovered that the model could recognise *which dataset* a normal image came from, by sharpness and brightness, instead of the tissue. We reduced this with the blur fix and documented the rest honestly."

---

## 6. Accuracy and scores

### The scores we use

"- **Accuracy:** the percentage of correct answers. But if one class has many more images, a model can get high accuracy by mostly guessing that class.
- **Balanced accuracy:** the accuracy of each class separately, then averaged, so every class counts equally. This is our main score.
- **Macro-F1:** combines 'how many of its answers were right' and 'how many of the real cases it found', averaged over classes.
- **± value:** we trained each model 3 times with different random starting points. The ± shows how much the result moves between runs, so we report the average, not our luckiest run."

### Our results

*Show: the key results table and the confusion matrix.*

"- **4-class classifier on patches: 92.2 ± 1.1% balanced accuracy**, macro-F1 92.0%.
- Most mistakes are between **neighbouring stages**: cancer vs dysplasia, or dysplasia vs no dysplasia. It never confuses cancer with 'without dysplasia'. Even pathologists find neighbouring stages hard.
- **Segmenter:** Dice 0.85.
- **Whole photos, split by patient:** about 71% balanced accuracy, and using only the ROI gave about the same (70.4% vs 71.7%)."

---

## 7. What we compare with

"We compare with the best published paper on the same dataset: **Maia et al. 2024**, a DenseNet-121 model, which reached **91.9% balanced accuracy**.

Their paper uses only the three NDB-UFES classes; normal tissue is our own addition. So for a fair comparison we take **our model's score on those same three classes, on the same official split: 90.0 ± 1.7%**.

So our result is **comparable to the best published result**: about 2 points lower, within the normal variation between training runs. In addition:
- we handle **four classes**, including normal tissue;
- our number is an **average of 3 runs**, not a single best run;
- we **checked for shortcuts** (colour, greyscale, sharpness, Grad-CAM) and reported what we found."

> Don't say "best accuracy" or "better than published". Our 90.0% is a little below 91.9%, and the guide may check. Say **"comparable to the best published result"**. That is true, and with the extra checks it is a strong position.

---

## 8. What we learned (the honest findings)

"1. **On patches, segmentation can't help**, because the patches are already the lesion.
2. **On whole photos, segmentation works** (Dice 0.85), but **classifying only the ROI did not improve accuracy**. The reason is that the model mostly learns which dataset a photo came from: the Brazilian photos were never called normal, and the Indian photos were almost never called leukoplakia. Removing the background doesn't remove that cue, because it's in the epithelium too.
3. **The official split is optimistic.** It lets pieces from the same patient be in both training and test. When we split by patient on whole photos, the three NDB-UFES classes drop to about 50–55%. This affects the published papers too, so it is an important finding.
4. **Combining datasets creates shortcuts.** A proper fix would need all four classes from the same hospital."

---

## 9. Closing line

"So, ma'am: we built the full pipeline from data preparation to segmentation, classification and explanation. Our classifier reaches **92.2%** on four classes and is **comparable to the best published result** on the standard comparison. We also tested segmentation properly; it finds the epithelium well but doesn't improve classification on this data, and we explained why. Our next step is the final report."

---

## Quick answers if she asks

- **"What is ROI?"** Region of interest: the part of the image that matters for the diagnosis. For us, the epithelium.
- **"Why epithelium?"** Cancer starts there and dysplasia is judged there.
- **"How does the model know it's cancer?"** It learned from thousands of labelled examples which patterns go with which class, such as invading cells, abnormal nuclei and disorganised layers. Grad-CAM shows where it looked.
- **"Why is the whole-image score lower?"** It's a stricter test: whole photos, and no patient in both training and test.
- **"Did you get the best accuracy?"** We are at the published level: 90.0% vs 91.9% on the same three classes and split, averaged over 3 runs, with four classes and shortcut checks on top.
