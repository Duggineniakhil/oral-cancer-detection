# Oral Cancer Detection using ConvNeXt + MedSAM

## Project Overview
This project focuses on developing an efficient deep learning pipeline for oral carcinoma detection using both classification and segmentation.

The proposed system integrates:
- ConvNeXt-Tiny for classification (disease / no disease)
- MedSAM for precise lesion segmentation
- Preprocessing (resize + normalization)
- Hybrid deployment (local + cloud)

---

## Problem Statement
Oral cancer detection suffers from:
- Late diagnosis
- Lack of precise lesion localization
- Heavy models not suitable for real-time deployment

---

## Proposed Solution
We propose a lightweight dual-branch pipeline:

```
Input Image
→ Preprocessing
→ ConvNeXt → Classification
→ MedSAM → Segmentation
→ Final Output
```

---

## Key Contributions
- Efficient and deployable architecture
- Integration of classification + segmentation
- Cloud-assisted segmentation for scalability

---

## Dataset
We use publicly available oral cancer datasets.

Note:
Due to size constraints, the dataset is not included in this repository.


Sample images are provided in `/data/sample_images/`.

---

## Tech Stack
- Python
- PyTorch
- OpenCV
- Google Colab (for MedSAM)

---

## Repository Structure
```
docs/ → Report & presentation
data/ → Sample images
src/ → Code (Phase 2)
```

---

## Current Status
Phase 1 (Completed):
- Literature survey
- Gap identification
- Proposed pipeline design

Phase 2 (Upcoming):
- Model implementation
- Evaluation
- Deployment

---

## Team Members
- K. Siddhartha  
- K. Yoganand  
- D. Akhil  
- D. Akash  

---

## Guide
Mrs. Bharathi D

---

## Repository Usage
This repository currently contains research and design artifacts. Implementation will be added in Phase 2.

---
