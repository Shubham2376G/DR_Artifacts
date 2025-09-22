# Dataset Card: DDR-Augmented-Artifacts

## 1. Dataset Summary
**Name:** DDR-Augmented-Artifacts  
**Derived From:** [DDR Dataset](https://github.com/nkicsl/DDR-dataset) (Li et al., 2019)  
**Description:** This dataset contains **artifact-overlay augmented retinal fundus images** derived from the DDR dataset. Augmentations simulate acquisition artifacts caused by reflections from blood vessels, improving model robustness for diabetic retinopathy classification.  

**License:** MIT License (same as original DDR dataset)  

---

## 2. Dataset Composition
- **Images:** RGB retinal fundus images from DDR dataset with artifact overlays applied.  


---

## 3. Intended Use
- **Primary purpose:**  
  - Research on **robustness of diabetic retinopathy models** to acquisition artifacts.  
  - Benchmarking and augmentation pipelines for AI in retinal imaging.  
- **Not intended for:**  
  - Clinical diagnosis.  
  - Direct training with patient-identifiable data.  

---

## 4. Data Augmentation Method

Augmentation is performed using the following procedure:

1. **Artifact Patch Collection:**  
   - Images with reflection artifacts caused by blood vessels were gathered.  
   - Small regions (patches) containing these artifacts were **cropped and segmented**.  

2. **Overlay Procedure:**  
   - For each DDR image, a random artifact patch is selected and resized (10–20% of retina image size).  
   - A **feathered circular mask** is generated to smooth patch boundaries.  
   - Poisson blending (`cv2.seamlessClone`) is used to overlay the artifact on the retina image.  
   - Random placement is applied near the central region of the retina.  
   - Multiple artifacts can be applied per image for diversity.  

3. **Reproducibility:**  
   - The full augmentation procedure can be reproduced using the provided Python script.


---

## 5. Limitations
- Augmented images are **synthetic**; the artifacts may not fully capture real-world camera/device artifacts.  
- Overlay artifacts are limited to small patches, feathered masks, and Poisson blending.
---

## 6. Ethics & Privacy
- All artifact patches were derived from **personal, fully anonymized retinal images**.  
- No patient-identifiable data from DDR or personal images is included.  
- Intended for **research use only**, not for clinical or diagnostic purposes.  

---

## 7. Citation
If you use this dataset, please cite both the original DDR dataset and this augmentation work:

```bibtex
@article{LI2019,
  title = "Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening",
  author = "Tao Li and Yingqi Gao and Kai Wang and Song Guo and Hanruo Liu and Hong Kang",
  journal = "Information Sciences",
  volume = "501",
  pages = "511 - 522",
  year = "2019",
  issn = "0020-0255",
  doi = "https://doi.org/10.1016/j.ins.2019.06.011",
  url = "http://www.sciencedirect.com/science/article/pii/S0020025519305377",
}

@misc{Aggarwal2025_arxiv,
  title = {DDR-Augmented-Artifacts: Synthetic Artifact Overlays for Robust Diabetic Retinopathy Models},
  author = {Shubham Aggarwal},
  year = {2025},
  url = {https://arxiv.org/abs/XXXX.XXXXX}
}

