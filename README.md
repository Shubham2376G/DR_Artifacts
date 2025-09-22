# DDR-Augmented-Artifacts

This repository provides **artifact-overlay augmented samples** of the **DDR (Diabetic Retinopathy) dataset**.  
The goal is to help researchers and practitioners simulate acquisition artifacts and study model robustness.

---

## 🔗 Original Dataset

The raw DDR dataset must be downloaded separately:  
[DDR Dataset GitHub Repository](https://github.com/nkicsl/DDR-dataset)  
> Paper citation: Li et al., *Information Sciences*, 2019.  

After downloading DDR, you can reference our augmentation code and sample images to reproduce augmented variants or use as a benchmark.

---

## 📂 Repository Contents

- `augmentations/` — scripts and configuration used for artifact overlay
- `CITATION.bib` — how to cite the original dataset and this augmentation work  
- `LICENSE` — MIT license

---

## 🖼️ Sample Images

| Original (DDR) | Augmented (this repo) |
|----------------|-----------------------|
| ![Clean](images/original.jpg) | ![Augmented](images/augmented.png) |

---

## ⚙️ Usage

1. Download and extract the **original DDR dataset** into the data folder.  
2. Use the augmentation scripts in `augmentations/` (optional) to apply artifact overlays on your own copy of DDR.    
3. Augmented samples in `augmentations/output/` demonstrate expected outputs.

> Note: Only a **subset of augmented images** is provided here; to generate the full augmented dataset, run the augmentation pipeline on complete dataset.

---

## 📑 Citation

Please cite **both the original DDR dataset and this augmentation work** if you use it:

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
