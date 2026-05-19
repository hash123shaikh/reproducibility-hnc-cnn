# Independent Reproducibility Assessment of CNN-based Head and Neck Cancer Prognostic Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

Independent assessment of reproducibility and external validation of image-based prognosis in head and neck cancer using convolutional neural networks.

## Table of Contents
- [Description](#description)
- [Study Design](#study-design)
- [Key Findings](#key-findings)
- [Requirements](#requirements)
- [Data Pre-processing](#data-pre-processing)
- [Running the Model](#running-the-model)
- [Results](#results)
- [Citation](#citation)

---

## Description

This repository contains code and documentation for an **independent reproducibility assessment** of the study by Mateus et al. (2023):
> *"Image based prognosis in head and neck cancer using convolutional neural networks: a case study in reproducibility and optimization"* - Scientific Reports

We performed:
1. **Exact reproduction** - Attempted to replicate original results using provided code, data, and methodology
2. **Conceptual reproduction** - External validation on a geographically distinct cohort (CMC Vellore, India, n=102)
3. **Dual-framework compliance assessment** - Evaluated the original study against CLAIM and TRIPOD+AI reporting guidelines

---

## Study Design

![Study Design](dual-framework_compliance_assessment.png)

**Figure 1.** Overview of the independent reproduction effort showing exact reproducibility (using original datasets and code), conceptual reproducibility (external validation on CMC cohort), and dual-framework compliance assessment (CLAIM and TRIPOD+AI).

---

## Key Findings

- ✅ **Exact reproduction achieved** - Results comparable to original (AUC differences: –0.22 to +0.10)
- 🐛 **Preprocessing bug discovered** - Slice selection algorithm corrected, improving success rate from 60% to 100%
- ⚠️ **7 barriers encountered** - Required 7 weeks to resolve through author communication and code modifications
- 📊 **External validation** - Performance degradation observed on Indian cohort (AUC 0.54-0.59 vs original 0.71-0.79)
- 📋 **Dual-framework assessment** - CLAIM: 86% vs 95% (self-reported); TRIPOD+AI: 77%

---

## Requirements

### System Requirements
- Ubuntu 24.04 LTS (or similar Linux distribution)
- Python 3.10
- NVIDIA GPU (recommended: Quadro RTX 5000 or equivalent with 16GB RAM)
- FSL (FMRIB Software Library) version 6.0

### Python Dependencies

Install using the provided requirements file:

```bash
pip install -r requirements.txt
```

**Key packages:**
- PyTorch 1.13.1
- NumPy 1.22.0
- Scikit-learn 1.5.0
- Pillow 10.3.0
- nibabel 3.2.1
- dcmrtstruct2nii 1.0.19

---

## Data Pre-processing

We identified and corrected a critical bug in the original preprocessing pipeline. Our corrected pipeline consists of:

1. **DICOM to NIfTI conversion** - Convert CT scans and RT STRUCT using `dcmrtstruct2nii`
2. **Reorientation and masking** - Apply GTV mask to NIfTI using FSL
3. **Corrected slice selection** ⚠️ - Two-pass algorithm focusing on tumor-containing slices
4. **HU windowing** - Apply windowing (-50 to 300 HU) and Gaussian smoothing (σ=0.5)
5. **Cropping and normalization** - Crop to 180×180 pixels, normalize to 0-255

### ⚠️ Critical Fix: Slice Selection Algorithm

**Original bug:** Selected slice by total pixel count, often choosing non-tumor slices with artifacts.

**Our correction:** Two-pass tumor-focused algorithm
```python
# Pass 1: Identify tumor-containing slices
# Pass 2: Select slice with largest tumor area among tumor-containing slices
```

**Impact:** Improved success rate from 60% to 100% across all datasets.

**Scripts:**
- Step 1-2: `scripts/preprocessing/convert_dicom.py`
- Step 3-5: `scripts/preprocessing/windowing_cropping.py` (corrected version)

---

## Running the Model

### Quick Start

```bash
# Clone repository
git clone https://github.com/hash123shaikh/reproducibility-hnc-cnn.git
cd reproducibility-hnc-cnn

# Install dependencies
pip install -r requirements.txt

# Download public datasets
# Canadian: https://doi.org/10.7937/K9/TCIA.2017.8oje5q00
# Dutch: https://doi.org/10.7937/TCIA.2019.8kap372n

# Run preprocessing
python scripts/preprocessing/windowing_cropping.py

# Train model (example: Distant Metastasis)
python scripts/training/training_dm.py
```

### Data Configuration

Configure data paths in training scripts:

```python
DATA = {
    TRAIN: {
        CLINICAL_DATA_PATH: "data/canada.csv",
        SCANS_PATH: "data/pre-processed/canada/",
    },
    VALIDATION: {
        CLINICAL_DATA_PATH: "data/canada.csv",
        SCANS_PATH: "data/pre-processed/canada/",
    },
    TESTING: {
        CLINICAL_DATA_PATH: "data/maastro.csv",
        SCANS_PATH: "data/pre-processed/maastro/",   
    }
}
```

### Model Architectures

Three model types are available:

1. **ANN (Clinical Only)** - Artificial Neural Network using only clinical variables
   - Input: 11 clinical features (T-stage, N-stage, tumor volume - one-hot encoded)
   - Architecture: 4 fully connected layers

2. **CNN (Imaging Only)** - Convolutional Neural Network using only CT images
   - Input: 180×180 pixel cropped CT images
   - Architecture: 3 convolutional blocks + 4 fully connected layers

3. **CNN + Clinical** - Combined model using both imaging and clinical data
   - Input: CT images + 11 clinical features
   - Architecture: CNN feature extraction + clinical data fusion

### Validation Strategies

**Cohort Split Approach:**
- Training: 2 Canadian centers (HGJ, CHUS)
- Validation: 2 Canadian centers (HMR, CHUM)
- External Test: Dutch dataset (MAASTRO)

**5-Fold Cross-Validation:**
- Stratified k-fold on full Canadian dataset
- External Test: Dutch dataset (MAASTRO)

### Training Scripts

Training scripts follow this naming convention:
- `training_{outcome}.py` = 5-fold cross-validation, **imaging only**
- `training_{outcome}_cd.py` = Cohort split, **imaging + clinical data**

where `{outcome}` is:
- `dm` - Distant Metastasis (2-year)
- `lrf` - Locoregional Failure (2-year)  
- `os` - Overall Survival (4-year)

**Examples:**

```bash
# Distant Metastasis
python scripts/training/training_dm.py       # imaging only
python scripts/training/training_dm_cd.py    # imaging + clinical

# Locoregional Failure
python scripts/training/training_lrf.py      # imaging only
python scripts/training/training_lrf_cd.py   # imaging + clinical

# Overall Survival
python scripts/training/training_os.py       # imaging only
python scripts/training/training_os_cd.py    # imaging + clinical
```

**Training time:** ~17-19 hours per outcome (5-fold CV), ~7GB peak memory usage

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
[Your paper citation - to be added after publication]
```

**And the original study:**

```bibtex
@article{mateus2023image,
  title={Image based prognosis in head and neck cancer using convolutional neural networks: a case study in reproducibility and optimization},
  author={Mateus, Pedro and Volmer, Leroy and Wee, Leonard and Aerts, Hugo JWL and Hoebers, Frank and Dekker, Andre and Bermejo, Inigo},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={18176},
  year={2023},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41598-023-45486-5}
}
```

---

## Acknowledgments

- **Original study authors** (Mateus et al.) for sharing code and providing clarifications during reproduction
- **The Cancer Imaging Archive (TCIA)** for providing public datasets
- **Christian Medical College Vellore** for computational resources and clinical data

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact

- **Hasan Shaikh** - [GitHub](https://github.com/hash123shaikh)
- **Dr. Hannah Mary Thomas T** - hannah.thomas@cmcvellore.ac.in
- **Institution:** Quantitative Imaging Research and AI Lab, Christian Medical College, Vellore, India

---

## Contributing

We welcome contributions! If you find issues or have suggestions for improvements, please:
1. Open an issue describing the problem
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

---

**Note:** This is a reproducibility study. For the original implementation, see the [original repository](https://github.com/MaastrichtU-CDS/hn_cnn).
