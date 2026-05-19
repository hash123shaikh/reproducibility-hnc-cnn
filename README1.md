# Independent Reproducibility Assessment of CNN-based Head and Neck Cancer Prognostic Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

Independent assessment of reproducibility and external validation of image-based prognosis in head and neck cancer using convolutional neural networks.

## Table of Contents
- [Description](#description)
- [Key Findings](#key-findings)
- [Requirements](#requirements)
- [Data Pre-processing](#data-pre-processing)
- [Running the Model](#running-the-model)
- [Reproducibility Barriers Documented](#reproducibility-barriers-documented)
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

Despite the original study's 95% CLAIM compliance and publicly available code, we encountered **7 distinct barriers** requiring **7 weeks** to resolve. This work demonstrates that code availability and reporting checklist compliance alone do not guarantee independent reproducibility.

---

## Key Findings

### Reproducibility Challenges
- ✅ **Exact reproduction achieved** with AUC differences ranging from –0.22 to +0.10
- ⚠️ **7 barriers identified** across all pipeline stages (data loading, preprocessing, training, evaluation)
- 🐛 **Critical preprocessing bug discovered** - slice selection algorithm selected by total pixel count rather than tumor content, affecting ~40% of training data
- ⏱️ **7 weeks resolution time** required through author communication and code modifications
- 📋 **Documentation gaps** found despite 95% CLAIM compliance

### External Validation
- 📊 **Performance degradation** observed on Indian cohort (AUC 0.54-0.59 vs original 0.71-0.79)
- 🌍 **Population differences** - Higher laryngeal representation (54% vs 15-35%), different event rates (43.1% vs 14.4-24.8%)
- ✅ **Preprocessing pipeline validated** on 102 CMC patients with 100% success rate

### Dual-Framework Assessment
- 📝 **CLAIM score**: 86% (independent assessment) vs 95% (self-reported)
- 📝 **TRIPOD+AI score**: 77%
- 🔍 **Complementary coverage** - CLAIM focused on technical details, TRIPOD+AI on clinical deployment

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

For complete environment setup, see [installation guide](docs/reproduction_guide.md).

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

### Training Options

**Models available:**
- CNN (Convolutional Neural Network) - Imaging only
- ANN (Artificial Neural Network) - Clinical data only
- CNN + ANN - Combined imaging and clinical data

**Validation strategies:**
- `COHORT_SPLIT` - Center-based partitioning (2 centers train, 2 validate, 1 external test)
- `CROSS_VALIDATION` - 5-fold stratified cross-validation

**Outcomes supported:**
- `DM` - Distant Metastasis (2-year)
- `LRF` - Locoregional Failure (2-year)
- `OS` - Overall Survival (4-year)

### Training Scripts

```bash
# Distant Metastasis
python scripts/training/training_dm.py      # 5-fold CV
python scripts/training/training_dm_cd.py   # Cohort split

# Locoregional Failure
python scripts/training/training_lrf.py     # 5-fold CV
python scripts/training/training_lrf_cd.py  # Cohort split

# Overall Survival
python scripts/training/training_os.py      # 5-fold CV
python scripts/training/training_os_cd.py   # Cohort split
```

**Expected training time:** ~17-19 hours per outcome (5-fold CV), ~7GB peak memory

---

## Reproducibility Barriers Documented

We systematically documented all 7 barriers encountered during reproduction:

| ID | Barrier | Stage | Resolution Time |
|----|---------|-------|-----------------|
| B1 | Column name mismatch | Data loading | 2 weeks |
| B2 | Incorrect slice selection | Preprocessing | 2 weeks |
| B3 | Unclear epoch selection | Training | 4 days |
| B4 | Missing pre-trained weights location | Training | 2 days |
| B5 | Cross-validation results overwriting | Evaluation | 1 day |
| B6 | Bootstrap CI implementation missing | Evaluation | 1 week |
| B7 | Hardware-dependent numerical variation | Evaluation | 1 week |

**Total resolution time:** ~7 weeks

For detailed documentation of each barrier, resolution approach, and impact on reproducibility, see [barriers_documentation.md](docs/barriers_documentation.md).

---

## Results

### Exact Reproduction Comparison

| Outcome | Model Type | Original AUC (Test) | Reproduced AUC (Test) | Δ AUC |
|---------|-----------|---------------------|----------------------|-------|
| DM (2y) | Clinical Only | 0.87 | 0.87 | 0.00 |
| DM (2y) | Imaging Only | 0.89 | 0.87 | -0.02 |
| DM (2y) | Clinical + Imaging | 0.93 | 0.92 | -0.01 |
| LRF (2y) | Clinical Only | 0.41 | 0.33 | -0.08 |
| LRF (2y) | Imaging Only | 0.45 | 0.49 | +0.04 |
| LRF (2y) | Clinical + Imaging | 0.59 | 0.57 | -0.02 |
| OS (4y) | Clinical Only | 0.69 | 0.65 | -0.04 |
| OS (4y) | Imaging Only | 0.67 | 0.67 | 0.00 |
| OS (4y) | Clinical + Imaging | 0.69 | 0.69 | 0.00 |

*Results shown for cohort split approach*

### External Validation (CMC Cohort, n=102)

| Model Type | AUC (Cohort Split) | AUC (5-fold CV) |
|------------|-------------------|-----------------|
| Clinical Only | 0.49 [0.37, 0.60] | 0.51 (0.40–0.61) |
| Imaging Only | 0.54 [0.43, 0.66] | 0.56 (0.54–0.59) |
| **Clinical + Imaging** | **0.59 [0.48, 0.70]** | 0.56 (0.53–0.60) |

---

## Documentation

- **[Reproduction Guide](docs/reproduction_guide.md)** - Step-by-step instructions for reproducing our work
- **[Barriers Documentation](docs/barriers_documentation.md)** - Detailed analysis of all 7 reproducibility barriers
- **[Preprocessing Pipeline](docs/preprocessing_pipeline.md)** - Complete preprocessing workflow with corrected slice selection

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
