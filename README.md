# 🧬 PolyTab

### Structure-Informed Reconstruction of Sparse Polymer Property Profiles

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20602506.svg)](https://doi.org/10.5281/zenodo.20602506)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[Paper](#citation) • [Installation](#installation) • [Datasets](#datasets) • [Quick Start](#running-polytab) • [Zenodo Archive](https://zenodo.org/records/20602506)

---

</div>

## 🎯 Overview

Polymer discovery demands multi-property screening, yet experimental data remains severely fragmented—individual polymers often have only **partial property records**, creating profile-level sparsity that undermines multi-constraint decision-making and comprehensive materials selection.

**PolyTab** solves this by fusing structural priors with tabular context: PSMILES embeddings from pretrained polymer language models meet adaptive regression layers that respect observed entries while reconstructing missing values.

![PolyTab TOC](https://github.com/user-attachments/assets/85c653ad-b10d-4dad-b8ff-96a5abcfcd80)

**Figure 1** | PSMILES → embeddings → quartile init → adaptive regression → complete profiles

</div>

---

## ⚙️ Architecture

![PolyTab main figure](https://github.com/user-attachments/assets/3a087339-f999-4615-8624-228144666185)

**Figure 2** | Hybrid architecture (a–c) and reconstruction performance across sparsity regimes (d)

</div>

**Core Components:**
- **Structural Priors**: Polymer embeddings from pretrained language models
- **Tabular Context**: Observed values + missingness masks + statistical distributions
- **Hybrid Pipeline**: Quartile classifier → stacked adaptive regression → preserved experimental entries

---

## 📦 Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

**GPU Support**: Install PyTorch with CUDA compatibility for your system.

---

## 📊 Datasets

| Task | File | SMILES Column | Rows |
|------|------|---------------|------|
| Electronic | `Table_Electronic.csv` | `SMILES` | — |
| Energy | `Table_Energy.csv` | `SMILES` | — |
| Gas Barrier | `Table_Permeability_Barrer.csv` | `SMILES` | — |
| DFT | `DFT_properties_simple.csv` | `SMILES` | — |
| MD | `MD_properties_simple.csv` | `SMILES` | — |
| PolyOmics | `polymer_MD.csv` | `smiles_list` | — |
| QC | `calculated_polymer_data.csv` | `psmiles` | — |

All datasets are in `data/`. See [`data/README.md`](data/README.md) for details.

---

## 🚀 Running PolyTab

```bash
python scripts/run_polytab.py --task md --keep-modes keep1 keep25 keep50 keep75
```

**Sparsity Modes:**
- `keep1`: Single-property retention per polymer
- `keep25/50/75`: Retain 25%/50%/75% of properties per polymer

**Outputs** → `results/<task>_polyBERT_cascade_<keep_mode>_260107_SAFE/`

| File | Description |
|------|-------------|
| `train_set_evaluation_masked.csv` | Train metrics (masked positions only) |
| `test_set_evaluation.csv` | Test-set reconstruction performance |
| `*_predictions.csv` | Reconstructed property tables |
| `training_curve.png` | Loss evolution |

---

## 📈 Baseline Comparisons

```bash
# Bayesian Linear Regression
python scripts/run_baseline.py --baseline blr --task md --keep-modes keep1

# Extremely Randomized Trees
python scripts/run_baseline.py --baseline etr --task md --keep-modes keep1

# TabPFN (requires GPU)
python scripts/run_baseline.py --baseline tabpfn --task md --keep-modes keep1 --device cuda
```

**Generate Plots:**
```bash
python scripts/plot_md_results.py  # → figures/
```

---

## 🗂️ Repository Structure

```
polytab/
├── data/                   # Benchmark property tables
├── models/                 # Pretrained weights (polyBERT, TabPFN)
├── src/polytab/            # Core package
├── scripts/                # CLI runners & plotting
├── notebooks/              # Jupyter examples
├── results/                # Experiment outputs
└── figures/                # Generated visualizations
```

---

## 🔧 Model Configuration

**PolyBERT Path**: `models/polyBERT/`  
Override via: `export POLYBERT_MODEL_PATH=/custom/path`

**TabPFN Checkpoint**: `models/tabpfn-v2.5-regressor-v2.5_default.ckpt`

> **Note**: Large model files use Git LFS. Run `git lfs install` before committing.

---

## 📚 Citation

```bibtex
@article{si2026polytab,
  title   = {Structure-Informed Reconstruction of Sparse Polymer Property Profiles},
  author  = {Si, Zhan and Ge, Bojun and Hu, Jingjing and Wang, Chen and 
             Yu, Haizhu and Liu, Deguang and Fu, Yao},
  journal = {Journal Name},
  year    = {2026},
  note    = {Manuscript}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable format.

---

## 🎓 Acknowledgements

This work was supported by:
- National Natural Science Foundation of China (U23A2090, 22293011, 22403087)
- Fundamental Research Funds for the Central Universities (WK2060000084)
- Supercomputing Center, University of Science and Technology of China
- Robotic AI-Scientist platform, Chinese Academy of Sciences

**Disclosure**: LLMs were used for language polishing during manuscript preparation.

---

## 📄 License

[[CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

This work is licensed under [Creative Commons Attribution 4.0 International](LICENSE).

---

<div align="center">

**[Data & Code Archive](https://zenodo.org/records/20602506)** | **[Documentation](docs/)** | **[Issues](../../issues)**

Made with ❤️ for the polymer science community

## Reproducibility Notes

The supported result-generation path is `scripts/run_polytab.py`, which calls `src/polytab/pipeline.py`. The released model follows the manuscript-level design: PSMILES-derived structural embeddings, observed/imputed property values, missingness masks, four quartile-interval classification outputs, interval-based initialization, CNN prior-feature construction with a 1 x 3 x 3 kernel over the stacked tabular-context tensor, and stacked adaptive regression. The supported code does not use a VAE component.

For row/property-column order checks, run:

```bash
python scripts/run_permutation_stability.py --task md --keep-mode keep25
```
