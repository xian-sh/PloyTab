# Structure-informed reconstruction of sparse polymer property profiles.
Polymer discovery requires multi-property screening, but experimental data is frequently fragmented, resulting in profile-level sparsity where individual polymers have only partially reported property records. This incompleteness hinders multi-constraint decision-making, correlation analysis, and comprehensive materials selection. Existing solutions are limited: structure-based models often underuse co-observed property information, while table-based imputation methods neglect critical structural priors, leading to unstable associations under extreme sparsity.
<!-- <img width="825" height="442" alt="image" src="https://github.com/user-attachments/assets/4fd9303f-ae12-4daf-bbf0-23db9f13d05b" /> -->

![PolyTab TOC](https://github.com/user-attachments/assets/85c653ad-b10d-4dad-b8ff-96a5abcfcd80)
*Figure 1: PSMILES → embeddings → quartile init → adaptive regression → complete profiles.*
## Architecture
We propose PolyTab, a structure-informed framework for reconstructing partially observed polymer property profiles. PolyTab integrates:

    Structural Priors: PSMILES-derived embeddings from a pretrained polymer language model.
    Tabular Context: Observed property values, missingness masks, and statistical priors.
    Hybrid Architecture: A quartile-based classifier for distribution-aware initial estimates, refined by stacked adaptive regression layers that preserve experimentally observed entries.

This approach effectively combines structural information with inter-property correlations, enabling robust profile reconstruction and improved downstream prediction even under highly sparse data conditions.
![PolyTab main figure](https://github.com/user-attachments/assets/3a087339-f999-4615-8624-228144666185)
*Figure 2: Architecture details (a–c) and reconstruction performance across sparsity levels (d).*

## Repository Layout 

```text
polytab/
  data/                  Polymer property tables used in the benchmarks
  docs/                  Manuscript summary and reference notes
  figures/               Generated figures
  legacy/                Original notebook-exported code kept for traceability
  models/                Local model files, including TabPFN checkpoint and polyBERT path
  notebooks/             Example notebook
  results/               Experiment outputs
  scripts/               Command-line runners and plotting scripts
  src/polytab/           Main PolyTab package
```

## Installation

```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install -e .
```

For a full reproduction environment, install the packages listed in `requirements.txt`. GPU execution requires a PyTorch build compatible with your CUDA version.

## Model Files

PolyTab expects the pretrained polymer language model under:

```text
models/polyBERT/
```

You can override this path with:

```bash
set POLYBERT_MODEL_PATH=path\to\polyBERT
```

The TabPFN checkpoint is stored in `models/tabpfn-v2.5-regressor-v2.5_default.ckpt`. The repository includes Git LFS rules for checkpoint-like files; run `git lfs install` before committing large model weights.

## Datasets

The benchmark tables are in `data/`. Main task keys:

| Task | CSV file | SMILES column |
| --- | --- | --- |
| `Electronic` | `Table_Electronic.csv` | `SMILES` |
| `Energy` | `Table_Energy.csv` | `SMILES` |
| `Gas Barrer` | `Table_Permeability_Barrer.csv` | `SMILES` |
| `DFT` | `DFT_properties_simple.csv` | `SMILES` |
| `MD` | `MD_properties_simple.csv` | `SMILES` |
| `PolyOmics (MD) Dataset` | `polymer_MD.csv` | `smiles_list` |
| `QC` | `calculated_polymer_data.csv` | `psmiles` |

See `data/README.md` for row counts and property-table notes.

## Running PolyTab

```bash
python scripts/run_polytab.py --task md --keep-modes keep1 keep25 keep50 keep75
```

`keep1` corresponds to the single-property retention setting. `keep25`, `keep50`, and `keep75` retain 25%, 50%, and 75% of properties per polymer, respectively.

Outputs are written to:

```text
results/<task>_polyBERT_cascade_<keep_mode>_260107_SAFE/
```

Typical output files include:

| File | Description |
| --- | --- |
| `train_set_evaluation_masked.csv` | Train-set evaluation on masked positions only |
| `test_set_evaluation.csv` | Held-out test-set reconstruction metrics |
| `train_test_comparison.csv` | Train/test metric summary |
| `*_train_predictions.csv` | Reconstructed train table |
| `*_test_predictions.csv` | Reconstructed test table |
| `*_class_probabilities.csv` | Quartile-class probability summary |
| `training_curve.png` | Training loss curve |

## Running Baselines

```bash
python scripts/run_baseline.py --baseline blr --task md --keep-modes keep1
python scripts/run_baseline.py --baseline etr --task md --keep-modes keep1
python scripts/run_baseline.py --baseline tabpfn --task md --keep-modes keep1 --device cuda
```

Supported baselines are Bayesian linear regression (`blr`), extremely randomized trees (`etr`), and TabPFN (`tabpfn`).

## Plotting

After generating PolyTab and baseline outputs for the `newmd` task:

```bash
python scripts/plot_md_results.py
```

Figures are saved to `figures/`.

## Data and Model Availability

The pre-trained model weights and experimental process data are available in this repository. For long-term preservation and citation, a citable archived version of the research data and code has been deposited in Zenodo at https://zenodo.org/records/20602506.

## Citation

```bibtex
@article{si2026polytab,
  title = {Structure-Informed Reconstruction of Sparse Polymer Property Profiles},
  author = {Si, Zhan and Ge, Bojun and Hu, Jingjing and Wang, Chen and Yu, Haizhu and Liu, Deguang and Fu, Yao},
  journal = {journal},
  year = {2026},
  note = {Manuscript}
}
```

Also see `CITATION.cff`.

## License
No license file has been specified yet. Add a license before making the repository public if reuse terms should be explicit.
# Acknowledgements
This work was supported by the National Natural Science Foundation of China (U23A2090, 22293011, 22403087) and the Fundamental Research Funds for the Central Universities (WK2060000084). The authors acknowledge the support from the Supercomputing Center of the University of Science and Technology of China. The AI-driven experiments, simulations and model training were performed on the robotic AI-Scientist platform of the Chinese Academy of Sciences. During the preparation of this manuscript, the authors used LLM for language polishing and improving the overall readability of the text.
