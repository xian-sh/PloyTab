# PolyTab

Structure-informed reconstruction of sparse polymer property profiles.

PolyTab is the code companion for the manuscript **"PolyTab: Structure-Informed Reconstruction of Sparse Polymer Property Profiles"**. The framework reconstructs fragmented polymer property records into multi-property profiles by combining PSMILES-derived structural embeddings with partially observed property tables. It uses quartile-based classification for distribution-aware initialization and stacked adaptive regression layers for coarse-to-fine refinement while preserving observed entries.
<img width="825" height="442" alt="image" src="https://github.com/user-attachments/assets/4fd9303f-ae12-4daf-bbf0-23db9f13d05b" />

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
| `ele` | `Table_Electronic.csv` | `SMILES` |
| `energy` | `Table_Energy.csv` | `SMILES` |
| `barrer` | `Table_Permeability_Barrer.csv` | `SMILES` |
| `dft` | `DFT_properties_simple.csv` | `SMILES` |
| `md` | `MD_properties_simple.csv` | `SMILES` |
| `newmd` | `polymer_MD.csv` | `smiles_list` |
| `qc` | `calculated_polymer_data.csv` | `psmiles` |

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

## Citation

```bibtex
@article{si2026polytab,
  title = {PolyTab: Structure-Informed Reconstruction of Sparse Polymer Property Profiles},
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
This work was supported by the National Natural Science Foundation of China (U23A2090, 22293011, 22403087) and the Fundamental Research Funds for the Central Universities (WK2060000084). The authors acknowledge the support from the Supercomputing Center of the University of Science and Technology of China. The AI-driven experiments, simulations and model training were performed on the robotic AI-Scientist platform of the Chinese Academy of Sciences. During the preparation of this manuscript, the authors used LLM for language polishing and improving the overall readability of the text. The authors reviewed and edited the AI-assisted output and take full responsibility for the content of the published article.
