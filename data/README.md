# Data

This directory contains polymer property tables used by the PolyTab manuscript benchmarks.

| File | Rows | SMILES column | Property columns | Notes |
| --- | ---: | --- | ---: | --- |
| `DFT_properties_simple.csv` | 1,077 | `SMILES` | 14 | Quantum/DFT descriptor table |
| `MD_properties_simple.csv` | 1,077 | `SMILES` | 26 | MD-derived thermophysical table |
| `Table_Electronic.csv` | 296 | `SMILES` | 3 | Electronic dielectric properties |
| `Table_Energy.csv` | 3,528 | `SMILES` | 3 | Energy and bandgap properties |
| `Table_Permeability_Barrer.csv` | 787 | `SMILES` | 5 | Gas permeability values in Barrer |
| `calculated_polymer_data.csv` | 1,000 | `psmiles` | 13 | Calculated polymer properties |
| `polymer_MD.csv` | 95,335 | `smiles_list` | 12 | Large MD multi-property benchmark |

The experiment runners remove rows with nonnumeric target-property values before constructing controlled sparse-retention benchmarks.
