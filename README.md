# NestedWGCNA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18959244.svg)](https://doi.org/10.5281/zenodo.18959244)

A two-stage gene co-expression network analysis algorithm for bulk RNA-Seq data. Compared to WGCNA, it uses:

- **Adjacency:** r² (β = 2) instead of an arbitrary scale-free parameter
- **Dissimilarity:** √(1 − r²), a metric, instead of the Szymkiewicz–Simpson coefficient
- **Clustering:** UMAP + HDBSCAN instead of hierarchical clustering + Dynamic Tree Cut

This produces a two-level module hierarchy: **Coarse-Grained Modules (CGMs)** representing major cell types or biological processes, and **Fine-Grained Modules (FGMs)** representing sub-processes within each CGM, discovered via core-based expression normalization (GenFocus).

## Repository contents

| File | Description |
|------|-------------|
| `utils.py` | Core functions: adjacency, dissimilarity, UMAP+HDBSCAN clustering, core decomposition |
| `GenFocus.py` | Core-based normalization (INGS selection) for FGM discovery |
| `NestedWGCNA_analysis.ipynb` | Full analysis pipeline notebook |
| `data/` | Example data and precomputed IMvigor210 cluster assignments |

## Analysis notebook

`NestedWGCNA_analysis.ipynb` demonstrates the full pipeline on **TCGA BLCA** (openly available) and includes:

1. Preprocessing and CGM discovery
2. CGM core annotation (xCell / BostonGene signatures)
3. Bootstrap reproducibility analysis
4. FGM discovery via core-based normalization
5. FGM GO and cell-type enrichment
6. TF enrichment analysis — overrepresentation of ChEA_2022 TF targets per module
7. TF knowledge score — quantitative measure of TF co-regulation enrichment within modules
8. Survival and response analysis (requires controlled-access IMvigor210 data)

> **Note:** Sections 6–7 use precomputed IMvigor210 cluster files from `data/`, as the raw expression data is available only under controlled access via the European Genome-Phenome Archive (EGA). All functions accept any `pd.Series(gene → module_id)` and can be applied to your own data.

## Installation

```bash
git clone https://github.com/ilyada/NestedWGCNA.git
cd NestedWGCNA
conda env create -f environment.yml
conda activate nestedwgcna
```

## Citation

Dyugay et al. *Improved gene co-expression network analysis and its application for biomarker discovery of the response to immunotherapy.* (under review)

Code archived at Zenodo: https://doi.org/10.5281/zenodo.18959244
