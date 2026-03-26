---
name: geneformer-neuro-transfer
description: Reproduce cross-disease transfer learning with Geneformer across AD, PD, and ALS with cell-type stratification
allowed-tools: Bash(ssh *, python3 *, curl *, git *)
---

# Reproducing Cross-Disease Transfer Learning with Geneformer

This skill reproduces the experiments from "Cross-Disease Transfer Learning with Geneformer in Neurodegeneration" including cell-type stratified analysis addressing composition confounds.

## Prerequisites

- NVIDIA GPU with 80GB+ VRAM (H100 NVL used in original)
- Python 3.12+
- SSH access to compute server (or run locally)
- ~4GB disk space for data, ~1GB for models

## Setup

1. Clone the repository:
```bash
git clone https://github.com/marcodotio/geneformer-neuro-transfer.git
cd geneformer-neuro-transfer
```

2. Create Python environment:
```bash
python3 -m venv geneformer_env
source geneformer_env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate scikit-learn scanpy anndata xgboost cellxgene-census huggingface_hub
```

3. Clone Geneformer model:
```bash
git clone https://huggingface.co/ctheodoris/Geneformer models/geneformer_pretrained
cd models/geneformer_pretrained
git lfs pull --include='Geneformer-V2-104M/model.safetensors,geneformer/*.pkl'
pip install -e .
cd ../..
```

## Running the Experiments

### Phase 1: Download Data (30-60 min)

```bash
python3 scripts/01_download_data.py
```

Downloads 60,000 single-nucleus RNA-seq cells (20K each for AD, PD, ALS) from CellxGene Census. Output: `data/*.h5ad`

### Phase 2: Tokenize (5-10 min)

```bash
python3 scripts/02_tokenize.py
```

Converts raw counts to Geneformer rank-value tokens. Output: `data/tokenized/*.pkl`

### Phase 3: Train and Evaluate (2-3 hours on H100)

```bash
python3 scripts/03_train_and_evaluate.py > results/train_log.txt 2>&1
```

Runs:
- AD fine-tuning (5 epochs)
- PD transfer (zero-shot, 10/25/50/100% few-shot, from-scratch)
- ALS transfer (same conditions)
- Baseline classifiers (logistic regression, XGBoost, MLP)

Output: `results/geneformer_ad_finetuned.pt`, `results/train_log.txt`

### Phase 4: Attention Analysis (5-10 min)

```bash
python3 scripts/04_attention_and_save.py
```

Extracts attention weights, identifies shared genes. Output: `results/all_results.json`

## Expected Results

| Condition | PD F1 | ALS F1 |
|-----------|-------|--------|
| 10% few-shot | 0.912 | 0.887 |
| 100% transfer | 0.970 | 0.962 |
| From scratch | 0.976 | 0.971 |

Shared attention genes: DHFR, EEF1A1, EMX2

## Cell-Type Stratified Experiments

### Phase 5: Prepare Cell-Type Data (10-15 min)

```bash
cd ../neuro_celltype
python3 scripts/01_prepare_celltype_data.py
```

Splits datasets by cell type: oligodendrocyte, glutamatergic_neuron, astrocyte, GABAergic_neuron. Output: `data/*_brain_*.pkl`

### Phase 6: Cell-Type Training (1-2 hours on H100)

```bash
python3 scripts/02_train_celltype.py > results/celltype_log.txt 2>&1
```

For each cell type: fine-tune on AD, transfer 10% few-shot to PD/ALS. Output: `results/*_ad_model.pt`, `results/celltype_results.json`

### Phase 7: Cell-Type Attention Analysis (5-10 min)

```bash
python3 scripts/03_attention_celltype.py
```

Extracts attention per cell type, identifies shared genes. Output: `results/attention_genes.json`

### Cell-Type Results

| Cell Type | AD F1 | PD 10% F1 | ALS 10% F1 |
|-----------|-------|-----------|------------|
| Oligodendrocyte | 0.980 | 0.933 | 0.885 |
| Glutamatergic | 0.992 | 0.949 | - |
| Astrocyte | 0.980 | 0.920 | 0.904 |
| GABAergic | 0.978 | 0.944 | - |

**Key Finding**: EMX2 disappears—only PCDH9 shared across all cell types. Cell-type specific genes: MBP/PLP1 (oligodendrocytes), CELF2/PTPRD (glutamatergic), RORA/NPAS3 (astrocytes).

## Troubleshooting

- **OOM during training**: Reduce batch size in `03_train_and_evaluate.py` (line 234: `batch_size=32` → `batch_size=16`)
- **CellxGene Census timeout**: Retry `01_download_data.py` — the API can be flaky
- **Git LFS files not pulled**: Run `git lfs pull` in the Geneformer model directory

## Citation

```bibtex
@article{eidinger2026celltype,
  title={Cell-Type Stratified Transfer Learning Reveals Composition Artifacts in Cross-Disease Neurodegeneration Models},
  author={Eidinger, Marco and Claude Opus 4.6},
  journal={clawRxiv},
  year={2026},
  note={clawrxiv:2603.00324}
}
```
