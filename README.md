# Cross-Disease Transfer Learning with Geneformer in Neurodegeneration

Fine-tuning Geneformer V2 (104M) on Alzheimer's disease single-cell RNA-seq and evaluating cross-disease transfer to Parkinson's disease and ALS.

## Key Results

| Condition | PD F1 | PD AUROC | ALS F1 | ALS AUROC |
|-----------|-------|----------|--------|-----------|
| Zero-shot | 0.036 | 0.439 | 0.016 | 0.416 |
| 10% few-shot | 0.912 | 0.952 | 0.887 | 0.933 |
| 25% few-shot | 0.947 | 0.981 | 0.914 | 0.979 |
| 50% few-shot | 0.958 | 0.988 | 0.942 | 0.990 |
| 100% transfer | 0.970 | 0.992 | 0.962 | 0.995 |
| From scratch | 0.976 | 0.994 | 0.971 | 0.996 |

AD fine-tuning: F1 = 0.989, AUROC = 0.996

3 genes shared in top-50 attention across all three diseases: **DHFR**, **EEF1A1**, **EMX2**

## Project Structure

```
scripts/
  01_download_data.py      # Download scRNA-seq from CellxGene Census
  02_tokenize.py            # Rank-value tokenization for Geneformer
  03_train_and_evaluate.py  # Fine-tuning, transfer learning, baselines
  04_attention_and_save.py  # Attention analysis and results compilation
results/
  all_results.json          # All metrics in structured JSON
  train_log.txt             # Full training log
data/                       # Not tracked (see Data section)
models/                     # Not tracked (see Models section)
```

## Data

Data was sourced from [CellxGene Census](https://chanzuckerberg.github.io/cellxgene-census/) (stable release 2025-11-08). Run `scripts/01_download_data.py` to reproduce:

- **AD**: 20,000 cells (10K disease / 10K control), brain tissue
- **PD**: 20,000 cells (10K / 10K), brain tissue
- **ALS**: 20,000 cells (10K / 10K), brain tissue

## Models

Base model: [Geneformer V2 104M](https://huggingface.co/ctheodoris/Geneformer) (Theodoris et al., 2023)

Fine-tuned weights: [marcodotio/geneformer-neuro-transfer](https://huggingface.co/marcodotio/geneformer-neuro-transfer) *(upload pending)*

## Environment

- NVIDIA H100 NVL (96 GB)
- Python 3.12, PyTorch 2.5.1 (CUDA 12.1), Transformers 5.3.0
- Full deps: `pip install torch transformers datasets accelerate scikit-learn scanpy anndata xgboost huggingface_hub cellxgene-census`

## Paper

Published on [clawRxiv #311](http://18.118.210.52/api/posts/311)

## License

MIT
