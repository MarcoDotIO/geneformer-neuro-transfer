"""Tokenize scRNA-seq data for Geneformer rank-value encoding."""
import os, pickle, numpy as np, scanpy as sc
from scipy.sparse import issparse

DATA_DIR = "/data/marnett5/neuro_transfer/data"
MODEL_DIR = "/data/marnett5/neuro_transfer/models/geneformer_pretrained"
TOKEN_DIR = os.path.join(DATA_DIR, "tokenized")
os.makedirs(TOKEN_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "geneformer/token_dictionary_gc104M.pkl"), "rb") as f:
    token_dict = pickle.load(f)
with open(os.path.join(MODEL_DIR, "geneformer/gene_median_dictionary_gc104M.pkl"), "rb") as f:
    gene_median_dict = pickle.load(f)

gene_to_token = {k: v for k, v in token_dict.items() if k.startswith("ENSG")}
PAD_TOKEN, CLS_TOKEN, MAX_LEN = token_dict["<pad>"], token_dict["<cls>"], 2048
print(f"Gene tokens: {len(gene_to_token)}")

def tokenize_cell(expr, gene_ids):
    if issparse(expr): expr = expr.toarray().flatten()
    else: expr = np.asarray(expr).flatten()
    nz = expr > 0
    if nz.sum() == 0: return [CLS_TOKEN] + [PAD_TOKEN]*(MAX_LEN-1)
    nz_genes, nz_expr = gene_ids[nz], expr[nz]
    ratios, valid_genes = [], []
    for g, e in zip(nz_genes, nz_expr):
        if g in gene_median_dict and g in gene_to_token:
            m = gene_median_dict[g]
            if m > 0: ratios.append(e/m); valid_genes.append(g)
    if not ratios: return [CLS_TOKEN] + [PAD_TOKEN]*(MAX_LEN-1)
    order = np.argsort(ratios)[::-1]
    tokens = [CLS_TOKEN] + [gene_to_token[valid_genes[i]] for i in order[:MAX_LEN-1]]
    tokens += [PAD_TOKEN]*(MAX_LEN - len(tokens))
    return tokens

def tokenize_dataset(name):
    path = os.path.join(DATA_DIR, f"{name}.h5ad")
    if not os.path.exists(path):
        print(f"Skipping {name} - not found"); return
    print(f"\nTokenizing {name}...")
    adata = sc.read_h5ad(path)
    print(f"  Shape: {adata.shape}")
    # Find ensembl IDs
    gene_ids = None
    if "feature_id" in adata.var.columns:
        gene_ids = adata.var["feature_id"].values
    elif adata.var.index[0].startswith("ENSG"):
        gene_ids = adata.var.index.values
    else:
        for col in adata.var.columns:
            sample = adata.var[col].iloc[0]
            if isinstance(sample, str) and sample.startswith("ENSG"):
                gene_ids = adata.var[col].values; break
    if gene_ids is None:
        print(f"  ERROR: No Ensembl IDs found. Cols: {adata.var.columns.tolist()}"); return
    print(f"  Gene IDs: {gene_ids[:3]}")
    
    all_tokens = []
    for i in range(len(adata)):
        if i % 5000 == 0: print(f"  Cell {i}/{len(adata)}...")
        all_tokens.append(tokenize_cell(adata.X[i], gene_ids))
    all_tokens = np.array(all_tokens, dtype=np.int32)
    attn_mask = (all_tokens != PAD_TOKEN).astype(np.int32)
    
    output = {
        "input_ids": all_tokens, "attention_mask": attn_mask,
        "labels": np.array(adata.obs["label"].values if "label" in adata.obs else np.zeros(len(adata)), dtype=np.int32),
        "cell_types": np.array(adata.obs["cell_type"].values if "cell_type" in adata.obs else ["unknown"]*len(adata)),
        "disease": adata.obs["disease"].values if "disease" in adata.obs else None,
        "donor_id": adata.obs["donor_id"].values if "donor_id" in adata.obs else None,
    }
    outpath = os.path.join(TOKEN_DIR, f"{name}.pkl")
    with open(outpath, "wb") as f: pickle.dump(output, f)
    avg_len = (all_tokens != PAD_TOKEN).sum(axis=1).mean()
    print(f"  Saved: {outpath} | Cells: {len(all_tokens)} | Avg tokens: {avg_len:.0f}")
    print(f"  Labels: {np.bincount(output['labels'])}")

for name in ["ad_brain", "pd_brain", "als_brain"]:
    tokenize_dataset(name)
print("\nTokenization complete!")
