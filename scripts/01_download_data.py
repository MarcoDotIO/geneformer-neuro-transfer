"""
Download scRNA-seq datasets for AD, PD, and ALS from CellxGene Census.
Memory-efficient: fetches obs metadata first, then expression for sampled cells.
"""
import os, subprocess, sys
import numpy as np

DATA_DIR = "/data/marnett5/neuro_transfer/data"
os.makedirs(DATA_DIR, exist_ok=True)

try:
    import cellxgene_census
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cellxgene-census", "-q"])
    import cellxgene_census

print("Opening CellxGene Census...")
census = cellxgene_census.open_soma(census_version="2025-11-08")

def download_disease(disease_ontology, tissue, filename, max_cells=15000):
    print(f"\n{'='*60}\nDownloading: {disease_ontology} / {tissue}")

    # Obs metadata only
    disease_filter = f'tissue_general == "{tissue}" and disease == "{disease_ontology}" and is_primary_data == True'
    obs_d = cellxgene_census.get_obs(census, "Homo sapiens", value_filter=disease_filter,
        column_names=["soma_joinid","cell_type","tissue","disease","donor_id"])
    print(f"  Disease cells: {len(obs_d)}")

    control_filter = f'tissue_general == "{tissue}" and disease == "normal" and is_primary_data == True'
    obs_c = cellxgene_census.get_obs(census, "Homo sapiens", value_filter=control_filter,
        column_names=["soma_joinid","cell_type","tissue","disease","donor_id"])
    print(f"  Control cells: {len(obs_c)}")

    n = min(max_cells // 2, len(obs_d), len(obs_c))
    print(f"  Sampling {n} per class...")
    np.random.seed(42)
    d_sample = obs_d.sample(n=n, random_state=42) if len(obs_d) > n else obs_d
    c_sample = obs_c.sample(n=n, random_state=42) if len(obs_c) > n else obs_c

    ids = list(d_sample["soma_joinid"].values) + list(c_sample["soma_joinid"].values)
    print(f"  Downloading expression for {len(ids)} cells...")
    adata = cellxgene_census.get_anndata(census, organism="Homo sapiens",
        obs_coords=ids, obs_column_names=["cell_type","tissue","disease","donor_id","assay"])

    adata.obs["label"] = (adata.obs["disease"] != "normal").astype(int)
    outpath = os.path.join(DATA_DIR, filename)
    adata.write_h5ad(outpath)
    print(f"  Saved: {outpath} | Shape: {adata.shape}")
    print(f"  Labels: {adata.obs['label'].value_counts().to_dict()}")
    print(f"  Top cell types: {adata.obs['cell_type'].value_counts().head(5).to_dict()}")
    return adata

ad = download_disease("Alzheimer disease", "brain", "ad_brain.h5ad", 20000)
pd_ = download_disease("Parkinson disease", "brain", "pd_brain.h5ad", 20000)
als = download_disease("amyotrophic lateral sclerosis", "brain", "als_brain.h5ad", 20000)

census.close()
print(f"\nDone! AD:{ad.shape} PD:{pd_.shape} ALS:{als.shape}")
