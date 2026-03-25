"""
Download scRNA-seq datasets for AD, PD, and ALS from direct sources.
More memory-efficient than CellxGene Census full queries.
"""
import os
import subprocess
import sys
import numpy as np

DATA_DIR = os.path.expanduser("~/neuro_transfer/data")
os.makedirs(DATA_DIR, exist_ok=True)

# Install cellxgene-census if not present
try:
    import cellxgene_census
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cellxgene-census", "-q"])
    import cellxgene_census

import scanpy as sc
import anndata as ad

print("Opening CellxGene Census (stable release)...")
census = cellxgene_census.open_soma(census_version="2025-11-08")

def download_small_batch(disease_ontology, tissue_general, filename, max_cells=15000):
    """Download a smaller, targeted subset using obs-first approach."""
    print(f"\n{'='*60}")
    print(f"Downloading: {disease_ontology} from {tissue_general}")
    
    # Step 1: Get obs metadata only (lightweight)
    print("  Fetching disease cell metadata...")
    disease_filter = f'tissue_general == "{tissue_general}" and disease == "{disease_ontology}" and is_primary_data == True'
    
    obs_disease = cellxgene_census.get_obs(
        census, "Homo sapiens",
        value_filter=disease_filter,
        column_names=["soma_joinid", "cell_type", "tissue", "disease", "donor_id", "dataset_id"],
    )
    n_disease = len(obs_disease)
    print(f"  Found {n_disease} disease cells")
    
    print("  Fetching control cell metadata...")
    control_filter = f'tissue_general == "{tissue_general}" and disease == "normal" and is_primary_data == True'
    
    obs_control = cellxgene_census.get_obs(
        census, "Homo sapiens",
        value_filter=control_filter,
        column_names=["soma_joinid", "cell_type", "tissue", "disease", "donor_id", "dataset_id"],
    )
    n_control = len(obs_control)
    print(f"  Found {n_control} control cells")
    
    # Step 2: Sample balanced subset
    n_per_class = min(max_cells // 2, n_disease, n_control)
    print(f"  Sampling {n_per_class} per class...")
    
    np.random.seed(42)
    if n_disease > n_per_class:
        disease_sample = obs_disease.sample(n=n_per_class, random_state=42)
    else:
        disease_sample = obs_disease
    
    if n_control > n_per_class:
        control_sample = obs_control.sample(n=n_per_class, random_state=42)
    else:
        control_sample = obs_control
    
    # Step 3: Download expression for selected cells only
    selected_ids = list(disease_sample["soma_joinid"].values) + list(control_sample["soma_joinid"].values)
    print(f"  Downloading expression for {len(selected_ids)} cells...")
    
    adata = cellxgene_census.get_anndata(
        census,
        organism="Homo sapiens",
        obs_coords=selected_ids,
        obs_column_names=["cell_type", "tissue", "disease", "donor_id", "dataset_id", "assay"],
    )
    
    # Add binary label
    adata.obs["label"] = (adata.obs["disease"] != "normal").astype(int)
    
    outpath = os.path.join(DATA_DIR, filename)
    adata.write_h5ad(outpath)
    
    print(f"  Saved: {outpath}")
    print(f"  Shape: {adata.shape}")
    print(f"  Disease dist: {adata.obs['disease'].value_counts().to_dict()}")
    print(f"  Label dist: {adata.obs['label'].value_counts().to_dict()}")
    print(f"  Top cell types: {adata.obs['cell_type'].value_counts().head(5).to_dict()}")
    
    return adata

# Download each disease
print("="*60)
print("DOWNLOADING NEURODEGENERATION DATASETS")
print("="*60)

ad_data = download_small_batch(
    "Alzheimer disease", "brain", "ad_brain.h5ad", max_cells=20000
)

pd_data = download_small_batch(
    "Parkinson disease", "brain", "pd_brain.h5ad", max_cells=20000
)

als_data = download_small_batch(
    "amyotrophic lateral sclerosis", "brain", "als_brain.h5ad", max_cells=20000
)

census.close()

print("\n" + "="*60)
print("ALL DOWNLOADS COMPLETE")
print(f"AD: {ad_data.shape}")
print(f"PD: {pd_data.shape}")
print(f"ALS: {als_data.shape}")
