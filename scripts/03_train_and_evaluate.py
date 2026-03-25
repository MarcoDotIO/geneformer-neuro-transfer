"""
Fine-tune Geneformer on AD, then evaluate transfer to PD and ALS.
Includes baselines (logistic regression, XGBoost, MLP) and attention analysis.
"""
import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertModel, BertConfig
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report
)
import xgboost as xgb
import time

# Paths
DATA_DIR = os.path.expanduser("/data/marnett5/neuro_transfer/data/tokenized")
MODEL_DIR = os.path.expanduser("/data/marnett5/neuro_transfer/models/geneformer_pretrained/Geneformer-V2-104M")
RESULTS_DIR = os.path.expanduser("/data/marnett5/neuro_transfer/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# Dataset
# ============================================================
class CellDataset(Dataset):
    def __init__(self, data_dict):
        self.input_ids = torch.tensor(data_dict["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(data_dict["attention_mask"], dtype=torch.long)
        self.labels = torch.tensor(data_dict["labels"], dtype=torch.long)
        self.cell_types = data_dict["cell_types"]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

# ============================================================
# Model: Geneformer + Classification Head
# ============================================================
class GeneformerClassifier(nn.Module):
    def __init__(self, model_path, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path, add_pooling_layer=False)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
    
    def forward(self, input_ids, attention_mask, labels=None, output_attentions=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # Use CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        result = {"loss": loss, "logits": logits}
        if output_attentions:
            result["attentions"] = outputs.attentions
        return result
    
    def get_embeddings(self, input_ids, attention_mask):
        """Extract CLS embeddings for baseline classifiers."""
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# ============================================================
# Training
# ============================================================
def train_model(model, train_loader, val_loader, epochs=5, lr=2e-5, warmup_steps=500):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Linear warmup + cosine decay
    total_steps = len(train_loader) * epochs
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_f1 = 0
    best_state = None
    history = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_f1 = f1_score(train_labels, train_preds, average="binary")
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validate
        val_metrics = evaluate_model(model, val_loader)
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "train_acc": train_acc,
            "train_f1": train_f1,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        
        print(f"Epoch {epoch+1}: Train F1={train_f1:.4f}, Val F1={val_metrics['f1']:.4f}, "
              f"Val Acc={val_metrics['accuracy']:.4f}, Val AUC={val_metrics['auroc']:.4f}")
        
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    return history

def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"]
            
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1].cpu().numpy()
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="binary"),
        "precision": precision_score(all_labels, all_preds, average="binary"),
        "recall": recall_score(all_labels, all_preds, average="binary"),
        "auroc": roc_auc_score(all_labels, all_probs),
    }

# ============================================================
# Extract embeddings for baselines
# ============================================================
def extract_embeddings(model, dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            emb = model.get_embeddings(input_ids, attention_mask)
            embeddings.append(emb)
    
    return np.concatenate(embeddings, axis=0)

# ============================================================
# Baseline classifiers
# ============================================================
def run_baselines(train_emb, train_labels, test_emb, test_labels):
    results = {}
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(train_emb, train_labels)
    lr_preds = lr.predict(test_emb)
    lr_probs = lr.predict_proba(test_emb)[:, 1]
    results["logistic_regression"] = {
        "accuracy": accuracy_score(test_labels, lr_preds),
        "f1": f1_score(test_labels, lr_preds, average="binary"),
        "precision": precision_score(test_labels, lr_preds, average="binary"),
        "recall": recall_score(test_labels, lr_preds, average="binary"),
        "auroc": roc_auc_score(test_labels, lr_probs),
    }
    
    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="logloss", verbosity=0
    )
    xgb_clf.fit(train_emb, train_labels)
    xgb_preds = xgb_clf.predict(test_emb)
    xgb_probs = xgb_clf.predict_proba(test_emb)[:, 1]
    results["xgboost"] = {
        "accuracy": accuracy_score(test_labels, xgb_preds),
        "f1": f1_score(test_labels, xgb_preds, average="binary"),
        "precision": precision_score(test_labels, xgb_preds, average="binary"),
        "recall": recall_score(test_labels, xgb_preds, average="binary"),
        "auroc": roc_auc_score(test_labels, xgb_probs),
    }
    
    # MLP
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
    mlp.fit(train_emb, train_labels)
    mlp_preds = mlp.predict(test_emb)
    mlp_probs = mlp.predict_proba(test_emb)[:, 1]
    results["mlp"] = {
        "accuracy": accuracy_score(test_labels, mlp_preds),
        "f1": f1_score(test_labels, mlp_preds, average="binary"),
        "precision": precision_score(test_labels, mlp_preds, average="binary"),
        "recall": recall_score(test_labels, mlp_preds, average="binary"),
        "auroc": roc_auc_score(test_labels, mlp_probs),
    }
    
    return results

# ============================================================
# Attention analysis
# ============================================================
def analyze_attention(model, dataset, token_dict, n_samples=500, batch_size=32):
    """Extract attention weights and identify top-attended genes."""
    reverse_token_dict = {v: k for k, v in token_dict.items()}
    
    loader = DataLoader(
        Subset(dataset, range(min(n_samples, len(dataset)))),
        batch_size=batch_size, shuffle=False
    )
    
    gene_attention_scores = {}
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            outputs = model(input_ids, attention_mask, output_attentions=True)
            
            # Average attention across all heads in last layer
            # Shape: (batch, heads, seq_len, seq_len)
            last_layer_attn = outputs["attentions"][-1]
            # Average across heads, get attention FROM CLS token
            cls_attn = last_layer_attn.mean(dim=1)[:, 0, :]  # (batch, seq_len)
            
            for i in range(len(input_ids)):
                tokens = input_ids[i].cpu().numpy()
                attn = cls_attn[i].cpu().numpy()
                
                for j, (tok, att) in enumerate(zip(tokens, attn)):
                    if tok in reverse_token_dict:
                        gene = reverse_token_dict[tok]
                        if gene.startswith("ENSG"):
                            if gene not in gene_attention_scores:
                                gene_attention_scores[gene] = []
                            gene_attention_scores[gene].append(att)
    
    # Compute mean attention per gene
    gene_mean_attn = {
        gene: np.mean(scores)
        for gene, scores in gene_attention_scores.items()
        if len(scores) >= 10  # require minimum observations
    }
    
    # Sort by attention
    sorted_genes = sorted(gene_mean_attn.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_genes[:200]  # Top 200 genes

# ============================================================
# Main experiment
# ============================================================
def main():
    all_results = {}
    
    # Load tokenized data
    print("Loading tokenized datasets...")
    datasets = {}
    for name in ["ad_brain", "pd_brain", "als_brain"]:
        path = os.path.join(DATA_DIR, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            datasets[name] = CellDataset(data)
            print(f"  {name}: {len(datasets[name])} cells, "
                  f"labels: {np.bincount(data['labels'])}")
        else:
            print(f"  {name}: NOT FOUND")
    
    if "ad_brain" not in datasets:
        print("ERROR: AD dataset not found. Exiting.")
        return
    
    # ========================================
    # Phase 1: Fine-tune on AD
    # ========================================
    print("\n" + "="*60)
    print("PHASE 1: Fine-tuning Geneformer on Alzheimer's Disease")
    print("="*60)
    
    ad_dataset = datasets["ad_brain"]
    
    # Train/val/test split (70/15/15), stratified
    labels = ad_dataset.labels.numpy()
    train_idx, temp_idx = train_test_split(
        range(len(ad_dataset)), test_size=0.3, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        stratify=labels[temp_idx], random_state=42
    )
    
    train_set = Subset(ad_dataset, train_idx)
    val_set = Subset(ad_dataset, val_idx)
    test_set = Subset(ad_dataset, test_idx)
    
    print(f"AD splits - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)
    
    # Initialize model
    model = GeneformerClassifier(MODEL_DIR).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Train
    t0 = time.time()
    history = train_model(model, train_loader, val_loader, epochs=5, lr=2e-5)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.0f}s")
    
    # Evaluate on AD test set
    ad_test_metrics = evaluate_model(model, test_loader)
    print(f"\nAD Test Results: {ad_test_metrics}")
    all_results["ad_finetune"] = {
        "metrics": ad_test_metrics,
        "history": history,
        "train_time_s": train_time,
    }
    
    # Save fine-tuned model
    ft_model_path = os.path.join(RESULTS_DIR, "geneformer_ad_finetuned.pt")
    torch.save(model.state_dict(), ft_model_path)
    print(f"Saved fine-tuned model to {ft_model_path}")
    
    # ========================================
    # Phase 2: Transfer to PD and ALS
    # ========================================
    for target_name in ["pd_brain", "als_brain"]:
        if target_name not in datasets:
            print(f"\nSkipping {target_name} - not available")
            continue
        
        disease_label = "PD" if "pd" in target_name else "ALS"
        print(f"\n{'='*60}")
        print(f"PHASE 2: Transfer Learning to {disease_label}")
        print("="*60)
        
        target_dataset = datasets[target_name]
        target_labels = target_dataset.labels.numpy()
        
        # Split target data
        target_train_idx, target_test_idx = train_test_split(
            range(len(target_dataset)), test_size=0.3,
            stratify=target_labels, random_state=42
        )
        target_test_set = Subset(target_dataset, target_test_idx)
        target_test_loader = DataLoader(target_test_set, batch_size=64, shuffle=False, num_workers=4)
        
        transfer_results = {}
        
        # --- Zero-shot: Apply AD model directly ---
        print(f"\n--- Zero-shot transfer to {disease_label} ---")
        zero_shot_metrics = evaluate_model(model, target_test_loader)
        print(f"Zero-shot: {zero_shot_metrics}")
        transfer_results["zero_shot"] = zero_shot_metrics
        
        # --- Few-shot: Fine-tune with 10%, 25%, 50% of target train data ---
        for frac in [0.10, 0.25, 0.50, 1.0]:
            label = f"few_shot_{int(frac*100)}pct"
            print(f"\n--- {label} transfer to {disease_label} ---")
            
            n_samples = max(int(len(target_train_idx) * frac), 10)
            np.random.seed(42)
            subset_idx = np.random.choice(target_train_idx, n_samples, replace=False)
            
            # Further split into train/val
            sub_labels = target_labels[subset_idx]
            if len(np.unique(sub_labels)) < 2:
                print(f"  Skipping - only one class in subset")
                continue
            
            sub_train, sub_val = train_test_split(
                subset_idx, test_size=0.2, stratify=sub_labels, random_state=42
            )
            
            sub_train_set = Subset(target_dataset, sub_train)
            sub_val_set = Subset(target_dataset, sub_val)
            
            sub_train_loader = DataLoader(sub_train_set, batch_size=32, shuffle=True, num_workers=4)
            sub_val_loader = DataLoader(sub_val_set, batch_size=64, shuffle=False, num_workers=4)
            
            # Re-initialize from AD fine-tuned weights
            ft_model = GeneformerClassifier(MODEL_DIR).to(DEVICE)
            ft_model.load_state_dict(torch.load(ft_model_path, map_location=DEVICE, weights_only=True))
            
            # Fine-tune with lower LR
            ft_history = train_model(
                ft_model, sub_train_loader, sub_val_loader,
                epochs=3, lr=1e-5, warmup_steps=100
            )
            
            ft_metrics = evaluate_model(ft_model, target_test_loader)
            print(f"{label}: {ft_metrics}")
            transfer_results[label] = ft_metrics
            
            del ft_model
            torch.cuda.empty_cache()
        
        # --- Train from scratch on target (no transfer) ---
        print(f"\n--- Train from scratch on {disease_label} (no transfer) ---")
        scratch_model = GeneformerClassifier(MODEL_DIR).to(DEVICE)
        
        scratch_train_set = Subset(target_dataset, target_train_idx)
        scratch_train_loader = DataLoader(scratch_train_set, batch_size=32, shuffle=True, num_workers=4)
        
        # Use a portion as val
        scratch_train_sub, scratch_val_sub = train_test_split(
            target_train_idx, test_size=0.15,
            stratify=target_labels[target_train_idx], random_state=42
        )
        scratch_val_loader = DataLoader(
            Subset(target_dataset, scratch_val_sub), batch_size=64, shuffle=False, num_workers=4
        )
        scratch_train_loader2 = DataLoader(
            Subset(target_dataset, scratch_train_sub), batch_size=32, shuffle=True, num_workers=4
        )
        
        scratch_history = train_model(
            scratch_model, scratch_train_loader2, scratch_val_loader,
            epochs=5, lr=2e-5
        )
        scratch_metrics = evaluate_model(scratch_model, target_test_loader)
        print(f"From scratch: {scratch_metrics}")
        transfer_results["from_scratch"] = scratch_metrics
        
        del scratch_model
        torch.cuda.empty_cache()
        
        all_results[f"{disease_label.lower()}_transfer"] = transfer_results
    
    # ========================================
    # Phase 3: Baseline classifiers
    # ========================================
    print(f"\n{'='*60}")
    print("PHASE 3: Baseline Classifiers (on Geneformer embeddings)")
    print("="*60)
    
    # Extract embeddings using AD-finetuned model
    model.eval()
    
    for target_name in ["ad_brain", "pd_brain", "als_brain"]:
        if target_name not in datasets:
            continue
        
        disease_label = target_name.split("_")[0].upper()
        print(f"\n--- Baselines for {disease_label} ---")
        
        target_dataset = datasets[target_name]
        target_labels = target_dataset.labels.numpy()
        
        # Extract embeddings
        print("  Extracting embeddings...")
        all_emb = extract_embeddings(model, target_dataset, batch_size=64)
        
        # Split
        train_idx, test_idx = train_test_split(
            range(len(target_dataset)), test_size=0.3,
            stratify=target_labels, random_state=42
        )
        
        train_emb = all_emb[train_idx]
        test_emb = all_emb[test_idx]
        train_lab = target_labels[train_idx]
        test_lab = target_labels[test_idx]
        
        baseline_results = run_baselines(train_emb, train_lab, test_emb, test_lab)
        
        for method, metrics in baseline_results.items():
            print(f"  {method}: F1={metrics['f1']:.4f}, AUC={metrics['auroc']:.4f}")
        
        all_results[f"{disease_label.lower()}_baselines"] = baseline_results
    
    # ========================================
    # Phase 4: Attention analysis
    # ========================================
    print(f"\n{'='*60}")
    print("PHASE 4: Attention Analysis")
    print("="*60)
    
    # Load token dict for gene mapping
    token_dict_path = os.path.expanduser(
        "/data/marnett5/neuro_transfer/models/geneformer_pretrained/geneformer/token_dictionary_gc104M.pkl"
    )
    with open(token_dict_path, "rb") as f:
        token_dict = pickle.load(f)
    
    # Load gene name mapping
    gene_name_path = os.path.expanduser(
        "/data/marnett5/neuro_transfer/models/geneformer_pretrained/geneformer/gene_name_id_dict_gc104M.pkl"
    )
    with open(gene_name_path, "rb") as f:
        gene_name_dict = pickle.load(f)
    ensembl_to_name = {v: k for k, v in gene_name_dict.items()}
    
    attention_results = {}
    for target_name in ["ad_brain", "pd_brain", "als_brain"]:
        if target_name not in datasets:
            continue
        
        disease_label = target_name.split("_")[0].upper()
        print(f"\nAnalyzing attention for {disease_label}...")
        
        top_genes = analyze_attention(model, datasets[target_name], token_dict, n_samples=500)
        
        # Map to gene names
        named_genes = []
        for ensembl_id, attn_score in top_genes[:50]:
            gene_name = ensembl_to_name.get(ensembl_id, ensembl_id)
            named_genes.append({"gene": gene_name, "ensembl_id": ensembl_id, "attention": float(attn_score)})
            print(f"  {gene_name:15s} ({ensembl_id}): {attn_score:.6f}")
        
        attention_results[disease_label] = named_genes
    
    # Find shared top genes across diseases
    if len(attention_results) >= 2:
        print("\n--- Shared high-attention genes across diseases ---")
        gene_sets = {}
        for disease, genes in attention_results.items():
            gene_sets[disease] = set(g["gene"] for g in genes[:50])
        
        diseases = list(gene_sets.keys())
        for i in range(len(diseases)):
            for j in range(i+1, len(diseases)):
                shared = gene_sets[diseases[i]] & gene_sets[diseases[j]]
                print(f"  {diseases[i]} ∩ {diseases[j]}: {len(shared)} shared genes")
                if shared:
                    print(f"    {sorted(shared)[:20]}")
        
        if len(diseases) >= 3:
            all_shared = gene_sets[diseases[0]] & gene_sets[diseases[1]] & gene_sets[diseases[2]]
            print(f"  All three: {len(all_shared)} shared genes")
            if all_shared:
                print(f"    {sorted(all_shared)}")
            attention_results["shared_all"] = sorted(all_shared)
    
    all_results["attention_analysis"] = attention_results
    
    # ========================================
    # Save all results
    # ========================================
    results_path = os.path.join(RESULTS_DIR, "all_results.json")
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    
    print(f"\nAll results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(json.dumps(all_results, indent=2, default=convert))

if __name__ == "__main__":
    main()
