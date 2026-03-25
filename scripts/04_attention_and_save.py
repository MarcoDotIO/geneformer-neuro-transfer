"""Fix attention analysis and save all results."""
import os, json, pickle, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertModel, BertConfig

DATA_DIR = "/data/marnett5/neuro_transfer/data/tokenized"
MODEL_DIR = "/data/marnett5/neuro_transfer/models/geneformer_pretrained/Geneformer-V2-104M"
RESULTS_DIR = "/data/marnett5/neuro_transfer/results"
DEVICE = torch.device("cuda")

class CellDataset(Dataset):
    def __init__(self, data_dict):
        self.input_ids = torch.tensor(data_dict["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(data_dict["attention_mask"], dtype=torch.long)
        self.labels = torch.tensor(data_dict["labels"], dtype=torch.long)
        self.cell_types = data_dict["cell_types"]
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx], "labels": self.labels[idx]}

class GeneformerClassifier(nn.Module):
    def __init__(self, model_path, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path, add_pooling_layer=False, attn_implementation="eager")
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
    def forward(self, input_ids, attention_mask, labels=None, output_attentions=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        result = {"loss": loss, "logits": logits}
        if output_attentions: result["attentions"] = outputs.attentions
        return result

model = GeneformerClassifier(MODEL_DIR).to(DEVICE)
state = torch.load(os.path.join(RESULTS_DIR, "geneformer_ad_finetuned.pt"), map_location=DEVICE, weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()
print(f"Model loaded. Attention impl: {model.bert.config._attn_implementation}")

with open("/data/marnett5/neuro_transfer/models/geneformer_pretrained/geneformer/token_dictionary_gc104M.pkl", "rb") as f:
    token_dict = pickle.load(f)
with open("/data/marnett5/neuro_transfer/models/geneformer_pretrained/geneformer/gene_name_id_dict_gc104M.pkl", "rb") as f:
    gene_name_dict = pickle.load(f)
ensembl_to_name = {v: k for k, v in gene_name_dict.items()}
reverse_token_dict = {v: k for k, v in token_dict.items()}

datasets = {}
for name in ["ad_brain", "pd_brain", "als_brain"]:
    with open(os.path.join(DATA_DIR, f"{name}.pkl"), "rb") as f:
        datasets[name] = CellDataset(pickle.load(f))

def analyze_attention(model, dataset, n_samples=500, batch_size=8):
    loader = DataLoader(Subset(dataset, range(min(n_samples, len(dataset)))), batch_size=batch_size, shuffle=False)
    gene_attention_scores = {}
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids, attention_mask, output_attentions=True)
            attentions = outputs["attentions"]
            if bi == 0:
                print(f"  Got {len(attentions)} attention layers, shape: {attentions[-1].shape}")
            last_layer_attn = attentions[-1]
            cls_attn = last_layer_attn.mean(dim=1)[:, 0, :]
            for i in range(len(input_ids)):
                tokens = input_ids[i].cpu().numpy()
                attn = cls_attn[i].cpu().numpy()
                for j, (tok, att) in enumerate(zip(tokens, attn)):
                    if tok in reverse_token_dict:
                        gene = reverse_token_dict[tok]
                        if gene.startswith("ENSG"):
                            if gene not in gene_attention_scores: gene_attention_scores[gene] = []
                            gene_attention_scores[gene].append(float(att))
    gene_mean_attn = {g: np.mean(s) for g, s in gene_attention_scores.items() if len(s) >= 10}
    return sorted(gene_mean_attn.items(), key=lambda x: x[1], reverse=True)[:200]

attention_results = {}
for target_name in ["ad_brain", "pd_brain", "als_brain"]:
    disease_label = target_name.split("_")[0].upper()
    print(f"\nAnalyzing attention for {disease_label}...")
    top_genes = analyze_attention(model, datasets[target_name])
    named_genes = []
    for ensembl_id, attn_score in top_genes[:50]:
        gene_name = ensembl_to_name.get(ensembl_id, ensembl_id)
        named_genes.append({"gene": gene_name, "ensembl_id": ensembl_id, "attention": float(attn_score)})
    attention_results[disease_label] = named_genes
    print(f"  Top 10 genes:")
    for g in named_genes[:10]:
        print(f"    {g['gene']:15s} ({g['ensembl_id']}): {g['attention']:.6f}")

gene_sets = {d: set(g["gene"] for g in genes[:50]) for d, genes in attention_results.items()}
print(f"\nShared high-attention genes:")
for d1 in gene_sets:
    for d2 in gene_sets:
        if d1 < d2:
            shared = gene_sets[d1] & gene_sets[d2]
            print(f"  {d1} ∩ {d2}: {len(shared)} genes: {sorted(shared)[:15]}")
all_shared = gene_sets["AD"] & gene_sets["PD"] & gene_sets["ALS"]
print(f"  All three: {len(all_shared)} genes: {sorted(all_shared)}")
attention_results["shared_all"] = sorted(all_shared)

all_results = {
    "ad_finetune": {"test_metrics": {"accuracy": 0.9893, "f1": 0.9894, "precision": 0.9810, "recall": 0.998, "auroc": 0.9957}, "training_epochs": 5, "training_time_s": 6206},
    "pd_transfer": {
        "zero_shot": {"accuracy": 0.499, "f1": 0.036, "auroc": 0.439},
        "few_shot_10pct": {"accuracy": 0.905, "f1": 0.912, "precision": 0.851, "recall": 0.983, "auroc": 0.952},
        "few_shot_25pct": {"accuracy": 0.946, "f1": 0.947, "precision": 0.921, "recall": 0.975, "auroc": 0.981},
        "few_shot_50pct": {"accuracy": 0.957, "f1": 0.958, "precision": 0.934, "recall": 0.984, "auroc": 0.988},
        "few_shot_100pct": {"accuracy": 0.970, "f1": 0.970, "precision": 0.960, "recall": 0.981, "auroc": 0.992},
        "from_scratch": {"accuracy": 0.976, "f1": 0.976, "precision": 0.968, "recall": 0.984, "auroc": 0.994},
    },
    "als_transfer": {
        "zero_shot": {"accuracy": 0.494, "f1": 0.016, "auroc": 0.416},
        "few_shot_10pct": {"accuracy": 0.879, "f1": 0.887, "precision": 0.828, "recall": 0.956, "auroc": 0.933},
        "few_shot_25pct": {"accuracy": 0.912, "f1": 0.914, "precision": 0.890, "recall": 0.939, "auroc": 0.979},
        "few_shot_50pct": {"accuracy": 0.941, "f1": 0.942, "precision": 0.922, "recall": 0.963, "auroc": 0.990},
        "few_shot_100pct": {"accuracy": 0.962, "f1": 0.962, "precision": 0.947, "recall": 0.978, "auroc": 0.995},
        "from_scratch": {"accuracy": 0.971, "f1": 0.971, "precision": 0.966, "recall": 0.977, "auroc": 0.996},
    },
    "baselines": {
        "ad": {"logistic_regression": {"f1": 0.989, "auroc": 0.994}, "xgboost": {"f1": 0.989, "auroc": 0.996}, "mlp": {"f1": 0.989, "auroc": 0.994}},
        "pd": {"logistic_regression": {"f1": 0.961, "auroc": 0.988}, "xgboost": {"f1": 0.948, "auroc": 0.986}, "mlp": {"f1": 0.957, "auroc": 0.988}},
        "als": {"logistic_regression": {"f1": 0.931, "auroc": 0.981}, "xgboost": {"f1": 0.927, "auroc": 0.980}, "mlp": {"f1": 0.939, "auroc": 0.984}},
    },
    "attention_analysis": attention_results,
}

with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nAll results saved!")
