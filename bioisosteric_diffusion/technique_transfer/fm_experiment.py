#!/usr/bin/env python3
"""Factorization Machine experiment: LR vs FM on LBC-Ranker features (3 seeds)."""
import json, glob, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

SEED = 20260603
OUT = Path("technique_transfer/results")
OUT.mkdir(parents=True, exist_ok=True)
RDLogger.DisableLog("rdApp.*")

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Factorization Machine (PyTorch) ──
class FactorizationMachine(nn.Module):
    """FM with L2 regularization. Output: sigmoid probability."""
    def __init__(self, n_features, k=8):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(n_features))
        self.V = nn.Parameter(torch.randn(n_features, k) * 0.01)  # latent factors

    def forward(self, x):
        # Linear part
        linear = self.w0 + (x @ self.w)
        # Pairwise interaction: 0.5 * (||V^T x||^2 - ||V^T (x^2)||_2)
        vx = x @ self.V  # [batch, k]
        vx_sq = vx ** 2
        x_sq = x ** 2
        v2_x2 = x_sq @ (self.V ** 2)  # [batch, k]
        interactions = 0.5 * (vx_sq - v2_x2).sum(dim=1)
        return torch.sigmoid(linear + interactions)

def train_fm(X, y, k=8, lr=0.01, l2=0.001, epochs=30, batch_size=4096, verbose=False):
    """Train FM with BCE loss + L2 regularization."""
    device = torch.device("cpu")
    n, d = X.shape
    model = FactorizationMachine(d, k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            Xb = X_t[i:end]
            yb = y_t[i:end]
            preds = model(Xb)
            loss = nn.BCELoss()(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (end - i)
        if verbose and (epoch + 1) % 10 == 0:
            log(f"    FM epoch {epoch+1}/{epochs}, loss={total_loss/n:.6f}")

    model.eval()
    return model

def predict_fm(model, X, scaler):
    X_scaled = torch.FloatTensor(scaler.transform(X))
    with torch.no_grad():
        return model(X_scaled).numpy()

# ── LambdaRank loss (simple gradient-weighted BCE) ──
def lambdarank_loss(scores, labels, sigma=1.0):
    """Approximate NDCG-weighted pairwise loss.
    For each query, weight BCE by |delta_NDCG| if two items swap.
    Simplified: weight each positive by (1/rank_pos) and each negative by (1/rank_neg).
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Sort by score to get approximate ranks
    _, indices = torch.sort(scores, descending=True)
    ranks = torch.zeros_like(scores)
    ranks[indices] = torch.arange(1, len(scores) + 1, dtype=torch.float32)

    # Weight: higher weight for top positions
    pos_weights = 1.0 / torch.log2(ranks[pos_mask] + 1)
    neg_weights = 1.0 / torch.log2(ranks[neg_mask] + 1)

    # Weighted BCE
    pos_loss = -torch.log(scores[pos_mask] + 1e-8) * pos_weights
    neg_loss = -torch.log(1 - scores[neg_mask] + 1e-8) * neg_weights
    return (pos_loss.sum() / pos_weights.sum() + neg_loss.sum() / neg_weights.sum()) / 2

# ── Data loading (same as baseline) ──
def mfp(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    fp = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), fp)
    return fp

def mprops(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    return np.array([m.GetNumHeavyAtoms(), Descriptors.RingCount(m),
        Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m)], dtype=np.float32)

def load_data():
    log("Loading...")
    manifest = {}
    mp = Path("plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl")
    with mp.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): q = json.loads(line); manifest[q["query_id"]] = q
    all_ofs = sorted({q["old_fragment_smiles"] for q in manifest.values()})
    of_fps, of_props = {}, {}
    for s in all_ofs:
        fp = mfp(s); props = mprops(s)
        if fp is not None and props is not None: of_fps[s] = fp; of_props[s] = props

    of_queries = defaultdict(list)
    for q in manifest.values():
        if q["old_fragment_smiles"] in of_fps: of_queries[q["old_fragment_smiles"]].append(q["query_id"])

    qlabels = defaultdict(dict)
    for split in ["train", "test"]:
        pattern = f"plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/matrices/{split}/{split}_features_shard_0*.jsonl"
        for shard in sorted(glob.glob(pattern)):
            with open(shard, encoding="utf-8") as f:
                for l in f:
                    if not l.strip(): continue
                    r = json.loads(l)
                    if r["query_id"] in manifest: qlabels[r["query_id"]][r["candidate"]] = int(r["label"])

    cand_fps, cand_props = {}, {}
    for labels in qlabels.values():
        for c in labels:
            if c in cand_fps: continue
            fp = mfp(c); props = mprops(c)
            if fp is not None and props is not None: cand_fps[c] = fp; cand_props[c] = props
    cand_list = sorted(cand_fps.keys())

    labeled_ofs = sorted({manifest[qid]["old_fragment_smiles"] for qid, labels in qlabels.items()
        if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())})
    log(f"  {len(labeled_ofs)} OFs, {len(cand_list)} candidates")

    # Precompute features
    of_data = {}
    for of_s in labeled_ofs:
        ofp = of_fps[of_s]; op = of_props[of_s]; ofp_n = ofp / (ofp.sum() + 1e-10)
        Xl, yl, qinfo = [], [], []
        for qid in of_queries.get(of_s, []):
            labels = qlabels.get(qid, {})
            if not labels: continue
            cands = list(labels.keys()); n_c = len(cands)
            feats = np.zeros((n_c, 7), dtype=np.float32)
            cil = []
            for idx, c in enumerate(cands):
                cfp = cand_fps.get(c); cp = cand_props.get(c)
                if cfp is None or cp is None: continue
                inter = float(cfp.dot(ofp)); denom = float(cfp.sum() + ofp.sum() - inter)
                morgan_v = inter / max(denom, 0.001)
                cfp_n = cfp / (cfp.sum() + 1e-10); bit_c = 0.0
                if cfp.sum() > 0 and ofp.sum() > 0:
                    corr = float(np.corrcoef(cfp_n, ofp_n)[0, 1])
                    bit_c = corr if np.isfinite(corr) else 0.0
                feats[idx] = [morgan_v, bit_c, abs(cp[0]-op[0]), abs(cp[1]-op[1]),
                    abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1), abs(cp[4]-op[4])/max(op[4]+1,1)]
                cil.append(cand_list.index(c) if c in cand_list else -1)
            Xl.append(feats); yl.append(np.array([labels[c] for c in cands], dtype=np.int8)); qinfo.append((qid, n_c, cil))
        if Xl: of_data[of_s] = {"X7": np.vstack(Xl), "y": np.concatenate(yl), "queries": qinfo}
    log(f"  {sum(len(v['X7']) for v in of_data.values()):,} pairs")
    return labeled_ofs, of_data, cand_list

def hit10(ld, s, c):
    o = np.argsort(-s); r = np.array([ld[c[i]] for i in o])
    p = np.where(r == 1)[0]
    return int(len(p) > 0 and p[0] < 10)

# ── Experiment ──
def run():
    from sklearn.linear_model import LogisticRegression
    labeled_ofs, of_data, cand_list = load_data()
    results = []

    for seed_idx in range(3):
        rng = np.random.RandomState(SEED + seed_idx)
        sh = list(labeled_ofs); rng.shuffle(sh)
        n_tr = int(len(sh) * 0.7)
        tr_ofs, te_ofs = sh[:n_tr], sh[n_tr:]

        freq = {}
        for o in tr_ofs:
            y = of_data[o]["y"]; queries = of_data[o]["queries"]
            off = 0
            for _, nc, cil in queries:
                for j in range(nc):
                    if y[off+j] == 1:
                        ci = cil[j]
                        if ci >= 0: freq[cand_list[ci]] = freq.get(cand_list[ci], 0) + 1
                off += nc
        mf = max(freq.values()) if freq else 1

        X7_tr = np.vstack([of_data[o]["X7"] for o in tr_ofs])
        y_tr = np.concatenate([of_data[o]["y"] for o in tr_ofs])
        X_tr = np.zeros((len(X7_tr), 8), dtype=np.float32); X_tr[:, :7] = X7_tr
        off = 0
        for o in tr_ofs:
            for _, nc, cil in of_data[o]["queries"]:
                for j in range(nc):
                    ci = cil[j]; c = cand_list[ci] if ci >= 0 else ""
                    X_tr[off+j, 7] = freq.get(c, 0) / max(mf, 1)
                off += nc
        sc = StandardScaler(); X_s = sc.fit_transform(X_tr)

        # LR baseline
        lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        lr.fit(X_s, y_tr)

        # FM
        log(f"  Seed {seed_idx}: training FM...")
        fm = train_fm(X_s, y_tr, k=8, lr=0.01, l2=0.001, epochs=30, verbose=False)

        # Evaluate
        lr_h10, fm_h10 = [], []
        for o in te_ofs:
            queries = of_data[o]["queries"]; yt = of_data[o]["y"]; X7 = of_data[o]["X7"]
            off = 0
            for _, nc, cil in queries:
                cands = [cand_list[ci] if ci >= 0 else "" for ci in cil]
                ld = dict(zip(cands, yt[off:off+nc].astype(int)))
                if sum(ld.values()) == 0: off += nc; continue
                Xq = np.zeros((nc, 8), dtype=np.float32)
                Xq[:, :7] = X7[off:off+nc]
                Xq[:, 7] = [freq.get(c, 0) / max(mf, 1) for c in cands]
                lr_s = lr.predict_proba(sc.transform(Xq))[:, 1]
                fm_s = predict_fm(fm, Xq, sc)
                lr_h10.append(hit10(ld, lr_s, cands))
                fm_h10.append(hit10(ld, fm_s, cands))
                off += nc

        lm = float(np.mean(lr_h10)); fmm = float(np.mean(fm_h10))
        results.append({"seed": seed_idx, "LR": lm, "FM": fmm, "delta": fmm - lm})
        log(f"    LR={lm:.4f}  FM={fmm:.4f}  Δ={fmm-lm:+.4f}")

    df = pd.DataFrame(results)
    print(f"\n=== FM vs LR (3 seeds, query-weighted) ===")
    print(df.to_string(index=False))
    print(f"LR mean={df['LR'].mean():.4f}  FM mean={df['FM'].mean():.4f}  Δ={df['delta'].mean():+.4f}")
    df.to_csv(OUT / "fm_vs_lr.csv", index=False)
    log("Done")

if __name__ == "__main__":
    run()
