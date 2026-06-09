#!/usr/bin/env python3
"""LambdaRank experiment: BCE vs NDCG-weighted loss. Same features, same LR model."""
import json, glob, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn, torch.optim as optim

SEED = 20260603
OUT = Path("technique_transfer/results")
RDLogger.DisableLog("rdApp.*")

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── LambdaRank loss: weight BCE by relevance gain ──
def lambdarank_weighted_bce(scores, labels, sigma=1.0):
    """NDCG-weighted BCE: higher weight for top-ranked positions.

    For each query:
      - Sort by true label first, compute ideal DCG
      - Weight each positive by 1/log2(rank+1)
      - Higher weight = this positive matters more for Hit@K
    """
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum().float()
    if n_pos == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Compute ideal DCG weights (position-based)
    ideal_ranks = torch.arange(1, int(n_pos) + 1, dtype=torch.float32)
    ideal_weights = 1.0 / torch.log2(ideal_ranks + 1)

    # For each positive, assign weight proportional to 1/log2(ideal_rank)
    # Top-ranked positives get highest weight
    # For negatives: uniform weight = 1/n_neg (diluted)
    pos_weight_val = ideal_weights.mean()  # average relevance weight

    pos_loss = nn.BCELoss(reduction='none')(scores[pos_mask], labels[pos_mask])
    pos_loss = pos_loss * pos_weight_val

    n_neg = neg_mask.sum().float()
    if n_neg > 0:
        neg_loss = nn.BCELoss(reduction='none')(scores[neg_mask], labels[neg_mask])
        # Negatives get lower weight proportional to class ratio
        neg_loss = neg_loss * (n_pos / max(n_neg, 1))
        return (pos_loss.sum() + neg_loss.sum()) / (n_pos * pos_weight_val + n_pos)
    return pos_loss.mean()

# ── LR model (same architecture, different loss) ──
class LinearRanker(nn.Module):
    def __init__(self, n_features=8):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)

def train_lr_lambdarank(X, y, query_info, lr=0.01, l2=0.001, epochs=20, batch_q=500):
    """Train LR with NDCG-weighted BCE. Process queries in batches."""
    device = torch.device("cpu")
    n, d = X.shape
    model = LinearRanker(d).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)

    # Build query offsets
    offsets = [0]
    for _, nc, _ in query_info:
        offsets.append(offsets[-1] + nc)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; n_q = 0
        # Shuffle query order
        q_order = np.random.permutation(len(query_info))
        for j in range(0, len(query_info), batch_q):
            batch_idx = q_order[j:j+batch_q]
            batch_loss = 0.0
            for qi in batch_idx:
                start, end = offsets[qi], offsets[qi+1]
                Xq = X_t[start:end]; yq = y_t[start:end]
                if yq.sum() == 0: continue
                scores = model(Xq)
                loss = lambdarank_weighted_bce(scores, yq)
                batch_loss += loss
                n_q += 1
            if batch_loss > 0:
                optimizer.zero_grad()
                (batch_loss / len(batch_idx)).backward()
                optimizer.step()
                total_loss += batch_loss.item()

    model.eval()
    return model

def predict_lr(model, X, scaler):
    X_s = torch.FloatTensor(scaler.transform(X))
    with torch.no_grad():
        return model(X_s).numpy()

# ── Data loading ──
def load_data():
    log("Loading...")
    manifest = {}
    mp = Path("plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl")
    with mp.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): q = json.loads(line); manifest[q["query_id"]] = q

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
        tr_qinfo = []
        off = 0
        for o in tr_ofs:
            for qid, nc, cil in of_data[o]["queries"]:
                for j in range(nc):
                    ci = cil[j]; c = cand_list[ci] if ci >= 0 else ""
                    X_tr[off+j, 7] = freq.get(c, 0) / max(mf, 1)
                tr_qinfo.append((qid, nc, cil))
                off += nc
        sc = StandardScaler(); X_s = sc.fit_transform(X_tr)

        # LR baseline
        lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        lr.fit(X_s, y_tr)

        # LambdaRank LR
        log(f"  Seed {seed_idx}: training LambdaRank...")
        lr_lam = train_lr_lambdarank(X_s, y_tr, tr_qinfo, lr=0.01, l2=0.001, epochs=20)

        # Evaluate
        b_h10, lam_h10 = [], []
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
                b_s = lr.predict_proba(sc.transform(Xq))[:, 1]
                lam_s = predict_lr(lr_lam, Xq, sc)
                b_h10.append(hit10(ld, b_s, cands))
                lam_h10.append(hit10(ld, lam_s, cands))
                off += nc

        bm = float(np.mean(b_h10)); lm = float(np.mean(lam_h10))
        results.append({"seed": seed_idx, "LR_BCE": bm, "LR_Lambda": lm, "delta": lm - bm})
        log(f"    BCE={bm:.4f}  Lambda={lm:.4f}  Δ={lm-bm:+.4f}")

    df = pd.DataFrame(results)
    print(f"\n=== LambdaRank vs BCE (3 seeds) ===")
    print(df.to_string(index=False))
    print(f"BCE mean={df['LR_BCE'].mean():.4f}  Lambda mean={df['LR_Lambda'].mean():.4f}  Δ={df['delta'].mean():+.4f}")
    df.to_csv(OUT / "lambda_rank_vs_bce.csv", index=False)
    log("Done")

if __name__ == "__main__":
    run()
