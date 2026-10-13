"""
Step 7: second external cohort, GSE21653 (n=266, GPL570 Affymetrix).

A genuine second validation cohort: the series carries PAM50 molecular-subtype calls
(LuminalA/LuminalB/Basal/Normal/ERBB2) and DFS time/event for every sample. Two
platforms are now tested against the METABRIC-trained model -- Agilent-independent
microarray here, RNA-seq in TCGA -- plus a third genuinely different expression
technology (RNA-seq vs Affymetrix vs Illumina array).

Probe-to-gene mapping comes from GPL570.annot.gz; where several probes map to the
same panel gene, the probe with the largest mean absolute expression is kept (a
fixed, label-free rule).
"""
import gzip, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_step5_tcga import rank_norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

CLASS_ORDER = ["LuminalA", "LuminalB", "ERBB2", "claudin-low", "Basal", "Normal"]
GSE_TO_IDX = {"LuminalA": 0, "LuminalB": 1, "ERBB2": 2, "Basal": 4, "Normal": 5}


def load_gse():
    f = gzip.open(os.path.join(ROOT, "data/geo/GSE21653_series_matrix.txt.gz"), "rt", errors="ignore")
    chars, gsm, table, in_tab = [], None, [], False
    for l in f:
        if l.startswith("!Sample_characteristics_ch1"):
            chars.append(l.rstrip("\n").split("\t")[1:])
        elif l.startswith("!Sample_geo_accession"):
            gsm = [x.strip('"') for x in l.rstrip("\n").split("\t")[1:]]
        elif l.startswith("!series_matrix_table_begin"):
            in_tab = True
        elif l.startswith("!series_matrix_table_end"):
            in_tab = False
        elif in_tab:
            table.append(l.rstrip("\n").split("\t"))

    def field(key):
        out = {}
        for r in chars:
            for i, v in enumerate(r):
                if key in v.lower():
                    out[i] = v.split(":", 1)[-1].strip().strip('"')
        return out

    sub, ev, tm = field("molecular subtype"), field("dfs evt"), field("dfs time")
    probes = [r[0].strip('"') for r in table[1:]]
    X = np.array([[float(c) if c not in ("", "null") else np.nan for c in r[1:]]
                  for r in table[1:]], dtype=np.float64)
    ok = [i for i in range(len(gsm)) if sub.get(i) in GSE_TO_IDX]
    y = np.array([GSE_TO_IDX[sub[i]] for i in ok])
    dur = np.array([float(tm[i]) if tm.get(i, "NA") not in ("NA", "") else np.nan for i in ok])
    evv = np.array([1 if ev.get(i, "0") in ("1", "1.0") else 0 for i in ok])
    return probes, X[:, ok], y, dur, evv, [gsm[i] for i in ok]


def map_probes(probes, X):
    """probe -> gene symbol, keeping the highest-mean probe per gene."""
    raw = gzip.open(os.path.join(ROOT, "data/geo/GPL570.annot.gz"), "rt", errors="ignore")
    sym_col = None; mapping = {}
    for l in raw:
        if l.startswith("!platform_table_begin"):
            break
    header = raw.readline().rstrip("\n").split("\t")
    try:
        sym_col = header.index("Gene symbol")
    except ValueError:
        for i, h in enumerate(header):
            if "gene symbol" in h.lower():
                sym_col = i; break
    assert sym_col is not None, f"no Gene symbol column in {header[:12]}"
    for l in raw:
        if l.startswith("!platform_table_end"):
            break
        p = l.rstrip("\n").split("\t")
        if len(p) <= sym_col:
            continue
        g = p[sym_col].split("///")[0].strip()
        if g and g != "---":
            mapping[p[0].strip('"')] = g
    idx = {p: i for i, p in enumerate(probes)}
    best = {}
    for pr, g in mapping.items():
        i = idx.get(pr)
        if i is None:
            continue
        score = float(np.nanmean(np.abs(X[i])))
        if g not in best or score > best[g][1]:
            best[g] = (i, score)
    return {g: i for g, (i, _) in best.items()}


if __name__ == "__main__":
    probes, X, y, dur, evv, gsm = load_gse()
    print(f"GSE21653: {X.shape[0]} probes x {X.shape[1]} samples; labels {np.bincount(y, minlength=6).tolist()}")

    # METABRIC reference features, same 412 genes
    mods, ytr, dur_tr, ev_tr, _ = pool_splits(["train"])
    genes = [g.strip() for g in json.load(open(os.path.join(ROOT, "data/features.json")))["mrna_genes"]] \
        if os.path.exists(os.path.join(ROOT, "data/features.json")) else None
    if genes is None:
        import glob
        cand = glob.glob(os.path.join(ROOT, "feature_names.json")) + \
               glob.glob(os.path.join(ROOT, "data/*.json"))
        for c in cand:
            j = json.load(open(c))
            if "mrna_genes" in j:
                genes = [g.strip() for g in j["mrna_genes"]]; break
    print(f"panel genes: {len(genes)}")

    pmap = map_probes(probes, X)
    found = [g for g in genes if g in pmap]
    print(f"panel genes mapped in GPL570: {len(found)}/{len(genes)} ({100*len(found)/len(genes):.1f}%)")

    gidx = [pmap[g] for g in found]
    Xg = X[gidx].T
    Xtr_full = mods[1]
    tr_cols = [genes.index(g) for g in found]
    Xtr = Xtr_full[:, tr_cols]

    Xtr_r = rank_norm(np.nan_to_num(Xtr, nan=0.0).astype(np.float64))
    Xte_r = rank_norm(np.nan_to_num(Xg, nan=0.0).astype(np.float64))

    sc = StandardScaler().fit(Xtr_r)
    lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(sc.transform(Xtr_r), ytr)
    pred = lr.predict(sc.transform(Xte_r))

    acc = float(accuracy_score(y, pred)); f1 = float(f1_score(y, pred, average="macro"))
    present = sorted(set(y.tolist()))
    acc5 = float(accuracy_score(y, pred, normalize=True))
    print(f"\nGSE21653 external: acc={acc*100:.2f}%  macro-F1={f1:.4f}  (n={len(y)})")
    cm = confusion_matrix(y, pred, labels=list(range(6)))
    per = {}
    for c in present:
        m = y == c
        r = float((pred[m] == c).mean())
        pr = float((pred == c).sum())
        prec = float(((pred == c) & (y == c)).sum() / pr) if pr else 0.0
        per[CLASS_ORDER[c]] = {"n": int(m.sum()), "recall": r, "precision": prec}

    # survival: Cox fit on METABRIC train predictions, applied to GSE21653
    P_tr = lr.predict_proba(sc.transform(Xtr_r))
    ok = ~np.isnan(dur)
    cols = [f"p{c}" for c in lr.classes_[1:]]
    dftr = pd.DataFrame(P_tr[:, 1:], columns=cols); dftr["T"] = dur_tr; dftr["E"] = ev_tr
    cph = CoxPHFitter(penalizer=0.1).fit(dftr[~dftr["T"].isna()], "T", "E")
    risk = cph.predict_partial_hazard(pd.DataFrame(lr.predict_proba(sc.transform(Xte_r))[:, 1:],
                                                    columns=cols)).values
    ci = float(concordance_index(dur[ok], -risk[ok], evv[ok]))
    hi = risk[ok] > np.median(risk[ok])
    lrp = float(logrank_test(dur[ok][hi], dur[ok][~hi], evv[ok][hi], evv[ok][~hi]).p_value)
    print(f"GSE21653 survival: n={ok.sum()} events={evv[ok].sum()}  C-index={ci:.4f}  log-rank p={lrp:.4f}")

    out = {"cohort": "GSE21653", "platform": "GPL570 (Affymetrix U133 Plus 2.0)",
           "n": int(len(y)), "n_genes_mapped": len(found), "n_panel_genes": len(genes),
           "accuracy": acc, "macro_f1": f1,
           "per_class": per, "confusion_matrix": cm.tolist(),
           "class_names": CLASS_ORDER,
           "survival": {"n": int(ok.sum()), "events": int(evv[ok].sum()),
                        "c_index": ci, "logrank_p": lrp,
                        "risk_model": "CoxPHFitter(penalizer=0.1) fit on METABRIC train, applied to GSE21653"}}
    with open(os.path.join(RESULTS, "gse21653_validation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/route_a/gse21653_validation.json")
