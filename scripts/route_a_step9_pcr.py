"""
Step 9: treatment-response (pCR) prediction in a neoadjuvant cohort.

Cohort: GSE20194 (GPL96, HG-U133A). pcr_vs_rd annotates 278 samples, of which the label filter
retains 56 pCR and 222 RD; all 278 are usable (dropped=0).
Question: does the 412-gene genomic panel add to standard clinical variables for
predicting pathological complete response? Three nested models, 5-fold CV AUC.

Also reports the METABRIC-trained PAM50 model's pCR association in this cohort.
"""
import gzip, json, os, re, glob, warnings, collections
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
os.chdir(ROOT)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

SEED = 42

def panel():
    for c in glob.glob("feature_names.json") + glob.glob("data/*.json"):
        j = json.load(open(c))
        if "mrna_genes" in j:
            return [g.strip() for g in j["mrna_genes"]]
    raise RuntimeError

def parse_series(path):
    """Return (samples, expr_probes x samples DataFrame, clinical dict of lists)."""
    with gzip.open(path, "rt", errors="ignore") as fh:
        t = fh.read()
    head, body = t.split("series_matrix_table_begin", 1)
    body = body.split("series_matrix_table_end")[0]
    gsm = None
    for l in head.split("\n"):
        if l.startswith("!Sample_geo_accession"):
            gsm = [v.strip('"') for v in l.split("\t")[1:] if v.strip()]
    clin = collections.defaultdict(lambda: [None] * len(gsm))
    for l in head.split("\n"):
        if not l.startswith("!Sample_characteristics_ch1"):
            continue
        cells = l.rstrip("\n").split("\t")[1:]
        for i, cell in enumerate(cells):
            if i >= len(gsm):
                break
            cell = cell.strip().strip('"')
            if ":" in cell:
                k, v = cell.split(":", 1)
                clin[k.strip()][i] = v.strip()
    rows = [r for r in body.split("\n") if r.strip() and not r.startswith("!")]
    if rows and "ID_REF" in rows[0]:
        rows = rows[1:]
    probes = [r.split("\t")[0].strip('"') for r in rows]
    M = np.array([[np.nan if x.strip().strip('"') in ("", "NA") else float(x)
                   for x in r.split("\t")[1:]] for r in rows], dtype=float)
    return gsm, probes, M, dict(clin)

def load_platform_symbols(annot_path):
    sym = {}
    with gzip.open(annot_path, "rt", errors="ignore") as fh:
        hdr = None
        for l in fh:
            if l.startswith("!platform_table_begin"):
                hdr = fh.readline().rstrip("\n").split("\t"); break
        ci = next(k for k, h in enumerate(hdr) if "gene symbol" in h.lower())
        for l in fh:
            if l.startswith("!platform_table_end"):
                break
            p = l.rstrip("\n").split("\t")
            if len(p) > ci:
                g = p[ci].split("///")[0].strip()
                if g and g != "---":
                    sym[p[0].strip('"')] = g
    return sym

def to_gene_matrix(probes, M, sym, genes):
    """probe x sample -> gene x sample, best probe (highest mean abs) per gene."""
    idx = {p: i for i, p in enumerate(probes)}
    best = {}
    for pr, g in sym.items():
        i = idx.get(pr)
        if i is None or g not in genes:
            continue
        s = float(np.nanmean(np.abs(M[i])))
        if g not in best or s > best[g][1]:
            best[g] = (i, s)
    found = [g for g in genes if g in best]
    X = np.vstack([M[best[g][0]] for g in found]).T
    return X, found

def rank_norm(X):
    from scipy.stats import rankdata, norm
    out = np.empty_like(X, float); n = X.shape[0]
    for j in range(X.shape[1]):
        r = rankdata(X[:, j]) / (n + 1.0)
        out[:, j] = norm.ppf(np.clip(r, 1e-3, 1 - 1e-3))
    return out

def cv_auc(X, y, seeds=(0, 1, 2, 3, 4)):
    aucs = []
    for s in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=s)
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            pipe = make_pipeline(StandardScaler(),
                                 LogisticRegression(C=0.01, max_iter=5000, random_state=SEED))
            pipe.fit(X[tr], y[tr])
            oof[te] = pipe.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.std(aucs))

if __name__ == "__main__":
    genes = panel()
    gsm, probes, M, clin = parse_series("data/geo/GSE20194_series_matrix.txt.gz")
    print(f"GSE20194: {len(gsm)} samples, {len(probes)} probes")
    lab = np.array([str(v).strip() for v in clin["pcr_vs_rd"]])
    keep = np.isin(lab, ["pCR", "RD"])
    y = (lab[keep] == "pCR").astype(int)
    print(f"  usable: n={keep.sum()}  pCR={int(y.sum())}  RD={int((y==0).sum())}  dropped={int((~keep).sum())}")

    sym = load_platform_symbols("data/geo/GPL96.annot.gz")
    X, found = to_gene_matrix(probes, M, sym, genes)
    X = X[keep]
    print(f"  panel genes on GPL96: {len(found)}/{len(genes)}")

    def num(key):
        v = np.array([str(clin[key][i]).strip() for i in range(len(gsm))])
        return v[keep]
    age = pd.to_numeric(pd.Series(num("age")), errors="coerce").values
    def binv(key, pos):
        v = num(key)
        return np.array([1.0 if x.upper() in pos else (0.0 if x.upper() in ("N", "NEG", "0") else np.nan) for x in v])
    er = binv("er_status", ("P",))
    pr = binv("pr_status", ("P",))
    her2 = binv("her2 status", ("P",))
    grd = pd.to_numeric(pd.Series(num("bmngrd")), errors="coerce").values
    hist = num("histology")
    hist_idc = np.array([1.0 if "IDC" in str(x).upper() else 0.0 for x in hist])

    C = np.column_stack([age, er, pr, her2, grd, hist_idc]).astype(float)
    med = np.nanmedian(C, axis=0)
    C = np.where(np.isnan(C), med[None, :], C)
    print(f"  clinical matrix: {C.shape}  missing before impute: {int(np.isnan(C).sum())}")

    Xr = rank_norm(np.nan_to_num(X, nan=0.0))
    Xcg = np.column_stack([C, Xr])

    res = {"cohort": "GSE20194", "platform": "GPL96 (HG-U133A)", "n": int(len(y)),
           "n_pcr": int(y.sum()), "n_rd": int((y == 0).sum()),
           "endpoint": "pcr_vs_rd (pathological complete response after neoadjuvant chemotherapy)",
           "n_panel_genes": len(found), "n_panel_total": len(genes),
           "clinical_features": ["age", "ER", "PR", "HER2", "grade(BMNG)", "histology_IDC"]}
    for name, Xm in [("clinical_only", C), ("genomic_only", Xr), ("clinical_plus_genomic", Xcg)]:
        m, s = cv_auc(Xm, y)
        res[name] = {"auc": m, "auc_std": s}
        print(f"  {name:22s} AUC={m:.4f} +/- {s:.4f}")

    # does the panel add over clinical? paired bootstrap on OOF predictions
    def oof(Xm, seed=0):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        o = np.zeros(len(y))
        for tr, te in skf.split(Xm, y):
            p = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000, random_state=SEED))
            p.fit(Xm[tr], y[tr]); o[te] = p.predict_proba(Xm[te])[:, 1]
        return o
    oc, og = oof(C), oof(Xcg)
    rng = np.random.RandomState(SEED); d = []
    for _ in range(2000):
        b = rng.randint(0, len(y), len(y))
        if len(np.unique(y[b])) < 2: continue
        d.append(roc_auc_score(y[b], og[b]) - roc_auc_score(y[b], oc[b]))
    d = np.array(d)
    res["delta_genomic_over_clinical"] = {
        "mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_gt0": float((d > 0).mean())}
    print(f"  Delta(clinical+genomic - clinical) = {d.mean():+.4f}  "
          f"95% CI [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]  P(>0)={ (d>0).mean():.4f}")

    # METABRIC-trained PAM50 model: association of predicted subtype with pCR
    import sys; sys.path.insert(0, "scripts")
    from route_a_harness import pool_splits
    mods, ytr, _, _, _ = pool_splits(["train"])
    tr_cols = [genes.index(g) for g in found]
    Xtr = rank_norm(np.nan_to_num(mods[1][:, tr_cols], nan=0.0))
    pam = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(StandardScaler().fit_transform(Xtr), ytr)
    P = pam.predict_proba(StandardScaler().fit(Xtr).transform(Xr))
    sub = pam.classes_[P.argmax(1)]
    names = ["LumA", "LumB", "Her2", "claudin-low", "Basal", "Normal"]
    tab = {names[c]: {"n": int((sub == c).sum()), "pcr": int(y[sub == c].sum()),
                      "rate": float(y[sub == c].mean()) if (sub == c).sum() else None} for c in np.unique(sub)}
    res["predicted_subtype_pcr"] = tab
    print("\n  predicted PAM50 subtype vs pCR rate:")
    for k, v in tab.items():
        print(f"    {k:12s} n={v['n']:3d}  pCR={v['pcr']:3d}  rate={v['rate'] if v['rate'] is None else round(v['rate'],3)}")
    a_sub = roc_auc_score(y, P[:, list(pam.classes_).index(4)]) if 4 in pam.classes_ else float("nan")
    res["auc_basal_probability"] = float(a_sub)
    print(f"    AUC(basal-like probability -> pCR) = {a_sub:.4f}")

    # --- does the predicted subtype add over clinical alone, and does ER alone suffice? ---
    # ER status is a member of the clinical block (column 1), so there is deliberately no
    # separate "clinical+ER+subtype" condition: it would be the same input space as
    # "clinical+subtype" and would test nothing.
    er = C[:, 1].reshape(-1, 1)
    a_er, _ = cv_auc(er, y)
    a_sub, _ = cv_auc(P, y)
    Xcs = np.column_stack([C, P])
    a_cs, _ = cv_auc(Xcs, y)
    ocs = oof(Xcs, seed=0)   # same seed as oc, so the paired bootstrap compares like with like
    d2 = []
    for _ in range(2000):
        b = rng.randint(0, len(y), len(y))
        if len(np.unique(y[b])) < 2:
            continue
        d2.append(roc_auc_score(y[b], ocs[b]) - roc_auc_score(y[b], oc[b]))
    d2 = np.array(d2)
    res["subtype_value_test"] = {
        "ER_status_alone": float(a_er),
        "clinical(6)": float(res["clinical_only"]["auc"]),
        "subtype_probs(6)": float(a_sub),
        "clinical+subtype": float(a_cs),
        "delta_clinical_plus_subtype": {
            "mean": float(d2.mean()),
            "ci95": [float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))],
            "p_gt0": float((d2 > 0).mean())}}
    res["notes_subtype_value_test"] = (
        "ER status is already a member of the clinical block (age, ER, PR, HER2, grade, histology), so a "
        "separate 'clinical+ER+subtype' condition is not a distinct input space and tests nothing beyond "
        "'clinical+subtype'; it is deliberately not reported as an independent comparison.")
    print(f"  ER alone AUC={a_er:.4f} | subtype probs={a_sub:.4f} | clinical={res['clinical_only']['auc']:.4f} "
          f"| clinical+subtype={a_cs:.4f}")
    print(f"  Delta(clinical+subtype - clinical) = {d2.mean():+.4f}  P(>0)={(d2>0).mean():.4f}")

    json.dump(res, open("results/route_a/pcr_gse20194.json", "w"), indent=2)
    print("\nwrote results/route_a/pcr_gse20194.json")
