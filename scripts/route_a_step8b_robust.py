"""Step 8b: is the correction label-free and stable? (a) mask from the annotation
standard only, never from y; (b) split-half transductive stability; (c) resampling."""
import json, os, sys, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_step5_tcga import rank_norm
from route_a_step7_gse21653 import CLASS_ORDER
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

PAM50_MASK = np.array([True, True, True, False, True, True])  # claudin-low NOT in PAM50

def panel_genes():
    for c in glob.glob(os.path.join(ROOT,"feature_names.json"))+glob.glob(os.path.join(ROOT,"data/*.json")):
        j=json.load(open(c))
        if "mrna_genes" in j: return [g.strip() for g in j["mrna_genes"]]

def mlls(P, prior_src, allowed, n_iter=1000, tol=1e-9):
    w=np.ones(P.shape[1]); w[~allowed]=0.0; pt=None
    for _ in range(n_iter):
        Q=P*w; s=Q.sum(1,keepdims=True); s[s==0]=1.0; Q/=s
        pt=Q.mean(0)
        wn=np.where(prior_src>0, pt/np.maximum(prior_src,1e-12), 0.0); wn[~allowed]=0.0
        if np.max(np.abs(wn-w))<tol: w=wn; break
        w=wn
    Q=P*w; s=Q.sum(1,keepdims=True); s[s==0]=1.0; return (Q/s).argmax(1), pt

genes=panel_genes()
d=np.load(os.path.join(ROOT,"data/tcga_brca/tcga_validation.npz"),allow_pickle=True)
mods,ytr,_,_,_=pool_splits(["train"])
Xs=rank_norm(np.nan_to_num(mods[1],nan=0.0).astype(np.float64))
sc=StandardScaler().fit(Xs)
lr=LogisticRegression(C=0.01,max_iter=5000,random_state=SEED).fit(sc.transform(Xs),ytr)
prior_src=np.bincount(ytr,minlength=6).astype(float)/len(ytr)
P=lr.predict_proba(sc.transform(rank_norm(np.nan_to_num(d["mrna"],nan=0.0).astype(np.float64))))
y=d["labels"]

base=P.argmax(1); corr,_=mlls(P.copy(),prior_src,PAM50_MASK)
print(f"a priori mask (never sees y): acc {accuracy_score(y,base)*100:.2f}% -> {accuracy_score(y,corr)*100:.2f}%  "
      f"macroF1 {f1_score(y,base,average='macro'):.4f} -> {f1_score(y,corr,average='macro'):.4f}")

# split-half transductive stability: estimate prior on half A, apply to half B and vice versa
rng=np.random.RandomState(SEED); idx=rng.permutation(len(y)); A,B=idx[:len(y)//2],idx[len(y)//2:]
accs=[]
for src,tgt in ((A,B),(B,A)):
    _,pt=mlls(P[src].copy(),prior_src,PAM50_MASK)
    Q=P[tgt]*(pt/np.maximum(prior_src,1e-12)); Q/=Q.sum(1,keepdims=True)
    accs.append(accuracy_score(y[tgt],Q.argmax(1)))
print(f"split-half transfer: base {accuracy_score(y[B],base[B])*100:.2f}%/{accuracy_score(y[A],base[A])*100:.2f}%"
      f" -> corrected {accs[0]*100:.2f}%/{accs[1]*100:.2f}%")

# bootstrap CI on the gain
gains=[]; n=len(y)
for _ in range(2000):
    b=rng.randint(0,n,n)
    gains.append(accuracy_score(y[b],corr[b])-accuracy_score(y[b],base[b]))
gains=np.array(gains)*100
print(f"bootstrap gain: {gains.mean():+.2f} pts  95% CI [{np.percentile(gains,2.5):+.2f}, {np.percentile(gains,97.5):+.2f}]  P(gain>0)={np.mean(gains>0):.4f}")

json.dump({"a_priori_mask_acc":float(accuracy_score(y,corr)),"baseline_acc":float(accuracy_score(y,base)),
           "a_priori_mask_macroF1":float(f1_score(y,corr,average='macro')),
           "baseline_macroF1":float(f1_score(y,base,average='macro')),
           "split_half_acc":accs,"bootstrap_gain_mean":float(gains.mean()),
           "bootstrap_ci":[float(np.percentile(gains,2.5)),float(np.percentile(gains,97.5))],
           "p_gain_positive":float(np.mean(gains>0)),"n":int(n)},
          open(os.path.join(RESULTS,"crosscohort_shift_robust.json"),"w"),indent=2)
print("wrote crosscohort_shift_robust.json")
