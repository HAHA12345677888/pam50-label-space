"""
Step 4: self-supervised pretraining of the modality encoders.

Protocol (honest-CV rule): for each of the 5 canonical folds, the encoder is
pretrained on THAT FOLD's training portion only, then the fusion head is
fine-tuned on the same portion and scored on the held-out fold. Pretraining on
the full dataset before cross-validation would leak and inflate CV.

Objective: masked-feature reconstruction. A random subset of input features in
each modality is zeroed; the encoder's modality token must reconstruct them
through a linear decoder. This is the cheapest objective that forces the
projection to encode cross-feature structure rather than memorise labels.
"""
import json, os, sys, time, warnings
import numpy as np, torch, torch.nn as nn
warnings.filterwarnings("ignore")
torch.set_num_threads(4)

ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from route_a_harness import pool_splits, stack, folds, BEST_MODS, RESULTS, SEED
from route_a_tf import LightTransformer, train_one, predict
from sklearn.metrics import accuracy_score, f1_score

EPOCHS = 60
PRE_EPOCHS = 40
MASK_FRAC = 0.3
REFERENCE = 0.7526


class MaskedAE(nn.Module):
    """Wraps LightTransformer's projection+encoder with per-modality decoders."""

    def __init__(self, in_dims, d_model=64, n_layers=2, nhead=4, dropout=0.3):
        super().__init__()
        self.in_dims = in_dims
        self.n_mod = len(in_dims)
        self.proj = nn.ModuleList([nn.Linear(d, d_model) for d in in_dims])
        self.norm_in = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.n_mod)])
        self.mod_tok = nn.Parameter(torch.zeros(1, self.n_mod, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.dec = nn.ModuleList([nn.Linear(d_model, d) for d in in_dims])

    def forward(self, mods):
        toks = []
        for i, x in enumerate(mods):
            toks.append(self.norm_in[i](self.proj[i](x)).unsqueeze(1) + self.mod_tok[:, i:i+1, :])
        h = self.encoder(torch.cat(toks, dim=1))
        return [self.dec[i](h[:, i, :]) for i in range(self.n_mod)]

    def transfer_to(self, model):
        """Copy pretrained encoder weights into a classification model."""
        with torch.no_grad():
            for i in range(self.n_mod):
                model.proj[i].weight.copy_(self.proj[i].weight)
                model.proj[i].bias.copy_(self.proj[i].bias)
                model.norm_in[i].weight.copy_(self.norm_in[i].weight)
                model.norm_in[i].bias.copy_(self.norm_in[i].bias)
            model.mod_tok.copy_(self.mod_tok)
            model.encoder.load_state_dict(self.encoder.state_dict())
        return model


def pretrain(tm, in_dims, seed=SEED):
    torch.manual_seed(seed)
    m = MaskedAE(in_dims)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    n = tm[0].size(0)
    for ep in range(PRE_EPOCHS):
        m.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 64):
            idx = perm[i:i+64]
            bm = [x[idx] for x in tm]
            masked, targets, masks = [], [], []
            for x in bm:
                mask = (torch.rand_like(x) < MASK_FRAC).float()
                masked.append(x * (1 - mask))
                targets.append(x)
                masks.append(mask)
            opt.zero_grad()
            rec = m(masked)
            loss = sum((((r - t) ** 2) * mk).sum() / mk.sum().clamp(min=1)
                       for r, t, mk in zip(rec, targets, masks)) / len(rec)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += loss.item()
    return m


def cv_pretrained(mode="pretrained"):
    mods, y, _, _, _ = pool_splits(["train"])
    Xs = [mods[i] for i in BEST_MODS]
    in_dims = [x.shape[1] for x in Xs]
    accs, f1s = [], []
    for k, (tr, te) in enumerate(folds().split(stack(mods, BEST_MODS), y)):
        tm_t = [torch.tensor(x[tr], dtype=torch.float32) for x in Xs]
        tm_v = [x[te] for x in Xs]
        model = LightTransformer(in_dims, d_model=64, n_layers=2, nhead=4,
                                 dim_ff=128, dropout=0.3)
        if mode == "pretrained":
            ae = pretrain(tm_t, in_dims)
            model = ae.transfer_to(model)
        h = train_one(model, tm_v if False else [x.numpy() for x in tm_t], y[tr], None,
                      val_mods=tm_v, val_y=y[te], epochs=EPOCHS, seed=SEED)
        p = predict(model, tm_v)
        accs.append(accuracy_score(y[te], p.argmax(1)))
        f1s.append(f1_score(y[te], p.argmax(1), average="macro"))
        print(f"  fold{k} [{mode}]: acc={accs[-1]:.4f}")
    return {"cv": float(np.mean(accs)), "std": float(np.std(accs)),
            "macro_f1": float(np.mean(f1s)), "per_fold": [float(a) for a in accs]}


if __name__ == "__main__":
    t0 = time.time()
    out = {"reference_logreg_cv": REFERENCE, "pre_epochs": PRE_EPOCHS, "mask_frac": MASK_FRAC}
    for mode in ["scratch", "pretrained"]:
        r = cv_pretrained(mode)
        out[mode] = r
        print(f"{mode}: CV={r['cv']*100:.2f}% +/- {r['std']*100:.2f}%  F1={r['macro_f1']:.3f}  "
              f"({time.time()-t0:.0f}s)")
        with open(os.path.join(RESULTS, "pretrain_cv.json"), "w") as f:
            json.dump(out, f, indent=2)
    out["delta"] = out["pretrained"]["cv"] - out["scratch"]["cv"]
    out["_elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "pretrain_cv.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\npretraining delta: {out['delta']*100:+.2f} points  ({out['_elapsed_s']}s)")
