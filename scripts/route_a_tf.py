"""
LightTransformer fusion models for Route A, with and without LogReg guidance.

Architecture (both variants):
    each modality -> linear projection to d_model
    [CLS] + one token per modality (+ optional guidance token)
        -> TransformerEncoder (n_layers, nhead)
    [CLS] representation -> linear -> 6-class logits

The guided variant appends a token computed from a frozen logistic-regression
class-probability vector, so the Transformer starts from the linear model's
posterior rather than from scratch.
"""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
SEED = 42


class LightTransformer(nn.Module):
    def __init__(self, in_dims, n_classes=6, d_model=64, n_layers=2, nhead=4,
                 dim_ff=128, dropout=0.3, guided=False):
        super().__init__()
        self.n_mod = len(in_dims)
        self.guided = guided
        self.proj = nn.ModuleList([nn.Linear(d, d_model) for d in in_dims])
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        # one learnable token per modality
        self.mod_tok = nn.Parameter(torch.zeros(1, self.n_mod, d_model))
        n_tok = self.n_mod + 1 + (1 if guided else 0)
        self.pos = nn.Parameter(torch.zeros(1, n_tok, d_model))
        if guided:
            self.guide_proj = nn.Linear(n_classes, d_model)
        self.norm_in = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.n_mod)])
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.mod_tok, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, mods, guide=None, modality_dropout=0.0):
        # mods: list of (B, d_i) tensors
        B = mods[0].size(0)
        toks = []
        for i, x in enumerate(mods):
            toks.append(self.norm_in[i](self.proj[i](x)).unsqueeze(1) + self.mod_tok[:, i:i+1, :])
        toks = torch.cat(toks, dim=1)
        # modality-level dropout: zero whole modality tokens during training
        if self.training and modality_dropout > 0:
            keep = (torch.rand(B, self.n_mod, 1, device=toks.device) > modality_dropout).float()
            toks = toks * keep
        seq = [self.cls.expand(B, -1, -1), toks]
        if self.guided:
            g = self.guide_proj(guide).unsqueeze(1)
            seq.append(g)
        h = torch.cat(seq, dim=1) + self.pos
        h = self.encoder(h)
        return self.head(self.head_drop(self.norm(h[:, 0])))


def to_tensors(mods, device):
    return [torch.tensor(m, dtype=torch.float32, device=device) for m in mods]


def train_one(model, train_mods, train_y, guide_tr, val_mods=None, val_y=None,
              guide_va=None, epochs=80, lr=1e-3, wd=0.01, batch=64,
              modality_dropout=0.0, label_smoothing=0.0, class_weight=None,
              seed=SEED, verbose=False):
    """Train in place; return per-epoch history for diagnosis plots."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    model = model.to(device)
    n = len(train_y)
    yt = torch.tensor(train_y, dtype=torch.long, device=device)
    gt = torch.tensor(guide_tr, dtype=torch.float32, device=device) if guide_tr is not None else None
    vt = None
    if val_mods is not None:
        vt = to_tensors(val_mods, device)
        vg = torch.tensor(guide_va, dtype=torch.float32, device=device) if guide_va is not None else None
    tm = to_tensors(train_mods, device)

    pw = None
    if class_weight is not None:
        pw = torch.tensor(class_weight, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=pw, label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tl, tc, tn = 0.0, 0, 0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            bm = [m[idx] for m in tm]
            bg = gt[idx] if gt is not None else None
            by = yt[idx]
            opt.zero_grad()
            out = model(bm, guide=bg, modality_dropout=modality_dropout)
            loss = crit(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(idx)
            tc += (out.argmax(1) == by).sum().item()
            tn += len(idx)
        sched.step()
        hist["train_loss"].append(tl / tn)
        hist["train_acc"].append(tc / tn)
        if val_mods is not None:
            model.eval()
            with torch.no_grad():
                vo = model(vt, guide=vg)
                vl = nn.functional.cross_entropy(vo, torch.tensor(val_y, dtype=torch.long, device=device)).item()
                va = (vo.argmax(1).cpu().numpy() == np.asarray(val_y)).mean()
            hist["val_loss"].append(vl)
            hist["val_acc"].append(float(va))
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    ep{ep:3d} train_loss={hist['train_loss'][-1]:.4f} "
                  f"train_acc={hist['train_acc'][-1]:.4f} "
                  f"val_loss={hist['val_loss'][-1] if hist['val_loss'] else float('nan'):.4f} "
                  f"val_acc={hist['val_acc'][-1] if hist['val_acc'] else float('nan'):.4f}")
    return hist


def predict(model, mods, guide=None, batch=256):
    device = torch.device("cpu")
    model.eval()
    tm = to_tensors(mods, device)
    gt = torch.tensor(guide, dtype=torch.float32, device=device) if guide is not None else None
    outs = []
    with torch.no_grad():
        for i in range(0, len(mods[0]), batch):
            bm = [m[i:i+batch] for m in tm]
            bg = gt[i:i+batch] if gt is not None else None
            outs.append(torch.softmax(model(bm, guide=bg), dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0)
