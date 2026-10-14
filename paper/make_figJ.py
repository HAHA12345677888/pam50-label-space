"""Figure J -- modality ablation in the harmonized METABRIC/TCGA space.

Reads results/route_a/multimodal_external.json (produced by
scripts/route_a_step16_multimodal_external.py) and renders the two-panel
internal-vs-external comparison.

Encoding notes, so the panels are read correctly:
  * Panel A draws two bars per configuration -- solid = 5-fold CV on pooled
    METABRIC train+val, hatched = held-out METABRIC test. The legend states this;
    the hatching is not decoration.
  * The dashed line in panel A is the ORIGINAL 713-dimension flagship's internal
    CV accuracy, which is not one of the plotted bars. It is drawn in neutral
    black, not in a data-series hue, so it cannot be mistaken for a series.
  * The methylation series uses orange specifically to avoid collision with any
    semantic red.
  * Panel B error bars are bootstrap 95% CIs over the 694 TCGA patients.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "route_a")
FIGDIR = os.path.join(ROOT, "paper", "figures")

# apply_figure_style() is supplied by the figure-style skill in interactive use;
# fall back to matplotlib defaults if the helper is unavailable.
try:
    apply_figure_style()
except NameError:
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "savefig.bbox": "tight"})

ORDER = ["mrna", "mrna+clinical_h", "mrna+methylation",
         "clinical_h+mrna+methylation"]
LABELS = ["mRNA", "mRNA\n+clinical", "mRNA\n+methylation",
          "clinical+mRNA\n+methylation"]
COLS = ["#3b6ea5", "#7aa6c2", "#d1793a", "#8c8c8c"]


def main():
    d = json.load(open(os.path.join(RESULTS, "multimodal_external.json")))
    cfg, flags = d["configs"], d["reference_original_flagship_69dim"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300)
    x = np.arange(4)
    w = 0.36

    ax = axes[0]
    cv = [cfg[k]["cv_acc_mean"] * 100 for k in ORDER]
    te = [cfg[k]["metabric_test_acc"] * 100 for k in ORDER]
    ax.bar(x - w / 2, cv, w, color=COLS, edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, te, w, color=COLS, edgecolor="black", linewidth=0.5,
           alpha=0.5, hatch="///")
    for xi, v in zip(x - w / 2, cv):
        ax.text(xi, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=6)
    for xi, v in zip(x + w / 2, te):
        ax.text(xi, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=6)
    ref = flags["cv_acc_mean"] * 100
    ax.axhline(ref, ls="--", lw=1.0, color="black")
    ax.text(3.30, ref + 0.15, "713-dim flagship (internal CV)",
            ha="right", va="bottom", fontsize=6, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=6)
    ax.set_ylim(68, 80.4)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Internal: clinical helps, methylation flat", fontsize=7)
    ax.legend(handles=[
        Patch(fc="#8c8c8c", ec="black", lw=0.5, label="5-fold CV"),
        Patch(fc="#8c8c8c", ec="black", lw=0.5, alpha=0.5, hatch="///",
              label="held-out test")],
        fontsize=6, frameon=False, loc="upper left", borderaxespad=0.2)
    ax.margins(x=0.02)

    ax = axes[1]
    acc = [cfg[k]["tcga_acc_restricted"] * 100 for k in ORDER]
    ci = [cfg[k]["tcga_acc_restricted_ci"] for k in ORDER]
    err = [[a - c[0] * 100 for a, c in zip(acc, ci)],
           [c[1] * 100 - a for a, c in zip(acc, ci)]]
    ax.bar(x, acc, 0.55, color=COLS, edgecolor="black", linewidth=0.5,
           yerr=err, error_kw=dict(lw=0.9, capsize=2.5, ecolor="black"))
    for xi, a, e in zip(x, acc, err[1]):
        ax.text(xi, a + e + 0.15, f"{a:.1f}", ha="center", va="bottom",
                fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=6)
    ax.set_ylim(73.5, 85.5)
    ax.set_ylabel("Accuracy (%), 5 PAM50 classes")
    ax.set_title("External (TCGA, $n$=694, 5 classes): no gain", fontsize=7)
    ax.margins(x=0.02)

    fig.tight_layout()
    out_pdf = os.path.join(FIGDIR, "figJ_multimodal_external.pdf")
    fig.savefig(out_pdf)
    fig.savefig(os.path.join(ROOT, "figJ_check.png"))

    # render-then-verify: no text/text overlap
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_visible()]
    ov = [(a.get_text(), b.get_text())
          for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
          if ba.overlaps(bb)]
    ticks = {ax: set(ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"))
             for ax in fig.axes}
    ov += [(t.get_text(), "spine") for t, bt in texts for ax in fig.axes
           for sp in ax.spines.values()
           if sp.get_visible() and bt.overlaps(sp.get_window_extent(r))
           and t not in ticks[ax]]
    # legend patch overlapping a data bar: compare patch boxes to bar boxes
    for ax in fig.axes:
        if ax.get_legend():
            lb = ax.get_legend().get_window_extent(r)
            for pt in ax.patches:
                pb = pt.get_window_extent(r)
                if lb.overlaps(pb):
                    ov.append(("legend", "data bar"))
    assert not ov, f"overlaps: {ov}"
    print("wrote", out_pdf)


if __name__ == "__main__":
    main()
