"""
Fetch TCGA-BRCA gene-level DNA methylation (HM450) from cBioPortal.

Why this exists
---------------
The METABRIC reference model is multimodal (clinical + mRNA + methylation),
but the published external validation was mRNA-only, because no TCGA file in
`data/tcga_brca/` carried methylation. That left the headline configuration
unvalidated externally. This script closes that gap.

Platform note
-------------
METABRIC methylation in this project is already reduced to gene level (top-232
genes by variance, gene symbols in `feature_names.json:meth_probes`). cBioPortal
serves `brca_tcga_methylation_hm450` at gene level as well, so the two are
comparable in kind rather than probe-for-probe. Cross-platform comparability is
handled downstream by within-cohort rank normalisation, not by probe matching.

Output
------
data/tcga_brca/tcga_methylation.npz  with arrays:
    methylation  (n_samples, 232) float32   gene-level beta values
    sample_ids   (n_samples,)     <U15     e.g. TCGA-AR-A1AR-01
    gene_symbols (232,)           <U12     column order; matches meth_probes

Requires network access to www.cbioportal.org.
"""
import json
import os
import socket
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TCGA = os.path.join(DATA, "tcga_brca")

PROFILE = "brca_tcga_methylation_hm450"
SAMPLE_LIST = "brca_tcga_methylation_hm450"
API = "https://www.cbioportal.org/api"
BATCH = 58          # cBioPortal times out on very large fetch payloads
TIMEOUT = 120

socket.setdefaulttimeout(TIMEOUT)


def _get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                              "User-Agent": "bc-mt/1.0"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def _post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "User-Agent": "bc-mt/1.0"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def resolve_entrez(symbols):
    """HUGO symbol -> Entrez id, via cBioPortal's gene resolver."""
    res = _post(f"{API}/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", list(symbols))
    return {g["hugoGeneSymbol"]: int(g["entrezGeneId"]) for g in res}


def main():
    fn = json.load(open(os.path.join(DATA, "feature_names.json")))
    symbols = fn["meth_probes"]
    print(f"[1/4] METABRIC methylation genes: {len(symbols)}")

    sym2ent = resolve_entrez(symbols)
    missing = [s for s in symbols if s not in sym2ent]
    if missing:
        print(f"      unresolved symbols ({len(missing)}): {missing[:10]}")
    ent2sym = {v: k for k, v in sym2ent.items()}
    ids = sorted(set(sym2ent.values()))
    print(f"[2/4] resolved {len(sym2ent)}/{len(symbols)} genes -> {len(ids)} Entrez ids")

    rows, t0 = [], time.time()
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        got = _post(f"{API}/molecular-profiles/{PROFILE}/molecular-data/fetch",
                    {"entrezGeneIds": chunk, "sampleListId": SAMPLE_LIST})
        rows.extend(got)
        print(f"      batch {i // BATCH + 1}: +{len(got):6d} records "
              f"(total {len(rows):6d})  {time.time() - t0:5.1f}s")

    samples = sorted({x["sampleId"] for x in rows})
    M = np.full((len(samples), len(ids)), np.nan, dtype=np.float32)
    si = {s: i for i, s in enumerate(samples)}
    gi = {g: i for i, g in enumerate(ids)}
    for x in rows:
        M[si[x["sampleId"]], gi[x["entrezGeneId"]]] = x["value"]

    covered = int((~np.isnan(M)).any(0).sum())
    print(f"[3/4] matrix {M.shape} | genes with data {covered}/{len(ids)} "
          f"| NaN fraction {np.isnan(M).mean():.4f}")

    out = os.path.join(TCGA, "tcga_methylation.npz")
    np.savez_compressed(out, methylation=M,
                        sample_ids=np.array(samples),
                        gene_symbols=np.array([ent2sym[g] for g in ids]))
    print(f"[4/4] wrote {out} ({os.path.getsize(out) // 1024} KB)")


if __name__ == "__main__":
    sys.exit(main())
