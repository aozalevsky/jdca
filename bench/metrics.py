"""Contact metrics beyond top-k precision for every estimator, on AlphaFold DB targets (docs/comparison.md, "Other contact metrics").

Estimators: jdca.gaussdca (GaussDCA.jl's estimator), its adaptive mode, jdca.humphreys_dca (the DCA used by Humphreys et al., their
settings and APC) and jdca.ridge_dca. Truth: Cb-Cb < 8 A (Ca for Gly) in the AFDB model; only pairs whose two residues have pLDDT >= 70,
|i - j| >= 6. For the top k = L/5, L/2 and L pairs:
  precision        fraction of true contacts;
  long-range       precision among the top k pairs with |i - j| >= 24;
  tolerant         fraction of pairs with a true contact within +-1 in i and in j;
  patch recall     true contacts linked when within +-2 in i and in j form patches; fraction of the patches hit;
  redundancy       correct predictions per hit patch (lower = the correct ones are spread over more patches);
  SSE-pair         DSSP helix (H, G, I) and strand (E, B) segments of >= 3 residues; fraction of segment pairs in contact that are hit;
and the average precision over the whole ranking.

Run bench/benchmark.py or bench/windowed.py first (they write bench/out/<id>/aln.fasta). Needs DSSP's mkdssp (conda-forge package
dssp; tested with 4.6.1), on the PATH or given with --mkdssp.

    python bench/metrics.py P34884 P0A9S1 P57740 P49588 P53396 P29474 P17301
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import jdca                                                     # noqa: E402

METHODS = {'gaussdca': lambda f: jdca.gaussdca(jdca.read_alignment(f)),
           'adaptive': lambda f: jdca.gaussdca(jdca.read_alignment(f, 1.0), mode='adaptive'),
           'humphreys': lambda f: jdca.humphreys_dca(jdca.read_alignment(f, 1.0)),
           'ridge': lambda f: jdca.ridge_dca(jdca.read_alignment(f, 1.0))}
KS = ('L/5', 'L/2', 'L')
SSE3 = {'H': 'a', 'G': 'a', 'I': 'a', 'E': 'b', 'B': 'b'}           # DSSP's 8 states -> helix 'a', strand 'b' (anything else: 'c')


def structure(cif, mkdssp='mkdssp'):
    """Cb coordinates (Ca for Gly), pLDDT and the DSSP helix / strand assignment per residue of the AFDB model."""
    import biotite.structure as bs
    import biotite.structure.io.pdbx as pdbx
    from biotite.application.dssp import DsspApp
    a = pdbx.get_structure(pdbx.CIFFile.read(cif), model=1, extra_fields=['b_factor'])
    a = a[a.hetero == False]                                    # noqa: E712
    xyz, plddt = [], []
    for r in bs.residue_iter(a):
        p = r[r.atom_name == ('CA' if r.res_name[0] == 'GLY' else 'CB')]
        p = p if len(p) else r[r.atom_name == 'CA']
        xyz.append(p.coord[0])
        plddt.append(p.b_factor[0])
    sse = DsspApp.annotate_sse(a, mkdssp)
    assert len(sse) == len(xyz), (cif, len(sse), len(xyz))
    return np.array(xyz), np.array(plddt), np.array([SSE3.get(s, 'c') for s in sse])


def segments(sse, minlen=3):
    """Segment index per residue for helix ('a') and strand ('b') runs of >= minlen residues, else -1."""
    seg, k, i = np.full(len(sse), -1), 0, 0
    while i < len(sse):
        j = i
        while j < len(sse) and sse[j] == sse[i]:
            j += 1
        if sse[i] in ('a', 'b') and j - i >= minlen:
            seg[i:j] = k
            k += 1
        i = j
    return seg


def patches(pairs):
    """Patch label per true contact: connected components, two contacts linked when |di| <= 2 and |dj| <= 2."""
    index = {p: n for n, p in enumerate(pairs)}
    parent = list(range(len(pairs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (i, j), n in index.items():
        for di in range(-2, 3):
            for dj in range(-2, 3):
                m = index.get((i + di, j + dj))
                if m is not None:
                    parent[find(n)] = find(m)
    return {p: find(n) for p, n in index.items()}


def evaluate(scores, xyz, plddt, sse):
    L = len(xyz)
    i, j = np.triu_indices(L, 6)
    keep = (plddt[i] >= 70) & (plddt[j] >= 70)
    i, j = i[keep], j[keep]
    contact = np.linalg.norm(xyz[i] - xyz[j], axis=1) < 8.0
    true = list(zip(i[contact].tolist(), j[contact].tolist()))
    tset, patch, seg = set(true), patches(true), segments(sse)
    sse_pairs = {(seg[a], seg[b]) for a, b in true if seg[a] >= 0 and seg[b] >= 0 and seg[a] != seg[b]}
    o = np.argsort(-scores[i, j], kind='stable')
    ri, rj, rc = i[o], j[o], contact[o]
    out = {}
    for name, k in zip(KS, (L // 5, L // 2, L)):
        ti, tj, tc = ri[:k], rj[:k], rc[:k]
        hit = [patch[(a, b)] for a, b, c in zip(ti.tolist(), tj.tolist(), tc) if c]
        far = np.flatnonzero(rj - ri >= 24)[:k]
        hs = {(seg[a], seg[b]) for a, b, c in zip(ti, tj, tc) if c and seg[a] >= 0 and seg[b] >= 0 and seg[a] != seg[b]}
        out[f'precision@{name}'] = float(tc.mean())
        out[f'long-range@{name}'] = float(rc[far].mean())
        out[f'tolerant@{name}'] = float(np.mean([any((a + x, b + y) in tset for x in (-1, 0, 1) for y in (-1, 0, 1))
                                                 for a, b in zip(ti.tolist(), tj.tolist())]))
        out[f'patch recall@{name}'] = len(set(hit)) / max(len(set(patch.values())), 1)
        out[f'redundancy@{name}'] = len(hit) / max(len(set(hit)), 1)
        out[f'SSE-pair@{name}'] = len(hs & sse_pairs) / max(len(sse_pairs), 1)
    cum = np.cumsum(rc)
    out['average precision'] = float((cum[rc] / (np.flatnonzero(rc) + 1)).sum() / max(rc.sum(), 1))
    info = dict(L=L, contacts=len(true), patches=len(set(patch.values())), sse_pairs=len(sse_pairs))
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ids', nargs='+')
    ap.add_argument('--mkdssp', default='mkdssp', help='the DSSP program (default: mkdssp on the PATH)')
    a = ap.parse_args()
    path = os.path.join(HERE, 'out', 'metrics.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    for uid in a.ids:
        fasta = os.path.join(HERE, 'out', uid, 'aln.fasta')
        xyz, plddt, sse = structure(sorted(glob.glob(os.path.join(HERE, 'data', uid, 'AF-*-model_v*.cif')))[-1], a.mkdssp)
        res[uid] = {}
        for m, fn in METHODS.items():
            res[uid][m], res[uid]['info'] = evaluate(fn(fasta).scores, xyz, plddt, sse)
        print(f'{uid} {res[uid]["info"]}: average precision ' +
              ', '.join(f'{m} {res[uid][m]["average precision"]:.3f}' for m in METHODS), flush=True)
        json.dump(res, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
