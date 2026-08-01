#!/usr/bin/env python3
"""Catalogue missing/inconsistent EHP map data, using exactness as ground truth.

On E2 the EHP fiber sequence

    U^N  -E->  U^{N+1}  -H->  U^{2N+1}  -P->  U^N  -E-> ...

is a genuine long exact sequence, so at every group M:  dim(M) = rank(in) + rank(out).
Any recorded violation is real evidence of missing (or wrong) map data — nothing is
inferred beyond that theorem. Each violation is classified as

  GENUINE     — all involved sources lie within the recorded-data frontier, so an
                empty cell there is certified zero and the deficit is a real gap;
  truncation  — some source lies beyond where that map's data was generated (an
                empty cell is "unrecorded", not "zero"), so the deficit is expected.

Run from anywhere:  python3 catalogue_ehp_gaps.py [path/to/data/E2]
Defaults to ../data/E2 relative to this file.
"""
import csv, os, sys
from collections import defaultdict

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'E2')

# ---- dims (which groups are nonzero) ----
dim = {}
max_sphere = max_total = 0
with open(os.path.join(DATA, 'E2_rank.csv')) as fh:
    r = csv.reader(fh); next(r)
    for n, s, f, d in r:
        n, s, f, d = int(n), int(s), int(f), int(d)
        if d > 0:
            dim[(n, s, f)] = d
            max_sphere = max(max_sphere, n); max_total = max(max_total, s + f)

def parse_elt(e):
    p = e.strip().strip('"').split('_')
    return (int(p[0]), int(p[1]), int(p[2])), (int(p[3]) if len(p) > 3 else 0)

def load_map(fn):
    rows = defaultdict(dict); recorded = defaultdict(set)
    for line in open(os.path.join(DATA, fn)):
        line = line.strip()
        if not line or line.startswith('"element"'):
            continue
        a, b = line.split('","')
        src, sidx = parse_elt(a.strip('"')); img = b.strip('"')
        recorded[src[0]].add((src[1], src[2]))
        mask = 0
        if img and img != '0':
            for term in img.split('+'):
                if term.strip():
                    _, tidx = parse_elt(term)
                    mask |= 1 << tidx
        rows[src][sidx] = rows[src].get(sidx, 0) | mask
    return rows, recorded

def f2rank(masks):
    piv = {}; rank = 0
    for m in masks:
        while m:
            hb = m.bit_length() - 1
            if hb in piv: m ^= piv[hb]
            else: piv[hb] = m; rank += 1; break
    return rank

rank = {}; frontier = {}; recorded = {}
for mp, fn in [('E', 'E2_E.csv'), ('H', 'E2_H.csv'), ('P', 'E2_P.csv')]:
    rows, rec = load_map(fn); recorded[mp] = rec
    for src, d in rows.items():
        rank[(mp,) + src] = f2rank(d.values())
    for n, sf in rec.items():
        frontier[(mp, n)] = (max(f for _, f in sf), max(s + f for s, f in sf))

D = lambda T: dim.get(T, 0)
R = lambda mp, T: rank.get((mp,) + T, 0)
def domain(mp, n):
    return n >= 2 if mp in ('E', 'H') else (n >= 5 and n % 2 == 1)
def src_of(mp, M):
    n, s, f = M
    if mp == 'E': return (n - 1, s, f)
    if mp == 'H': nn = (n + 1) // 2; return (nn, s + nn - 1, f + 1)
    return (2 * n + 1, s - n + 1, f - 2)
def target_degree(mp, T):
    n, s, f = T
    if mp == 'E': return (n + 1, s, f)
    if mp == 'H': return (2 * n - 1, s - n + 1, f - 1)
    return ((n - 1) // 2, s + (n - 1) // 2 - 1, f + 2)
valid = lambda T: T[0] >= 2 and T[1] >= 0 and T[2] >= 0
def beyond(mp, T):
    fr = frontier.get((mp, T[0]))
    return True if fr is None else (T[2] > fr[0] or T[1] + T[2] > fr[1])

# ---- exactness sweep: each nonzero middle group in each of its roles ----
violations = []; seen = set()
for M, d in dim.items():
    m, s, f = M
    roles = []
    if m >= 3 and domain('H', m):                    roles.append(('E', src_of('E', M), 'H'))
    if m % 2 == 1 and m >= 5:                         roles.append(('H', src_of('H', M), 'P'))
    if domain('P', 2 * m + 1) and domain('E', m):     roles.append(('P', src_of('P', M), 'E'))
    for mi, in_src, mo in roles:
        key = (M, mi, mo)
        if key in seen: continue
        seen.add(key)
        iv = valid(in_src)
        rin = R(mi, in_src) if iv else 0
        rout = R(mo, M)
        if d == rin + rout: continue
        trunc = (iv and beyond(mi, in_src)) or beyond(mo, M) or (iv and in_src[0] > max_sphere)
        violations.append(dict(M=M, dim=d, in_map=mi, in_src=in_src, rank_in=rin,
                               out_map=mo, rank_out=rout, deficit=d - rin - rout,
                               kind='truncation' if trunc else 'GENUINE'))

# ---- structural absences (map defined + nonzero target, but sphere never a source) ----
dom_nonzero = defaultdict(lambda: defaultdict(int))
for (n, s, f), d in dim.items():
    for mp in ('E', 'H', 'P'):
        if domain(mp, n) and D(target_degree(mp, (n, s, f))) > 0:
            dom_nonzero[mp][n] += 1
absent = {mp: [n for n in sorted(dom_nonzero[mp]) if n not in recorded[mp]] for mp in 'EHP'}

gen = [v for v in violations if v['kind'] == 'GENUINE']
tr  = [v for v in violations if v['kind'] == 'truncation']

print(f"# EHP map-data gap catalogue (E2, from {os.path.relpath(DATA)})")
print(f"# rank universe: spheres 2..{max_sphere}, total degree up to {max_total}\n")
print("## Recorded extent per map (frontier = generation cap)")
for mp in 'EHP':
    ns = sorted(recorded[mp])
    fs = [fr for (m, _), fr in frontier.items() if m == mp]
    print(f"  {mp}: spheres {ns[0]}..{ns[-1]} ({len(ns)}); max source f={max(a for a,_ in fs)}, "
          f"max source s+f={max(b for _,b in fs)}")
print("\n## Structural absences (defined + nonzero target, sphere never a source)")
for mp in 'EHP':
    print(f"  {mp}: {absent[mp] or '(none)'}")
print(f"\n## Exactness violations: {len(gen)} GENUINE, {len(tr)} truncation-explained")
roots = defaultdict(int)
for v in gen:
    roots[(v['in_map'], v['out_map'])] += 1
print("  GENUINE by (in,out):", dict(roots))
missing_P0 = sorted({v['in_src'][0] for v in gen if v['in_map'] == 'P' and v['in_src'][1:] == (0, 0)}
                    | {v['M'][0] for v in gen if v['out_map'] == 'P' and v['M'][1:] == (0, 0)})
print(f"  odd spheres 2N+1 with missing fundamental-class P-row ({len(missing_P0)}): {missing_P0}")
