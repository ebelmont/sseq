# EHP map-data gap catalogue (E2)

Catalogue of places where the vendored EHP map data (`ext/data/E2/E2_{E,H,P}.csv`)
is missing or inconsistent. Reproduce with `python3 notes/catalogue_ehp_gaps.py`.

## Method — exactness is the arbiter, nothing is inferred

On the E2 page the EHP fiber sequence

    U^N  --E-->  U^{N+1}  --H-->  U^{2N+1}  --P-->  U^N  --E--> ...

is a genuine **long exact sequence** (James). So at every group `M`,

    dim(M) = rank(map into M) + rank(map out of M).

This must hold identically. Any place the *recorded* data violates it is hard
evidence that a map's image is missing or wrong there — we never guess "hidden";
we only report deviations from a theorem. Ranks are computed over F2 from the
recorded images; dims come from `E2_rank.csv`.

Each violation is classified:
- **GENUINE** — every source involved lies *within* the recorded-data frontier,
  so an empty cell there is a certified zero and the deficit is a real gap.
- **truncation** — some source lies *beyond* where that map's data was generated
  (an empty cell is "unrecorded", not "zero"), so the deficit is expected.

Universe: `E2_rank.csv` covers spheres 2..260, total degree s+f up to 130.
Result of the sweep: **176 GENUINE violations, 890 630 truncation-explained.**

## Headline finding — ONE systematic genuine gap

**All 176 genuine violations trace to a single omission: the P-map image of every
odd-sphere fundamental class `(2N+1, 0, 0)` is not recorded.** These images are the
Whitehead squares `P(ι_{2N+1}) = [ι_N, ι_N] ∈ π_{2N-1}(S^N)` at bidegree
`(N, N-1, 2)`. Each missing P-row surfaces twice in the sweep:
- 85× as `in=H, out=P` at the source `U^{2N+1}` position `(2N+1,0,0)` (the class ι
  is neither hit by H nor visibly mapped out by P → deficit 1);
- 91× as `in=P, out=E` at the target `U^N` position `(N,N-1,2)` (the Whitehead
  square looks un-hit because its P-source row is absent → deficit 1).

The data omits **precisely** the rows where the Whitehead square is nonzero: the
91 affected spheres `2N+1` are exactly the odd spheres ≥5 **excluding** the
Hopf-invariant-one dimensions `{7, 15, 31, 63, 127, 191}` (where `[ι_N,ι_N]=0`
and ι is instead hit by H, so exactness already holds and no row is expected).

Concrete anchor (N=2): `P(ι_5)` should land on `S2_1_2` = `h0·h1` = `2η ∈ π_3(S^2)`;
`E2_P.csv` has rows for `5_0_1, 5_0_2, …` but **no** `5_0_0` row, so `S2_1_2`
appears un-hit and `E2` exactness fails there by 1.

### Full list — odd spheres `2N+1` missing their fundamental-class P-row (91)
```
5 9 11 13 17 19 21 23 25 27 29 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61
65 67 69 71 73 75 77 79 81 83 85 87 89 91 93 95 97 99 101 103 105 107 109 111
113 115 117 119 121 123 125 129 131 133 135 137 139 141 143 145 147 149 151 153
155 157 159 161 163 165 167 169 171 173 175 177 179 181 183 185 187 189 193 195 197
```
(Full per-class list, including the mirror `(N,N-1,2)` positions, in the script's
`genuine_gaps` dump.)

## Generation frontiers (truncation — systematic, expected)

Beyond these caps the maps simply weren't generated; empty cells there are
"unknown", not zero, and account for the 890 630 truncation-classified deficits
(mostly h0-towers continuing above each cap).

| map | source spheres | max source f | max source s+f |
|-----|----------------|--------------|----------------|
| E   | 2 .. 130       | 100          | 100            |
| H   | 2 .. 98        | 99           | 100            |
| P   | 5 .. 197       | **46**       | 97             |

P's `f ≤ 46` cap is the tightest and the one most likely to matter in practice:
any P-source above filtration 46 is unrecorded.

## Structural absences (whole spheres with no rows)

A sphere in a map's domain with nonzero target groups but **no recorded source
rows at all**. Exactness cannot certify these either way (the whole sphere reads
as "unrecorded"), so they are candidate gaps unless known to be zero:

- **P: `7, 15, 31, 63, 127, 191`** — the Hopf-invariant-one dimensions
  `2N+1` for `N ∈ {3,7,15,…}`. Standard theory: `P(ι)=[ι_N,ι_N]=0` and ι is hit by
  H, so P is genuinely zero here — **not** a gap (verified: none produce a genuine
  exactness violation). Plus the tail `199..259` (beyond P's sphere frontier).
- **H: `65, 81, 89, 93, 97`** — isolated absences *inside* H's recorded range
  (2..98) with no H rows. Not certifiable from exactness alone (sphere reads as
  unrecorded) → **candidate gaps worth a direct check**. Plus the tail `99..129`.
- **E: `131..259`** — entirely the tail beyond E's sphere frontier (130).

## What is NOT a gap (dashed-line clarification)

The vast majority of empty E/H/P cells are **exactness-consistent genuine zeros**,
not missing data. Example: `E(S2_4_3)=0` is *forced* because `S2_4_3` is hit by P
from `S5_3_1` (`im P = ker E`). A dashed stub in the fiber viewer marks "no
recorded image with a nonzero target group present" — which is a genuine zero in
almost every case. Only the cells the **exactness-diagnostics panel flags** (the
Whitehead-square positions above) are real data gaps.

## Reproduction

```
python3 notes/catalogue_ehp_gaps.py            # uses ../data/E2
python3 notes/catalogue_ehp_gaps.py /path/to/E2
```
The script prints the frontiers, structural absences, violation counts, and the
missing-Whitehead-square sphere list. It is pure exactness bookkeeping over the
vendored CSVs — no solve, ~1 s.
