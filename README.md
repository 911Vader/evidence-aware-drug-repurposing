# Evidence-Aware Multimodal Fusion for Cold-Start Drug Repurposing

CBIO 7215 — Final Project. A relation-aware GNN (R-GCN-style message passing,
matching the Week 3/4 lecture formulas) with a DistMult link-scoring decoder,
trained to predict **Compound-treats-Disease** edges on **Hetionet v1.0**,
comparing five ways of building a compound's node representation and asking
not just "does chemistry help?" but **when** it helps, and whether any gain
is real chemistry or just architecture.

See `report/CBIO7215_Final_Project_Report.docx` for the full write-up. This
README covers how to run the code and what changed from the first pass.

## What changed since the first draft (and why)

Three real issues were found and fixed before these results were trusted:

1. **A normalization bug in the R-GCN layer.** The formula taught in lecture
   requires each relation's neighbor mean to be computed separately, then the
   per-relation means summed. The original implementation instead pooled
   every relation's messages and neighbor counts into one shared accumulator
   and divided once — a single global mean over the union of all neighbors,
   which let high-degree relations (Gene-participates-BiologicalProcess,
   ~560K edges) drown out rare ones (Compound-treats-Disease, 755 edges).
   Fixed in `src/model.py`'s `RelationalGraphConvLayer`.
2. **Single-seed, 50-negative-sample evaluation was not reliable enough to
   support a claim.** Evaluation now ranks every test edge against *every*
   eligible disease (Hetionet only has 137), and every experiment is run
   across 3 seeds and reported as mean ± SD. The seed-to-seed spread turned
   out to be large for some variants (e.g. cold-start fingerprint-fusion
   AUROC ranged 0.75–0.86 across seeds) — exactly why this mattered.
3. **A one-compound edge case in the `random_fingerprint` shuffle.** The
   derangement fixup in `make_shuffled_fingerprints` special-cased "exactly
   one fixed point remains" by leaving it unchanged (`torch.roll` on a
   length-1 tensor is a no-op), so whenever a random draw happened to leave
   precisely one compound mapped to itself, that compound silently kept its
   own real fingerprint instead of a shuffled one. This affected 2 of the 3
   seeds (0 and 2). Fixed in `src/train.py`; all 6 `random_fingerprint` runs
   were re-trained under the corrected, verified derangement.

Those fixes changed the scientific story in places — see the report's
Section 5.1 and Discussion for exactly what moved and by how much.

## The five variants compared

- **baseline** — learned per-node ID embedding only.
- **fingerprint** — the embedding concatenated with a projected real Morgan
  fingerprint for Compound nodes.
- **random_fingerprint** — identical architecture to `fingerprint`, but each
  compound's fingerprint row is swapped for a different, randomly assigned
  compound's fingerprint (fixed per seed, true derangement). A control:
  isolates whether any gain from `fingerprint` is real chemical signal or
  just extra capacity.
- **adaptive** — the proposed extension: a learned evidence gate
  `g_c = sigmoid(a·log(1+degree_c) + b)` convexly blends the embedding and
  the projected fingerprint per compound — degree-conditioned molecular
  feature fusion before relational message passing, not a literal choice
  between "topology" and "chemistry" — where `degree_c` is that compound's
  non-CtD graph degree (computed from the train-only graph, so no leakage).
- **adaptive_random_fingerprint** — a novelty control: the exact `adaptive`
  architecture (same gate), fed the same shuffled fingerprints as
  `random_fingerprint`. Isolates whether `adaptive`'s advantage comes from
  the gating mechanism itself or from the real compound↔structure
  correspondence it is given. Not a model anyone would deploy — it exists
  purely to answer that question (see report Section 7).

## 1. Fast demo (toy data, ~15 seconds, no download needed)

```bash
pip install -r requirements.txt
python src/train.py --data toy_data/graph_toy.pt --split random --variant adaptive --epochs 15 --toy --seed 0
```

Swap `--variant` between `baseline` / `fingerprint` / `random_fingerprint` /
`adaptive` / `adaptive_random_fingerprint`, and `--split` to `cold_start`,
to see the other combinations.

## 2. Full-scale reproduction (real Hetionet, ~3–5 min per run on CPU)

```bash
bash scripts/download_data.sh        # pulls ~17MB of raw TSV/SIF files
python src/data_prep.py              # builds data/processed/graph.pt
```

Then reproduce the main 3-seed x 2-split x 4-variant matrix (24 runs):

```bash
for seed in 0 1 2; do
  for split in random cold_start; do
    for variant in baseline fingerprint random_fingerprint adaptive; do
      python src/train.py --split "$split" --variant "$variant" --epochs 50 --seed "$seed"
    done
  done
done
```

...plus the 3-seed x 2-split novelty-control ablation (6 more runs, 30 total):

```bash
for seed in 0 1 2; do
  for split in random cold_start; do
    python src/train.py --split "$split" --variant adaptive_random_fingerprint --epochs 50 --seed "$seed"
  done
done

python src/summarize_results.py   # Table 2/3/4, Figures 1/2/3 -- everything the report shows
```

...plus a third, additional-baseline comparison for the report's Section 5.2
(3-seed x 2-split, `--variant baseline --num-layers 0`): a plain learned-
embedding + DistMult model with the R-GCN message-passing layers switched
off entirely, so it's guaranteed -- by construction, not by inspecting a
paper's methods section -- to use identical data and evaluation settings to
every other number above:

```bash
for seed in 0 1 2; do
  for split in random cold_start; do
    python src/train.py --split "$split" --variant baseline --num-layers 0 --seed "$seed" --tag nognn
  done
done
```

`--tag nognn` keeps these 6 runs in their own
`results/{split}_baseline_seed{N}_nognn.json` files, separate from the
normal `baseline` runs above (same variant name, different architecture via
`--num-layers 0`, so the tag avoids a filename collision).

Each run saves its own `results/{split}_{variant}_seed{N}.json` (so re-runs
never clobber each other), a loss-curve PNG, and a full log. Every number in
the report traces back to one of these files.

## 3. Project layout

```
src/
  data_prep.py          load Hetionet + DrugBank, compute fingerprints, build splits
  make_toy_data.py       build the small checked-in toy_data/graph_toy.pt subgraph
  model.py               R-GCN encoder (fixed normalization) + DistMult decoder +
                         EvidenceGate + the 5-variant fusion logic
  train.py               leakage-safe training loop for any {split, variant, seed};
                         also where make_shuffled_fingerprints (fixed derangement) lives
  evaluate.py            exhaustive (all-eligible-disease) ranking: AUROC/AUPRC/Hits@k
                         + per-edge diagnostics (compound, graph degree, rank)
  summarize_results.py   mean +/- SD aggregation, comparison figure, the
                         low/medium/high graph-evidence stratification analysis,
                         and the adaptive vs. adaptive_random_fingerprint ablation
scripts/
  download_data.sh       fetches the ~17MB of raw public data
toy_data/
  graph_toy.pt            small subgraph for the fast demo (checked in, ~9MB)
data/                   raw/ and processed/ populated by the commands above
                        (processed graph not shipped -- regenerate as needed;
                        it's ~400MB once the fingerprint matrix is in; raw/
                        IS included here since it's small, ~17MB)
results/                 one JSON per run (metrics + loss history + per-edge
                        diagnostics), plus summary_table.csv,
                        evidence_stratification.csv, and
                        adaptive_ablation_table.csv
figures/                 training-loss curves, the comparison bar chart, the
                        evidence-stratification bar chart, and the
                        adaptive-vs-shuffled ablation bar chart
logs/                   plain-text run logs
report/                  the technical report (.docx)
```

## 4. What is and isn't novel here

Reused directly from course material: R-GCN message passing (Week 3 L2),
the DistMult decoder (Week 4 L1, TxGNN), Hetionet as the knowledge graph,
and the drug-held-out cold-start split definition (Week 4 L2).

The contribution is the **evidence-aware gated fusion** mechanism, the
**controlled comparison against a shuffled-fingerprint ablation**, and a
**second ablation that isolates the gate's own architecture from the
chemistry it's given** (`adaptive_random_fingerprint`). Together these show
that naively concatenating chemical structure onto a KG-GNN is not reliably
beneficial — and can underperform a graph-only model in the realistic
cold-start regime — while the evidence-aware gate recovers overall
performance; and that most of the gate's advantage over naive fusion comes
from the gating/degree-conditioning architecture itself (it still wins even
fed scrambled fingerprints), with a smaller, metric-dependent contribution
from real chemistry visible mainly in top-k retrieval. This is consistent
with, and was partly motivated by, recent literature questioning whether
explicit molecular structure helps KG-based repurposing models (see the
report's references).

## 4b. Comparison to additional baselines (report Section 5.2)

Two more points of comparison were added after instructor feedback asking
for more baselines, and the report is explicit about which is verified and
which isn't. (1) Our own no-message-passing baseline above (`--num-layers
0`) is guaranteed to match this project's data and evaluation settings
exactly, since it's the same codebase and pipeline. It collapses to
near-chance in the cold-start split (AUROC ~0.48 vs. 0.84 for the R-GCN
baseline) because a held-out compound's embedding never gets updated by
message passing from its other graph edges — a clean demonstration that the
R-GCN encoder itself, not just the gated fusion on top of it, is what makes
cold-start generalization possible here. (2) Liu et al. (2021; arXiv:2103.10367,
workshop version arXiv:2007.05292) report an R-GCN baseline on the identical
755-edge Hetionet CtD relation with a matching 151-edge random-split test
set, but their paper does not state whether ranking is against all ~47K
entities or only the 137 Disease nodes, and no public code for it could be
found to check — so per our instructor's own bar ("same data AND evaluation
settings"), it's cited as related work with that caveat stated plainly,
not as a verified head-to-head number.

## 5. Honesty note

Every metric in the report comes from an actual run of this exact code —
the JSON files in `results/` are the source of truth, and the mean ± SD
tables are computed, not estimated. Two claims in an earlier draft of the
report overstated what the numbers actually showed (adaptive does **not**
beat baseline on literally every metric — cold-start Hits@1 is one
exception — and it does **not** stay above baseline in literally every
evidence-stratification bucket — cold-start medium-evidence is roughly
tied). Both are corrected to the precise, narrower claims the data
supports. Similarly, where a result contradicted the original hypothesis
(the evidence-aware gate's benefit is concentrated in the medium/high
graph-evidence tertiles, not the low one, in the cold-start stratification),
the report says so rather than reframing it.
