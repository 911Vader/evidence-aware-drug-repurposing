"""
train.py

Trains a Relational-GNN + DistMult link predictor for Compound-treats-
Disease (CtD) edge prediction on Hetionet, in one of FOUR architecture
variants and one of two evaluation splits:

  variants : "baseline"           -- node features = learned embedding only
             "fingerprint"        -- learned embedding CONCATENATED with a
                                      projected real Morgan fingerprint for
                                      Compound nodes
             "random_fingerprint" -- identical architecture to "fingerprint",
                                      but each compound's fingerprint row is
                                      replaced by a DIFFERENT, randomly
                                      assigned compound's fingerprint (fixed
                                      per seed). Isolates whether any gain
                                      from "fingerprint" comes from real
                                      chemical signal or just from the extra
                                      parameters/dimensionality.
             "adaptive"           -- evidence-aware gated fusion: a learned
                                      gate g_c = sigmoid(a*log(1+degree_c)+b)
                                      convexly blends the learned embedding
                                      and the projected fingerprint, instead
                                      of concatenating them, so compounds
                                      with little non-CtD graph evidence can
                                      lean more on chemistry.
             "adaptive_random_fingerprint" -- identical architecture to
                                      "adaptive" (same gate, same fusion
                                      mechanism), but fed the SAME shuffled
                                      fingerprints as "random_fingerprint".
                                      Isolates whether adaptive's gain comes
                                      from the correct compound<->structure
                                      correspondence or just from the gated
                                      fusion/regularization mechanism itself,
                                      independent of which fingerprint it is
                                      given. Not a model anyone would deploy
                                      -- it exists purely as a control.

  splits   : "random"      -- transductive 80/20 edge split
             "cold_start"  -- drug-held-out split (Week 4 L2's strictest
                               realistic evaluation)

CRITICAL leakage control: for whichever split is requested, the CtD edges
used for MESSAGE PASSING (i.e. present in the graph the encoder runs over)
are restricted to that split's TRAIN edges only. Test edges are never in
the graph the encoder sees, in either split.

Evaluation now ranks every test edge against ALL eligible diseases (not a
sampled subset) -- see evaluate.py.

Usage:
    python src/train.py --split random --variant baseline --seed 0
    python src/train.py --split cold_start --variant adaptive --seed 2 --epochs 40
"""

import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn

from evaluate import evaluate_link_prediction
from model import (DistMultDecoder, RelationalGNNEncoder,
                    all_relation_names_with_inverse,
                    build_bidirectional_edge_index,
                    compute_non_target_degree)

TARGET_RELATION = "CtD"

VARIANT_TO_FUSION_MODE = {
    "baseline": None,
    "fingerprint": "concat",
    "random_fingerprint": "concat",
    "adaptive": "gated",
    "adaptive_random_fingerprint": "gated",
}

# Variants that must be fed shuffled (not real) fingerprints.
SHUFFLED_FINGERPRINT_VARIANTS = {"random_fingerprint", "adaptive_random_fingerprint"}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def restrict_ctd_to_train(rel_edges, ctd_src, ctd_dst, train_idx):
    """Returns a copy of rel_edges with the CtD entry replaced by ONLY the
    training-split edges, so message passing never sees test edges."""
    out = dict(rel_edges)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    out[TARGET_RELATION] = (ctd_src[train_idx_t], ctd_dst[train_idx_t])
    return out


def build_true_dst_by_src(ctd_src, ctd_dst):
    d = {}
    for s, t in zip(ctd_src.tolist(), ctd_dst.tolist()):
        d.setdefault(s, set()).add(t)
    return d


def sample_train_negatives(pos_src, disease_pool, true_dst_by_src, rng):
    """One negative disease per positive edge, filtered against ALL known
    true CtD targets for that compound (not just the train split), so we
    never train the model to call a real-but-held-out treatment 'false'."""
    negs = []
    pool_size = len(disease_pool)
    for s in pos_src.tolist():
        known = true_dst_by_src.get(s, set())
        while True:
            cand = disease_pool[rng.randrange(pool_size)]
            if cand not in known:
                negs.append(cand)
                break
    return torch.tensor(negs, dtype=torch.long)


def make_shuffled_fingerprints(fp_matrix, fp_mask, seed):
    """Random-fingerprint control: permute fingerprint rows AMONG the
    compounds that actually have one, so every compound still gets some
    real compound's fingerprint (same value distribution, same dimension,
    same number of nonzero rows) but not its own -- breaking the
    structure<->identity correspondence while preserving architecture and
    parameter count exactly.

    This must be a TRUE derangement (no compound ever maps to itself). A
    single random permutation of ~1,552 elements has, on average, about one
    fixed point; those are resolved by repeatedly swapping each fixed
    position with its neighbor until none remain, then verified with an
    assertion. (An earlier version of this fixup special-cased the
    "exactly one fixed point" case by leaving it unchanged instead of
    swapping it -- i.e. `torch.roll` on a length-1 tensor is a no-op --
    which meant that whenever a random draw happened to leave precisely one
    compound mapped to itself, it silently kept its own fingerprint. Fixed
    here.)"""
    g = torch.Generator().manual_seed(seed + 10_000)
    valid_idx = torch.nonzero(fp_mask, as_tuple=True)[0]
    n = valid_idx.numel()
    perm = torch.randperm(n, generator=g)
    positions = torch.arange(n)
    for _ in range(10):
        fixed = (perm == positions)
        if not fixed.any():
            break
        for i in fixed.nonzero(as_tuple=True)[0].tolist():
            if perm[i].item() != i:
                continue  # already resolved by an earlier swap in this pass
            j = (i + 1) % n
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    assert (perm == positions).sum().item() == 0, "derangement fixup did not converge"
    perm_idx = valid_idx[perm]
    shuffled = torch.zeros_like(fp_matrix)
    shuffled[valid_idx] = fp_matrix[perm_idx]
    return shuffled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/graph.pt")
    ap.add_argument("--split", choices=["random", "cold_start"], required=True)
    ap.add_argument("--variant", choices=list(VARIANT_TO_FUSION_MODE.keys()), required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--fp-proj-dim", type=int, default=32)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--toy", action="store_true",
                     help="Smaller defaults appropriate for the toy_data subgraph.")
    ap.add_argument("--tag", default="",
                     help="Optional suffix appended to run_name, so an "
                          "architecture variant that reuses an existing "
                          "--variant name under different hyperparameters "
                          "(e.g. --num-layers 0 for a no-GNN embedding-only "
                          "baseline) doesn't overwrite that variant's normal "
                          "results file.")
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    run_name = f"{args.split}_{args.variant}_seed{args.seed}" + ("_toy" if args.toy else "") + (f"_{args.tag}" if args.tag else "")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{run_name}.log")
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"=== Run: {run_name} ===")
    log(f"Loading {args.data} ...")
    payload = torch.load(args.data, weights_only=False)

    kind_to_id = {k: i for i, k in enumerate(payload["kind_list"])}
    disease_pool = torch.nonzero(
        payload["node_kind_idx"] == kind_to_id["Disease"], as_tuple=True
    )[0].tolist()
    log(f"num_nodes={payload['num_nodes']}  disease_pool_size={len(disease_pool)}")

    ctd_src, ctd_dst = payload["ctd_src"], payload["ctd_dst"]
    split = payload["splits"][args.split]
    train_idx, test_idx = split["train"], split["test"]
    train_src, train_dst = ctd_src[train_idx], ctd_dst[train_idx]
    test_src, test_dst = ctd_src[test_idx], ctd_dst[test_idx]
    log(f"split={args.split}  train_edges={len(train_idx)}  test_edges={len(test_idx)}")
    if args.split == "cold_start":
        log(f"  held-out compounds: {len(split['heldout_compounds'])}")

    all_true_dst_by_src = build_true_dst_by_src(ctd_src, ctd_dst)

    rel_edges_train_graph = restrict_ctd_to_train(payload["rel_edges"], ctd_src, ctd_dst, train_idx)
    relation_names_bidir = all_relation_names_with_inverse(payload["relation_names"])
    rel_edge_index = build_bidirectional_edge_index(rel_edges_train_graph, payload["relation_names"])

    # Non-CtD graph degree is computed from the SAME train-only graph the
    # encoder actually sees, so the evidence-aware gate (and any later
    # evidence-stratified analysis) never uses information leaked from the
    # held-out test edges.
    compound_degree = compute_non_target_degree(
        rel_edges_train_graph, payload["num_nodes"], target_relation=TARGET_RELATION)

    fusion_mode = VARIANT_TO_FUSION_MODE[args.variant]
    use_fp = fusion_mode is not None
    if args.num_layers == 0:
        # No-GNN ablation (plain learned-embedding + DistMult, no message
        # passing at all): with zero RelationalGraphConvLayers the encoder
        # returns the raw node embedding (width embed_dim) unchanged, so it
        # must equal hidden_dim, the width DistMultDecoder below is built
        # for.
        assert args.embed_dim == args.hidden_dim, (
            "--num-layers 0 requires --embed-dim == --hidden-dim "
            "(no projection layer exists to bridge them)")
        assert not use_fp, "--num-layers 0 is only defined for variant=baseline (no fingerprint fusion)"
    encoder = RelationalGNNEncoder(
        num_nodes=payload["num_nodes"],
        relation_names=relation_names_bidir,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        fingerprint_bits=payload["fingerprint_bits"] if use_fp else None,
        fingerprint_proj_dim=args.fp_proj_dim if use_fp else None,
        fusion_mode=fusion_mode,
    )
    decoder = DistMultDecoder(args.hidden_dim)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    if args.variant in SHUFFLED_FINGERPRINT_VARIANTS:
        fp_matrix = make_shuffled_fingerprints(payload["fingerprint_matrix"], payload["fingerprint_mask"], args.seed)
    elif use_fp:
        fp_matrix = payload["fingerprint_matrix"]
    else:
        fp_matrix = None
    fp_mask = payload["fingerprint_mask"] if use_fp else None

    loss_history = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        optimizer.zero_grad()

        node_emb = encoder(rel_edge_index, fingerprint_matrix=fp_matrix, fingerprint_mask=fp_mask,
                            compound_degree=compound_degree)

        neg_dst = sample_train_negatives(train_src, disease_pool, all_true_dst_by_src, rng)

        pos_scores = decoder.score(node_emb[train_src], node_emb[train_dst])
        neg_scores = decoder.score(node_emb[train_src], node_emb[neg_dst])

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        loss = bce(scores, labels)

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            log(f"  epoch {epoch:3d}/{args.epochs}  loss={loss.item():.4f}  "
                f"elapsed={time.time() - t_start:.1f}s")

    train_time = time.time() - t_start
    log(f"Training done in {train_time:.1f}s")

    log("Evaluating on test split (ranked against ALL eligible diseases)...")
    encoder.eval()
    with torch.no_grad():
        final_node_emb = encoder(rel_edge_index, fingerprint_matrix=fp_matrix, fingerprint_mask=fp_mask,
                                  compound_degree=compound_degree)
    metrics, per_edge = evaluate_link_prediction(
        final_node_emb, decoder, test_src, test_dst, disease_pool, all_true_dst_by_src,
        seed=args.seed, compound_degree=compound_degree,
    )
    log(f"RESULTS {run_name}: " + json.dumps(metrics, indent=None))

    if fusion_mode == "gated":
        a_val, b_val = encoder.gate.a.item(), encoder.gate.b.item()
        log(f"  learned gate: a={a_val:.4f} b={b_val:.4f} "
            f"(g=0.5 at degree={torch.exp(torch.tensor(-b_val / max(a_val, 1e-6))).item() - 1:.1f})")
        gate_params = {"a": a_val, "b": b_val}
    else:
        gate_params = None

    result_record = {
        "run_name": run_name,
        "split": args.split,
        "variant": args.variant,
        "seed": args.seed,
        "toy": args.toy,
        "hyperparams": {
            "epochs": args.epochs, "lr": args.lr, "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
            "fp_proj_dim": args.fp_proj_dim if use_fp else None,
            "weight_decay": args.weight_decay, "seed": args.seed,
        },
        "data_stats": {
            "num_nodes": payload["num_nodes"],
            "train_edges": len(train_idx),
            "test_edges": len(test_idx),
        },
        "gate_params": gate_params,
        "train_time_sec": train_time,
        "loss_history": loss_history,
        "metrics": metrics,
        "per_edge": per_edge,
    }
    out_path = os.path.join(args.out_dir, f"{run_name}.json")
    with open(out_path, "w") as f:
        json.dump(result_record, f, indent=2)
    log(f"Saved results to {out_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 3.5))
        plt.plot(loss_history)
        plt.xlabel("epoch")
        plt.ylabel("BCE training loss")
        plt.title(run_name)
        plt.tight_layout()
        fig_path = os.path.join(args.fig_dir, f"{run_name}_loss.png")
        plt.savefig(fig_path, dpi=120)
        plt.close()
        log(f"Saved loss curve to {fig_path}")
    except Exception as e:
        log(f"(figure save skipped: {e})")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
