"""
make_toy_data.py

Builds a small induced subgraph of the full processed Hetionet payload
(data/processed/graph.pt) so a grader can run the ENTIRE pipeline --
data loading, model, training, evaluation -- in well under a minute,
without downloading or processing the full ~2.4M-edge graph.

Strategy: sample a capped number of nodes per node type (all 137 Disease
nodes are kept, since that's already small and is the full candidate
pool for the prediction task), then keep every edge, in every one of the
24 relations, whose BOTH endpoints survived sampling. Node indices are
remapped to a dense 0..N-1 range for the toy graph. The same two
train/test split rules used for the full graph (random 80/20; cold-start
by held-out compound) are re-applied to whatever CtD edges survive in
the toy subgraph.

Run:
    python src/make_toy_data.py --in data/processed/graph.pt \
        --out toy_data/graph_toy.pt
"""

import argparse
import random

import torch

from data_prep import make_splits

NODES_PER_TYPE_CAP = {
    "Disease": None,       # keep all (there are only 137)
    "Compound": 150,
    "Gene": 400,
    "Anatomy": 40,
    "Symptom": 60,
    "SideEffect": 150,
    "BiologicalProcess": 150,
    "CellularComponent": 60,
    "MolecularFunction": 100,
    "Pathway": 100,
    "PharmacologicClass": 60,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/graph.pt")
    ap.add_argument("--out", default="toy_data/graph_toy.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    payload = torch.load(args.inp, weights_only=False)
    kind_list = payload["kind_list"]
    kind_to_id = {k: i for i, k in enumerate(kind_list)}
    node_kind_idx = payload["node_kind_idx"]

    # --- Make sure every Compound that has a CtD edge is a CANDIDATE for
    # sampling preference, so the toy graph actually has CtD edges to
    # predict (a pure uniform sample would mostly miss the ~385 compounds
    # that treat something, out of 1552 total compounds). ---
    ctd_src_full = payload["ctd_src"].tolist()
    compounds_with_ctd = sorted(set(ctd_src_full))

    keep_idx = set()
    for kind in kind_list:
        cap = NODES_PER_TYPE_CAP.get(kind, 50)
        all_of_kind = torch.nonzero(node_kind_idx == kind_to_id[kind], as_tuple=True)[0].tolist()
        if cap is None or cap >= len(all_of_kind):
            keep_idx.update(all_of_kind)
            continue
        if kind == "Compound":
            # Prioritise compounds that have at least one CtD edge, so the
            # toy subgraph keeps the prediction task non-trivial.
            preferred = [i for i in all_of_kind if i in set(compounds_with_ctd)]
            rng.shuffle(preferred)
            chosen = preferred[: min(cap, len(preferred))]
            remaining_cap = cap - len(chosen)
            if remaining_cap > 0:
                rest = [i for i in all_of_kind if i not in set(chosen)]
                rng.shuffle(rest)
                chosen += rest[:remaining_cap]
            keep_idx.update(chosen)
        else:
            rng.shuffle(all_of_kind)
            keep_idx.update(all_of_kind[:cap])

    keep_idx = sorted(keep_idx)
    old_to_new = {old: new for new, old in enumerate(keep_idx)}
    print(f"Toy subgraph: kept {len(keep_idx)} / {payload['num_nodes']} nodes")

    new_node_kind_idx = node_kind_idx[torch.tensor(keep_idx, dtype=torch.long)]
    new_node_name = [payload["node_name"][i] for i in keep_idx]

    new_rel_edges = {}
    for rel, (src, dst) in payload["rel_edges"].items():
        src_l, dst_l = src.tolist(), dst.tolist()
        new_s, new_d = [], []
        for s, d in zip(src_l, dst_l):
            if s in old_to_new and d in old_to_new:
                new_s.append(old_to_new[s])
                new_d.append(old_to_new[d])
        new_rel_edges[rel] = (torch.tensor(new_s, dtype=torch.long),
                               torch.tensor(new_d, dtype=torch.long))
        print(f"  {rel:6s}: {len(new_s)} edges (was {len(src_l)})")

    ctd_src_new, ctd_dst_new = new_rel_edges["CtD"]
    print(f"Toy CtD edges: {ctd_src_new.numel()}")

    new_kind_list = kind_list  # keep the same label set/order
    splits = make_splits(ctd_src_new.tolist(), ctd_dst_new.tolist(),
                          [new_kind_list[k] for k in new_node_kind_idx.tolist()],
                          seed=args.seed)
    print(f"  random:     train={len(splits['random']['train'])} test={len(splits['random']['test'])}")
    print(f"  cold_start: train={len(splits['cold_start']['train'])} "
          f"test={len(splits['cold_start']['test'])} "
          f"heldout_compounds={len(splits['cold_start']['heldout_compounds'])}")

    fp_matrix_full = payload["fingerprint_matrix"]
    fp_mask_full = payload["fingerprint_mask"]
    keep_t = torch.tensor(keep_idx, dtype=torch.long)
    new_fp_matrix = fp_matrix_full[keep_t]
    new_fp_mask = fp_mask_full[keep_t]
    print(f"Toy compounds with usable fingerprint: {int(new_fp_mask.sum())}")

    toy_payload = {
        "num_nodes": len(keep_idx),
        "node_kind_idx": new_node_kind_idx,
        "kind_list": new_kind_list,
        "node_name": new_node_name,
        "relation_names": payload["relation_names"],
        "rel_edges": new_rel_edges,
        "ctd_src": ctd_src_new,
        "ctd_dst": ctd_dst_new,
        "splits": splits,
        "fingerprint_matrix": new_fp_matrix,
        "fingerprint_mask": new_fp_mask,
        "fingerprint_bits": payload["fingerprint_bits"],
    }
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(toy_payload, args.out)
    print(f"Saved toy graph to {args.out}")


if __name__ == "__main__":
    main()
