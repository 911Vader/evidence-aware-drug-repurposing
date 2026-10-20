"""
data_prep.py

Loads Hetionet v1.0 (Himmelstein et al., eLife 2017 — the same knowledge
graph named in this course's Week 4 lecture on network-based drug
repurposing) from the raw TSV/SIF files, builds a node-indexed
multi-relational graph, computes Morgan fingerprints for every Compound
node from its DrugBank InChI (via the companion dhimmel/drugbank dataset),
and builds two train/test splits for the Compound-treats-Disease (CtD)
link-prediction task:

  1. "random": a random 80/20 split of CtD edges (standard transductive
     link-prediction evaluation).
  2. "cold_start": a split by COMPOUND identity — a held-out 20% of
     compounds have ALL their CtD edges moved to the test set, so the
     model has never seen ANY treats-relationship for those compounds
     during training. This is exactly the "drug held out" condition this
     course's Week 4 Lecture 2 slides define ("at least one constituent
     drug unseen... the split defines the discovery claim"). Held-out
     compounds keep their other edges (binds, causes-side-effect,
     upregulates/downregulates, resembles, pharmacologic-class) in the
     graph, since cold-start here means the treats *label* is unseen, not
     that the compound is erased from the knowledge graph — this is the
     realistic zero-shot setting TxGNN targets.

Run:
    python src/data_prep.py --data-dir data/raw --out data/processed/graph.pt
"""

import argparse
import csv
import gzip
import os
import random

import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

RELATION_ABBREV_TO_NAME = {
    "AdG": "Anatomy-downregulates-Gene", "AeG": "Anatomy-expresses-Gene",
    "AuG": "Anatomy-upregulates-Gene", "CbG": "Compound-binds-Gene",
    "CcSE": "Compound-causes-SideEffect", "CdG": "Compound-downregulates-Gene",
    "CpD": "Compound-palliates-Disease", "CrC": "Compound-resembles-Compound",
    "CtD": "Compound-treats-Disease", "CuG": "Compound-upregulates-Gene",
    "DaG": "Disease-associates-Gene", "DdG": "Disease-downregulates-Gene",
    "DlA": "Disease-localizes-Anatomy", "DpS": "Disease-presents-Symptom",
    "DrD": "Disease-resembles-Disease", "DuG": "Disease-upregulates-Gene",
    "GcG": "Gene-covaries-Gene", "GiG": "Gene-interacts-Gene",
    "GpBP": "Gene-participates-BiologicalProcess",
    "GpCC": "Gene-participates-CellularComponent",
    "GpMF": "Gene-participates-MolecularFunction",
    "GpPW": "Gene-participates-Pathway", "Gr>G": "Gene-regulates-Gene",
    "PCiC": "PharmacologicClass-includes-Compound",
}
TARGET_RELATION = "CtD"
FINGERPRINT_BITS = 2048
SEED = 0


def load_nodes(nodes_path):
    node_id_to_idx, node_kind, node_name = {}, [], []
    with open(nodes_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            node_id_to_idx[row["id"]] = len(node_id_to_idx)
            node_kind.append(row["kind"])
            node_name.append(row["name"])
    return node_id_to_idx, node_kind, node_name


def load_edges(edges_gz_path, node_id_to_idx):
    """Returns {relation_abbrev: (src_idx list, dst_idx list)}."""
    rel_edges = {r: ([], []) for r in RELATION_ABBREV_TO_NAME}
    with gzip.open(edges_gz_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            src, rel, dst = row["source"], row["metaedge"], row["target"]
            if rel not in rel_edges:
                raise ValueError(f"Unknown relation abbreviation: {rel}")
            rel_edges[rel][0].append(node_id_to_idx[src])
            rel_edges[rel][1].append(node_id_to_idx[dst])
    return rel_edges


def load_compound_fingerprints(drugbank_tsv_path, node_id_to_idx, node_kind):
    """Returns a dict {node_idx: fingerprint tensor (FINGERPRINT_BITS,)} for
    every Compound node whose DrugBank InChI parses successfully."""
    inchi_by_drugbank_id = {}
    with open(drugbank_tsv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("inchi"):
                inchi_by_drugbank_id[row["drugbank_id"]] = row["inchi"]

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FINGERPRINT_BITS)
    fps = {}
    failed = []
    for node_id, idx in node_id_to_idx.items():
        if node_kind[idx] != "Compound":
            continue
        drugbank_id = node_id.split("::")[1]
        inchi = inchi_by_drugbank_id.get(drugbank_id)
        if inchi is None:
            failed.append(drugbank_id)
            continue
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            failed.append(drugbank_id)
            continue
        bv = fpgen.GetFingerprintAsNumPy(mol)
        fps[idx] = torch.tensor(bv, dtype=torch.float32)
    return fps, failed


def make_splits(ctd_src, ctd_dst, node_kind, seed=SEED):
    rng = random.Random(seed)
    n = len(ctd_src)
    idx = list(range(n))
    rng.shuffle(idx)

    # --- random split ---
    n_test = max(1, round(n * 0.2))
    random_test_idx = set(idx[:n_test])
    random_train_idx = set(idx[n_test:])

    # --- cold-start split: hold out 20% of DISTINCT compounds that appear
    # in CtD edges; ALL their CtD edges go to test. ---
    distinct_compounds = sorted(set(ctd_src))
    rng.shuffle(distinct_compounds)
    n_heldout_compounds = max(1, round(len(distinct_compounds) * 0.2))
    heldout_compounds = set(distinct_compounds[:n_heldout_compounds])
    cold_test_idx = {i for i in range(n) if ctd_src[i] in heldout_compounds}
    cold_train_idx = set(range(n)) - cold_test_idx

    return {
        "random": {"train": sorted(random_train_idx), "test": sorted(random_test_idx)},
        "cold_start": {
            "train": sorted(cold_train_idx),
            "test": sorted(cold_test_idx),
            "heldout_compounds": sorted(heldout_compounds),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed/graph.pt")
    args = ap.parse_args()

    nodes_path = os.path.join(args.data_dir, "hetionet-v1.0-nodes.tsv")
    edges_path = os.path.join(args.data_dir, "hetionet-v1.0-edges.sif.gz")
    drugbank_path = os.path.join(args.data_dir, "drugbank_dhimmel.tsv")

    print("Loading nodes...")
    node_id_to_idx, node_kind, node_name = load_nodes(nodes_path)
    print(f"  {len(node_id_to_idx)} nodes, kinds: "
          f"{sorted(set(node_kind))}")

    print("Loading edges...")
    rel_edges = load_edges(edges_path, node_id_to_idx)
    for r, (s, d) in rel_edges.items():
        print(f"  {r:6s} ({RELATION_ABBREV_TO_NAME[r]}): {len(s)} edges")

    print("Computing Morgan fingerprints for Compound nodes...")
    fps, failed = load_compound_fingerprints(drugbank_path, node_id_to_idx, node_kind)
    print(f"  {len(fps)} compounds with a usable fingerprint, {len(failed)} failed")

    ctd_src, ctd_dst = rel_edges[TARGET_RELATION]
    print(f"Building splits on {len(ctd_src)} {TARGET_RELATION} edges...")
    splits = make_splits(ctd_src, ctd_dst, node_kind)
    print(f"  random:     train={len(splits['random']['train'])} "
          f"test={len(splits['random']['test'])}")
    print(f"  cold_start: train={len(splits['cold_start']['train'])} "
          f"test={len(splits['cold_start']['test'])} "
          f"heldout_compounds={len(splits['cold_start']['heldout_compounds'])}")

    kind_list = sorted(set(node_kind))
    kind_to_id = {k: i for i, k in enumerate(kind_list)}
    node_kind_idx = torch.tensor([kind_to_id[k] for k in node_kind], dtype=torch.long)

    fp_matrix = torch.zeros(len(node_id_to_idx), FINGERPRINT_BITS, dtype=torch.float32)
    fp_mask = torch.zeros(len(node_id_to_idx), dtype=torch.bool)
    for idx, vec in fps.items():
        fp_matrix[idx] = vec
        fp_mask[idx] = True

    payload = {
        "num_nodes": len(node_id_to_idx),
        "node_kind_idx": node_kind_idx,
        "kind_list": kind_list,
        "node_id_to_idx": node_id_to_idx,
        "node_name": node_name,
        "relation_names": list(RELATION_ABBREV_TO_NAME.keys()),
        "rel_edges": {r: (torch.tensor(s, dtype=torch.long), torch.tensor(d, dtype=torch.long))
                      for r, (s, d) in rel_edges.items()},
        "ctd_src": torch.tensor(ctd_src, dtype=torch.long),
        "ctd_dst": torch.tensor(ctd_dst, dtype=torch.long),
        "splits": splits,
        "fingerprint_matrix": fp_matrix,
        "fingerprint_mask": fp_mask,
        "fingerprint_bits": FINGERPRINT_BITS,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(payload, args.out)
    print(f"Saved processed graph to {args.out}")


if __name__ == "__main__":
    main()
