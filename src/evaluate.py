"""
evaluate.py

Link-prediction evaluation for the Compound-treats-Disease (CtD) task.

For every positive test edge (c, d), we rank the true disease d against
EVERY disease node in the graph that is not already a known CtD target of
c (across train AND test) -- i.e. full, exhaustive, filtered ranking, not
a sampled handful of negatives. Hetionet only has 137 Disease nodes total,
so this is cheap and removes the sampling variance an earlier, 50-negative
version of this file had. We report:

  - AUROC  : ranking quality over the combined positive+negative pool
  - AUPRC  : precision-recall area, more informative than AUROC when
             negatives vastly outnumber positives (true here: most
             compound-disease pairs are not a treatment)
  - Hits@k : for k in {1, 3, 10}, the fraction of positive test edges whose
             disease is ranked in the top-k among all eligible candidates

This matches the cold-start discovery framing taught in Week 4 Lecture 2:
we are always asking "of all the diseases this compound could plausibly
treat, how high does the model rank the one we know is true?"
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


@torch.no_grad()
def evaluate_link_prediction(node_embeddings, decoder, test_src, test_dst,
                              disease_idx_pool, all_true_dst_by_src, seed=0,
                              compound_degree=None):
    """
    node_embeddings: (num_nodes, dim) tensor, already computed by one
        encoder forward pass (full-batch, so we evaluate all test edges
        against the same fixed embeddings).
    decoder: DistMultDecoder.
    test_src, test_dst: 1-D LongTensors of positive test edges.
    disease_idx_pool: list of all Disease node indices in the graph
        (the full candidate pool -- every disease not already known true
        for this compound is used as a negative; nothing is sampled).
    all_true_dst_by_src: dict {compound_idx: set(disease_idx)} across BOTH
        train and test, so a candidate that is secretly another true
        treatment is excluded rather than incorrectly penalised.
    seed: unused (kept for call-site compatibility / reproducibility
        bookkeeping); ranking is now exhaustive, so there is nothing left
        to randomize.
    """
    disease_pool_t = torch.tensor(disease_idx_pool, dtype=torch.long)
    y_true, y_score = [], []
    hits_at = {1: 0, 3: 0, 10: 0}
    n_eval = 0
    per_edge = []

    for i in range(test_src.numel()):
        c, true_d = int(test_src[i]), int(test_dst[i])
        known_true = all_true_dst_by_src.get(c, {true_d})

        candidates = [true_d] + [d for d in disease_idx_pool
                                  if d != true_d and d not in known_true]

        h_c = node_embeddings[c].unsqueeze(0).expand(len(candidates), -1)
        h_d = node_embeddings[torch.tensor(candidates, dtype=torch.long)]
        scores = decoder.score(h_c, h_d).cpu().numpy()

        labels = np.zeros(len(candidates))
        labels[0] = 1
        y_true.extend(labels.tolist())
        y_score.extend(scores.tolist())

        rank = int((-scores).argsort().tolist().index(0)) + 1  # 1-indexed rank of the true edge
        for k in hits_at:
            if rank <= k:
                hits_at[k] += 1
        n_eval += 1

        per_edge.append({
            "compound": c,
            "disease": true_d,
            "degree": float(compound_degree[c]) if compound_degree is not None else None,
            "rank": rank,
            "n_candidates": len(candidates),
        })

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    hits = {f"hits@{k}": v / n_eval for k, v in hits_at.items()}

    metrics = {"auroc": auroc, "auprc": auprc, "n_eval": n_eval,
               "avg_candidates": len(y_true) / n_eval, **hits}
    return metrics, per_edge
