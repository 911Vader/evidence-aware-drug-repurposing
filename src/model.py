"""
model.py

A Relational Graph Convolutional encoder plus a DistMult link-scoring
decoder, implemented from the exact formulas taught in this course
(Week 3 Lecture 2, "R-GCN message passing" and Week 4 Lecture 1's
TxGNN slide):

    h_v^(l+1) = ReLU( W_0 h_v^(l)
                      + sum_r mean_{u in N_r(v)} W_r h_u^(l) )

    score(h, r, t) = sigmoid( sum_k h_k * w_r_k * t_k )          (DistMult)

For every relation in the graph we also add its inverse (so information
can flow in both directions across a directed edge, standard R-GCN
practice), each with its own learned weight matrix.

CORRECTNESS NOTE (fixed after initial submission): the formula requires
the mean over neighbors to be computed SEPARATELY for each relation type,
and the resulting per-relation means to be SUMMED across relations. The
first version of this file instead pooled every relation's messages and
neighbor counts into one shared accumulator and divided once at the end
-- which computes a single global mean over the union of all neighbors,
not a sum of per-relation means. That silently lets high-degree relations
(e.g. Gene-participates-BiologicalProcess, ~560K edges) drown out rare
ones (e.g. Compound-treats-Disease, 755 edges) in the normalization.
RelationalGraphConvLayer.forward below now accumulates each relation's
sum/count separately and sums the resulting per-relation means, matching
the slide formula exactly.

This is a from-scratch implementation using plain PyTorch tensor ops
(scatter via index_add_), not a graph-library black box, so every step
is inspectable and matches the slide-level formula directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, relation_names):
        super().__init__()
        self.relation_names = relation_names
        self.out_dim = out_dim
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)
        self.W_rel = nn.ModuleDict({
            name: nn.Linear(in_dim, out_dim, bias=False) for name in relation_names
        })

    def forward(self, x, rel_edge_index):
        """x: (num_nodes, in_dim). rel_edge_index: {relation_name: (src, dst)
        LongTensors}, already including inverse relations.

        h_v' = ReLU( W_0 x_v + sum_r  [ (sum_{u in N_r(v)} W_r x_u) / |N_r(v)| ] )

        Each relation's neighbor mean is computed with ITS OWN denominator
        (|N_r(v)|, the count of v's neighbors under relation r only), and
        the per-relation means are summed -- not a single mean over the
        pooled neighbor set across all relations.
        """
        num_nodes = x.size(0)
        total = torch.zeros(num_nodes, self.out_dim, device=x.device)

        for name, (src, dst) in rel_edge_index.items():
            if src.numel() == 0:
                continue
            messages = self.W_rel[name](x[src])  # (num_edges, out_dim)

            rel_sum = torch.zeros(num_nodes, self.out_dim, device=x.device)
            rel_count = torch.zeros(num_nodes, 1, device=x.device)
            rel_sum.index_add_(0, dst, messages)
            rel_count.index_add_(0, dst, torch.ones(src.numel(), 1, device=x.device))

            rel_mean = rel_sum / rel_count.clamp(min=1.0)  # 0 where count==0, correctly
            total = total + rel_mean

        total = total + self.W_self(x)
        return F.relu(total)


class EvidenceGate(nn.Module):
    """Learns how much a compound's node feature should come from the
    knowledge-graph-learned embedding vs. its molecular fingerprint,
    as a function of how much non-treatment graph evidence exists for
    that compound: g_c = sigmoid(a * log(1 + degree_c) + b).

    g_c -> 1 means "trust the graph embedding"; g_c -> 0 means "trust
    chemistry". a and b are the only two learned parameters of the gate
    itself (the two representations it blends are learned normally).
    """

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.3))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, degree):
        """degree: (num_nodes,) non-negative float tensor (0 for non-compound
        nodes / compounds with no non-CtD edges)."""
        return torch.sigmoid(self.a * torch.log1p(degree) + self.b)


class RelationalGNNEncoder(nn.Module):
    """fusion_mode:
        None / "none"        -- baseline, learned ID embedding only.
        "concat"              -- concatenate a projected fingerprint onto
                                  the Compound embedding (used for both the
                                  real-fingerprint and shuffled-fingerprint-
                                  control variants; which fingerprint matrix
                                  is passed in at forward() time decides which
                                  one this is).
        "gated"                -- evidence-aware adaptive blend: a learned
                                  gate g_c (function of non-CtD graph degree)
                                  convexly combines the embedding and the
                                  projected fingerprint, instead of
                                  concatenating them.
    """

    def __init__(self, num_nodes, relation_names, embed_dim=64, hidden_dim=64,
                 num_layers=2, fingerprint_bits=None, fingerprint_proj_dim=None,
                 fusion_mode=None):
        super().__init__()
        self.node_embed = nn.Embedding(num_nodes, embed_dim)
        nn.init.xavier_uniform_(self.node_embed.weight)

        self.fusion_mode = fusion_mode if fingerprint_bits is not None else None

        if self.fusion_mode == "concat":
            proj_dim = fingerprint_proj_dim or embed_dim
            self.fp_proj = nn.Linear(fingerprint_bits, proj_dim)
            in_dim = embed_dim + proj_dim
        elif self.fusion_mode == "gated":
            # gated blend requires both representations at the same width
            self.fp_proj = nn.Linear(fingerprint_bits, embed_dim)
            self.gate = EvidenceGate()
            in_dim = embed_dim
        else:
            in_dim = embed_dim

        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            RelationalGraphConvLayer(dims[i], dims[i + 1], relation_names)
            for i in range(num_layers)
        ])

    def forward(self, rel_edge_index, fingerprint_matrix=None, fingerprint_mask=None,
                compound_degree=None):
        all_idx = torch.arange(self.node_embed.num_embeddings,
                                device=self.node_embed.weight.device)
        x = self.node_embed(all_idx)

        if self.fusion_mode == "concat":
            fp_feat = self.fp_proj(fingerprint_matrix)
            fp_feat = fp_feat * fingerprint_mask.unsqueeze(1).float()
            x = torch.cat([x, fp_feat], dim=1)
        elif self.fusion_mode == "gated":
            fp_feat = self.fp_proj(fingerprint_matrix)
            fp_feat = fp_feat * fingerprint_mask.unsqueeze(1).float()
            g = self.gate(compound_degree).unsqueeze(1)  # (num_nodes, 1)
            # Only blend where a fingerprint actually exists; elsewhere g=1
            # (pure embedding) regardless of the gate's output, since there
            # is no chemistry to fall back on.
            g = torch.where(fingerprint_mask.unsqueeze(1), g, torch.ones_like(g))
            x = g * x + (1 - g) * fp_feat

        for layer in self.layers:
            x = layer(x, rel_edge_index)
        return x


class DistMultDecoder(nn.Module):
    """One diagonal relation vector, used for the single target relation
    (Compound-treats-Disease) this project predicts."""

    def __init__(self, dim):
        super().__init__()
        self.rel_vec = nn.Parameter(torch.ones(dim))
        nn.init.uniform_(self.rel_vec, 0.9, 1.1)

    def score(self, h_src, h_dst):
        return (h_src * self.rel_vec * h_dst).sum(dim=-1)


def build_bidirectional_edge_index(rel_edges, relation_names):
    """Adds an inverse relation 'REL_inv' for every relation so the
    encoder can propagate information in both directions."""
    full = {}
    for name in relation_names:
        src, dst = rel_edges[name]
        full[name] = (src, dst)
        full[name + "_inv"] = (dst, src)
    return full


def all_relation_names_with_inverse(relation_names):
    out = []
    for name in relation_names:
        out.append(name)
        out.append(name + "_inv")
    return out


def compute_non_target_degree(rel_edges, num_nodes, target_relation="CtD"):
    """Non-target-relation graph degree for every node: how many edges (as
    source or destination, across every relation EXCEPT the target one)
    touch this node. Used both as the EvidenceGate's input feature and, at
    analysis time, to stratify compounds into low/medium/high graph-
    evidence groups."""
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    for name, (src, dst) in rel_edges.items():
        if name == target_relation:
            continue
        if src.numel() == 0:
            continue
        degree.index_add_(0, src, torch.ones(src.numel()))
        degree.index_add_(0, dst, torch.ones(dst.numel()))
    return degree
