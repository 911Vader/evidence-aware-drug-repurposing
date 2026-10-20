const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, BorderStyle, AlignmentType, ImageRun, ShadingType, PageBreak,
} = require("docx");

const US_LETTER = { width: 12240, height: 15840 };

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
}
function pRich(children, opts = {}) {
  return new Paragraph({ children, spacing: { after: 160 }, ...opts });
}
function bullet(text, level = 0) {
  return new Paragraph({ text, bullet: { level }, spacing: { after: 80 } });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 20, color: "555555" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
  });
}
function image(path, width, height) {
  const data = fs.readFileSync(path);
  return new Paragraph({
    children: [new ImageRun({ data, transformation: { width, height }, type: "png" })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
  });
}
function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { fill: "2F5496", type: ShadingType.CLEAR, color: "auto" } : (opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR, color: "auto" } : undefined),
    children: [new Paragraph({
      children: [new TextRun({ text, bold: !!opts.header || !!opts.bold, color: opts.header ? "FFFFFF" : (opts.color || "000000"), size: opts.size || 19 })],
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
    })],
    verticalAlign: "center",
  });
}

function datasetTable() {
  const colWidths = [3800, 2400, 3800];
  const header = new TableRow({
    children: [
      cell("Quantity", { header: true, width: colWidths[0] }),
      cell("Value", { header: true, width: colWidths[1], center: true }),
      cell("Notes", { header: true, width: colWidths[2] }),
    ],
    tableHeader: true,
  });
  const data = [
    ["Total nodes", "47,031", "11 node types"],
    ["Total edges", "2,250,197", "24 relation types (metaedges)"],
    ["Compound nodes", "1,552", "all DrugBank-sourced"],
    ["Disease nodes", "137", "MeSH/DOID-sourced"],
    ["Compound-treats-Disease (CtD) edges", "755", "the prediction target"],
    ["Distinct compounds in CtD", "387", "of 1,552 total compounds"],
    ["Compounds with a usable Morgan fingerprint", "1,552 / 1,552 (100%)", "via DrugBank InChI → RDKit"],
  ];
  const rows = data.map((r) => new TableRow({
    children: [cell(r[0], { width: colWidths[0] }), cell(r[1], { width: colWidths[1], center: true }), cell(r[2], { width: colWidths[2], size: 18 })],
  }));
  return new Table({ width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: colWidths, rows: [header, ...rows] });
}

// Raw [mean, std] per split/variant/metric. Colors in resultsTable() are
// DERIVED from these numbers (compared against that split's baseline),
// never hand-set -- so the table can't silently drift out of sync with a
// claim in the prose the way the hand-colored version once did.
const RESULTS = {
  random: {
    baseline:           { auroc: [0.840, 0.011], auprc: [0.135, 0.010], h1: [0.212, 0.038], h3: [0.362, 0.033], h10: [0.609, 0.066] },
    fingerprint:         { auroc: [0.859, 0.020], auprc: [0.118, 0.035], h1: [0.170, 0.023], h3: [0.362, 0.038], h10: [0.640, 0.011] },
    random_fingerprint:  { auroc: [0.837, 0.032], auprc: [0.105, 0.009], h1: [0.196, 0.033], h3: [0.338, 0.034], h10: [0.616, 0.046] },
    adaptive:            { auroc: [0.885, 0.007], auprc: [0.150, 0.007], h1: [0.265, 0.025], h3: [0.468, 0.042], h10: [0.762, 0.047] },
  },
  cold_start: {
    baseline:           { auroc: [0.837, 0.024], auprc: [0.076, 0.013], h1: [0.162, 0.014], h3: [0.356, 0.045], h10: [0.650, 0.046] },
    fingerprint:         { auroc: [0.801, 0.044], auprc: [0.068, 0.033], h1: [0.092, 0.040], h3: [0.237, 0.100], h10: [0.508, 0.117] },
    random_fingerprint:  { auroc: [0.811, 0.028], auprc: [0.062, 0.021], h1: [0.106, 0.086], h3: [0.248, 0.111], h10: [0.575, 0.106] },
    adaptive:            { auroc: [0.846, 0.030], auprc: [0.102, 0.020], h1: [0.150, 0.044], h3: [0.382, 0.099], h10: [0.683, 0.027] },
  },
};
const METRICS5 = ["auroc", "auprc", "h1", "h3", "h10"];
const VARIANT_LABEL = { baseline: "baseline", fingerprint: "fingerprint", random_fingerprint: "random_fingerprint (control)", adaptive: "adaptive" };
const SPLIT_LABEL = { random: "Random", cold_start: "Cold-start" };

// Additional no-message-passing baseline: identical codebase, data, split,
// negative sampling and exhaustive-ranking evaluation as every number in
// Table 2 above (src/train.py --variant baseline --num-layers 0) -- the
// R-GCN's two RelationalGraphConvLayers are simply not applied, so the
// decoder scores the raw, directly-learned node embeddings. 6 runs (3
// seeds x 2 splits), aggregated the same way as Table 2.
const NOGNN_RESULTS = {
  random:     { auroc: [0.835, 0.003], auprc: [0.143, 0.008], h1: [0.214, 0.020], h3: [0.336, 0.003], h10: [0.525, 0.008] },
  cold_start: { auroc: [0.476, 0.029], auprc: [0.010, 0.004], h1: [0.025, 0.014], h3: [0.060, 0.024], h10: [0.163, 0.018] },
};

function noGnnTable() {
  const colWidths = [2300, 1420, 1420, 1420, 1420, 1420];
  const header = new TableRow({
    children: [
      cell("Run (split × variant)", { header: true, width: colWidths[0] }),
      cell("AUROC", { header: true, width: colWidths[1], center: true }),
      cell("AUPRC", { header: true, width: colWidths[2], center: true }),
      cell("Hits@1", { header: true, width: colWidths[3], center: true }),
      cell("Hits@3", { header: true, width: colWidths[4], center: true }),
      cell("Hits@10", { header: true, width: colWidths[5], center: true }),
    ],
    tableHeader: true,
  });
  const rows = [];
  for (const split of ["random", "cold_start"]) {
    const rgcn = RESULTS[split].baseline;
    const nognn = NOGNN_RESULTS[split];
    const mk = (label, rec, dim) => new TableRow({
      children: [cell(label, { width: colWidths[0], fill: dim ? "F2EFE6" : undefined })].concat(
        METRICS5.map((m, i) => cell(fmtMS(rec[m]), { width: colWidths[i + 1], center: true, fill: dim ? "F2EFE6" : undefined }))
      ),
    });
    rows.push(mk(`${SPLIT_LABEL[split]} · baseline (R-GCN, Table 2)`, rgcn, false));
    rows.push(mk(`${SPLIT_LABEL[split]} · no-GNN (embedding only)`, nognn, true));
  }
  return new Table({ width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: colWidths, rows: [header, ...rows] });
}

function fmtMS([m, s]) { return `${m.toFixed(3)} ± ${s.toFixed(3)}`; }

function resultsTable() {
  const colWidths = [2300, 1420, 1420, 1420, 1420, 1420];
  const header = new TableRow({
    children: [
      cell("Run (split × variant)", { header: true, width: colWidths[0] }),
      cell("AUROC", { header: true, width: colWidths[1], center: true }),
      cell("AUPRC", { header: true, width: colWidths[2], center: true }),
      cell("Hits@1", { header: true, width: colWidths[3], center: true }),
      cell("Hits@3", { header: true, width: colWidths[4], center: true }),
      cell("Hits@10", { header: true, width: colWidths[5], center: true }),
    ],
    tableHeader: true,
  });
  const rows = [];
  for (const split of ["random", "cold_start"]) {
    const base = RESULTS[split].baseline;
    for (const variant of ["baseline", "fingerprint", "random_fingerprint", "adaptive"]) {
      const rec = RESULTS[split][variant];
      const isAdaptive = variant === "adaptive";
      const rowFill = isAdaptive ? "EAF5F3" : undefined;
      const cells = [cell(`${SPLIT_LABEL[split]} · ${VARIANT_LABEL[variant]}`, { width: colWidths[0], bold: isAdaptive, fill: rowFill })];
      METRICS5.forEach((m, i) => {
        const underperforms = variant !== "baseline" && rec[m][0] < base[m][0];
        cells.push(cell(fmtMS(rec[m]), {
          width: colWidths[i + 1], center: true, bold: isAdaptive, fill: rowFill,
          color: underperforms ? "C0392B" : (isAdaptive ? "1F7A6C" : "000000"),
        }));
      });
      rows.push(new TableRow({ children: cells }));
    }
  }
  return new Table({ width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: colWidths, rows: [header, ...rows] });
}

function stratTable() {
  const colWidths = [2600, 1800, 2400, 2400];
  const header = new TableRow({
    children: [
      cell("Variant · evidence tertile", { header: true, width: colWidths[0] }),
      cell("n (avg)", { header: true, width: colWidths[1], center: true }),
      cell("Hits@10", { header: true, width: colWidths[2], center: true }),
      cell("Mean rank", { header: true, width: colWidths[3], center: true }),
    ],
    tableHeader: true,
  });
  const data = [
    ["baseline · low", "68", "0.569 ± 0.102", "17.19 ± 3.63"],
    ["baseline · medium", "69", "0.744 ± 0.049", "12.46 ± 3.55"],
    ["baseline · high", "67", "0.637 ± 0.043", "12.00 ± 1.03"],
    ["fingerprint · low", "68", "0.407 ± 0.084", "23.23 ± 7.66"],
    ["fingerprint · medium", "69", "0.652 ± 0.171", "14.28 ± 7.04"],
    ["fingerprint · high", "67", "0.463 ± 0.136", "17.77 ± 4.96"],
    ["random_fingerprint · low", "68", "0.461 ± 0.112", "16.75 ± 5.84"],
    ["random_fingerprint · medium", "69", "0.686 ± 0.060", "11.81 ± 2.53"],
    ["random_fingerprint · high", "67", "0.577 ± 0.157", "11.61 ± 3.93"],
    ["adaptive · low", "68", "0.583 ± 0.087", "15.47 ± 3.44", true],
    ["adaptive · medium", "69", "0.725 ± 0.031", "14.14 ± 3.56", true],
    ["adaptive · high", "67", "0.741 ± 0.037", "9.77 ± 1.29", true],
  ];
  const rows = data.map((r) => new TableRow({
    children: [
      cell(r[0], { width: colWidths[0], bold: r[4], fill: r[4] ? "FBEFE3" : undefined }),
      cell(r[1], { width: colWidths[1], center: true, fill: r[4] ? "FBEFE3" : undefined }),
      cell(r[2], { width: colWidths[2], center: true, bold: r[4], fill: r[4] ? "FBEFE3" : undefined }),
      cell(r[3], { width: colWidths[3], center: true, fill: r[4] ? "FBEFE3" : undefined }),
    ],
  }));
  return new Table({ width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: colWidths, rows: [header, ...rows] });
}

// Novelty-control ablation: adaptive (real fingerprints) vs
// adaptive_random_fingerprint (identical gated architecture, shuffled
// fingerprints). 3 seeds x 2 splits, 6 runs.
const ABLATION = {
  random: {
    adaptive:                     { auroc: [0.885, 0.007], auprc: [0.150, 0.007], h1: [0.265, 0.025], h3: [0.468, 0.042], h10: [0.762, 0.047] },
    adaptive_random_fingerprint:  { auroc: [0.873, 0.008], auprc: [0.143, 0.002], h1: [0.223, 0.033], h3: [0.413, 0.041], h10: [0.687, 0.019] },
  },
  cold_start: {
    adaptive:                     { auroc: [0.846, 0.030], auprc: [0.102, 0.020], h1: [0.150, 0.044], h3: [0.382, 0.099], h10: [0.683, 0.027] },
    adaptive_random_fingerprint:  { auroc: [0.843, 0.009], auprc: [0.092, 0.021], h1: [0.132, 0.022], h3: [0.399, 0.044], h10: [0.613, 0.061] },
  },
};
const ABLATION_LABEL = { adaptive: "adaptive (real fingerprints)", adaptive_random_fingerprint: "adaptive (shuffled fingerprints)" };

function ablationTable() {
  const colWidths = [3000, 1368, 1368, 1368, 1368, 1368];
  const header = new TableRow({
    children: [
      cell("Run (split × variant)", { header: true, width: colWidths[0] }),
      cell("AUROC", { header: true, width: colWidths[1], center: true }),
      cell("AUPRC", { header: true, width: colWidths[2], center: true }),
      cell("Hits@1", { header: true, width: colWidths[3], center: true }),
      cell("Hits@3", { header: true, width: colWidths[4], center: true }),
      cell("Hits@10", { header: true, width: colWidths[5], center: true }),
    ],
    tableHeader: true,
  });
  const rows = [];
  for (const split of ["random", "cold_start"]) {
    for (const variant of ["adaptive", "adaptive_random_fingerprint"]) {
      const rec = ABLATION[split][variant];
      const isReal = variant === "adaptive";
      const rowFill = isReal ? "EAF5F3" : undefined;
      const cells = [cell(`${SPLIT_LABEL[split]} · ${ABLATION_LABEL[variant]}`, { width: colWidths[0], bold: isReal, fill: rowFill })];
      METRICS5.forEach((m, i) => {
        cells.push(cell(fmtMS(rec[m]), { width: colWidths[i + 1], center: true, bold: isReal, fill: rowFill }));
      });
      rows.push(new TableRow({ children: cells }));
    }
  }
  return new Table({ width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: colWidths, rows: [header, ...rows] });
}

const doc = new Document({
  styles: { default: { document: { run: { font: "Calibri", size: 22 } } } },
  sections: [{
    properties: { page: { size: US_LETTER, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    footers: {
      default: new (require("docx").Footer)({
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [require("docx").PageNumber.CURRENT], size: 18, color: "888888" })] })],
      }),
    },
    children: [
      new Paragraph({
        children: [new TextRun({ text: "Evidence-Aware Multimodal Fusion for Cold-Start Drug Repurposing", bold: true, size: 34, color: "1F3864" })],
        spacing: { after: 100 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Dynamically integrating molecular structure and biomedical knowledge-graph evidence for drug–disease link prediction", italics: true, size: 24, color: "444444" })],
        spacing: { after: 240 },
      }),
      p("CBIO 7215 — Multimodal AI for Therapeutic Discovery", { bold: true }),
      p("Final Project Technical Report"),
      p("Ghaya  ·  MSc Computational Biology, MBZUAI"),
      p("October 2026"),

      h1("1. Introduction and Motivation"),
      p("The first pass of this project asked a narrow question — does concatenating a compound's Morgan fingerprint onto a knowledge-graph GNN's node embedding improve Compound-treats-Disease link prediction? That combination already exists in the literature (SumGNN fuses knowledge-graph and fingerprint channels for drug-drug interaction prediction [Yu et al., 2021]), and a 2026 study specifically investigating pharmacology knowledge graphs found that explicit chemical structure is not always necessary for, and can even underperform, topology-only repurposing models. So the narrow question is not a novel one, and answering it alone risks reporting a result the field already has reason to doubt."),
      p("This report instead asks a sharper question: not whether chemistry helps, but when it helps, and whether a model can learn that for itself. Different compounds are represented very differently in a knowledge graph like Hetionet — some have dozens of gene-interaction, side-effect, and similarity edges; others have almost none outside the relation being predicted. A model that treats every compound's graph evidence as equally trustworthy has no way to fall back on chemistry when the graph has little to say. This project proposes an evidence-aware gated fusion mechanism that lets the model learn exactly that trade-off, and tests it against two controls — a graph-only baseline and a shuffled-fingerprint ablation — under both an easy transductive split and the strict, drug-held-out cold-start split this course's Week 4 Lecture 2 argues is the one that actually supports a discovery claim. The contribution here is not the gating mechanism in the abstract — degree-conditioned gating of this kind has precedent elsewhere — but its specific application to cold-start knowledge-graph drug repurposing on Hetionet, paired with the two-control experimental design (the shuffled-fingerprint ablation above, and the architecture-isolating novelty-control ablation in Section 7) that pins down how much of its benefit is real chemistry versus the gating architecture alone."),
      p("Three corrections were made before any of the results below were trusted, and all are reported here because they bore on the scientific conclusion: a normalization bug in the R-GCN encoder (Section 3.1); a move from single-seed, 50-negative-sample evaluation to 3-seed, exhaustive (all-eligible-disease) evaluation reported as mean ± SD (Section 4) — Section 6 shows why that mattered, since some single-seed results would have supported a different, less honest story; and a one-compound edge case in the shuffled-fingerprint control's derangement, caught during review and fixed before trusting any random_fingerprint number (Section 5.1)."),

      h2("1.1 Summary of Main Novelties and Findings"),
      p("Novelty. Degree-conditioned gated fusion is not a new mechanism in the abstract — this report does not claim to have invented it. What is novel here is its specific application to cold-start knowledge-graph drug repurposing on Hetionet, paired with a two-tier control design built to rule out the two most common ways a multimodal-fusion result can be misleading: (1) a shuffled-fingerprint ablation (fingerprint vs. random_fingerprint) that tests whether any gain from adding chemistry is real chemical signal or just added model capacity (Section 5.1), and (2) an architecture-isolating novelty-control ablation (adaptive vs. adaptive_random_fingerprint) that tests whether the gate's own benefit is the gating mechanism itself or the real chemistry it happens to be given (Section 7). Section 5.2 below adds a third control for the same reason: a no-message-passing baseline that is guaranteed, by construction, to use identical data and evaluation settings to every other number in this report."),
      bullet("Finding 1 (negative/cautionary) — naive concatenation of a molecular fingerprint onto a KG-GNN's node embedding is not reliably beneficial, and under the strict cold-start split it underperforms a graph-only model on every metric (Section 5.1). This holds for both the real-fingerprint and shuffled-fingerprint variants, which are numerically close to each other in cold-start — so the degradation is attributable to the static fusion mechanism itself, not to being given the wrong chemistry."),
      bullet("Finding 2 (positive) — an evidence-aware gate that blends embedding and chemistry as a function of a compound's non-target graph degree recovers and exceeds graph-only performance overall: strongest variant on AUROC, AUPRC, Hits@3 and Hits@10 in both splits, with tighter seed-to-seed variance than the static-fusion variants (Section 5.1). It is not uniformly ahead on every single metric — baseline is marginally higher on cold-start Hits@1 (0.162 vs. 0.150) — and that exception is reported rather than omitted."),
      bullet("Finding 3 (negative/surprising) — the gate's advantage is concentrated in the medium- and high-graph-evidence tertiles, not the low-evidence tertile the original hypothesis targeted (Section 6). The data does not support \"chemistry rescues low-evidence drugs\"; it supports a narrower claim about where gating helps."),
      bullet("Finding 4 (mixed/layered) — most of the gate's benefit over static fusion is architectural: even fed deliberately scrambled fingerprints, the gated architecture still beats both static-fusion variants on every metric in both splits (Section 7). Real chemistry adds a smaller, metric-dependent increment on top of that, visible mainly in top-k retrieval (Hits@10) and not reliably in AUROC/AUPRC at this sample size."),
      bullet("Finding 5 (methodological) — two real implementation bugs were found and fixed before any result was trusted (R-GCN per-relation normalization, Section 3.1; a one-compound derangement edge case in the shuffled-fingerprint control, Section 5.1), and moving from single-seed, 50-negative-sample evaluation to 3-seed, exhaustive evaluation changed which variant looked best (Section 4, Section 6). Mean ± SD over 3 seeds is reported throughout rather than a single run."),

      h1("2. Dataset"),
      p("Hetionet v1.0 (Himmelstein et al., eLife 2017) is the heterogeneous biomedical knowledge graph used throughout — the same KG named in this course's Week 4 network-medicine lecture, and the graph the original Project Rephetio repurposing study was built on."),
      datasetTable(),
      caption("Table 1. Hetionet v1.0 graph statistics and the Compound-treats-Disease target relation."),
      p("A 2048-bit Morgan (radius-2) fingerprint was computed with RDKit from each Compound's DrugBank InChI (via the dhimmel/drugbank companion dataset). All 1,552 Hetionet compounds matched — 100% coverage, verified before committing to any fingerprint-based design."),

      h1("3. Method"),
      h2("3.1 Encoder: relational graph convolution (R-GCN) — and a bug that was found and fixed"),
      p("Each layer follows the message-passing rule from Week 3 Lecture 2: every relation r has its own learned weight matrix W_r; a node's incoming messages are averaged WITHIN each relation type, and the resulting per-relation means are summed, then passed through a self-loop term and ReLU:"),
      pRich([new TextRun({ text: "h_v' = ReLU( W0·h_v + Σ_r  [ mean_{u in N_r(v)} Wr·h_u ] )", italics: true, font: "Consolas" })], { alignment: AlignmentType.CENTER }),
      p("The first implementation of this layer pooled every relation's messages and neighbor counts into ONE shared accumulator and divided once at the end — which computes a single global mean over the union of all neighbors regardless of relation, not a sum of per-relation means. Because Hetionet's relations range from 390 edges (Compound-palliates-Disease) to 559,504 (Gene-participates-BiologicalProcess), that bug let common relations silently dominate the normalization and under-weight rare ones, including the target relation itself. This was corrected in src/model.py's RelationalGraphConvLayer: each relation now accumulates its own sum and count before the per-relation means are summed. All results in this report use the corrected layer. For every one of Hetionet's 24 relations an inverse relation is added, for 48 relations total, each with its own weight matrix."),
      h2("3.2 Decoder: DistMult"),
      p("Link scores use the DistMult decoder taught in Week 4 Lecture 1's TxGNN slide, specialized to the CtD target relation:"),
      pRich([new TextRun({ text: "score(c, d) = Σ_k  h_c,k · w_k · h_d,k", italics: true, font: "Consolas" })], { alignment: AlignmentType.CENTER }),
      h2("3.3 Four variants"),
      p("adaptive, below, is this project's proposed variant; the other three are a graph-only baseline and two controls that establish whether any gain from chemistry is real. The contribution is the application and the controls, not the gating idea in the abstract — Section 7 isolates how much of adaptive's advantage is the gating architecture itself versus real chemistry."),
      bullet("baseline — learned per-node ID embedding only (identical in kind for all 11 node types)."),
      bullet("fingerprint — the embedding concatenated with a projected real Morgan fingerprint, for Compound nodes only."),
      bullet("random_fingerprint (control) — architecturally identical to fingerprint, but each compound's fingerprint row is swapped for a different, randomly assigned compound's fingerprint, fixed per seed. This isolates whether any gain from fingerprint is real chemical signal or simply the extra parameters/dimensionality."),
      bullet("adaptive (proposed) — an evidence-aware gate convexly blends the embedding and the projected fingerprint instead of concatenating them: g_c = sigmoid(a·log(1+degree_c) + b), h_c = g_c·h_embed + (1−g_c)·h_chem. degree_c is compound c's non-CtD graph degree, computed from the TRAIN-only graph (so cold-start test compounds' degree never leaks future information), making this feature available even for a compound the model has never seen treat anything. a and b are the gate's only two parameters, learned jointly with the rest of the model."),
      h2("3.4 Training and leakage control"),
      p("Training is full-batch (one encoder forward pass per epoch over all 47,031 nodes), with one sampled negative disease per positive CtD edge (filtered against every known true CtD target for that compound), optimized with Adam (lr = 0.01, weight decay = 1e-5, 50 epochs). Whichever split is active, the CtD edges available for message passing are restricted to that split's TRAIN edges only — test edges are never visible to the encoder during training or evaluation, since CtD is simultaneously the message-passing relation and the prediction target."),

      h1("4. Evaluation Protocol"),
      bullet("Random split — transductive 80/20 split of the 755 CtD edges (604 train / 151 test)."),
      bullet("Cold-start split — 20% of the 387 distinct CtD compounds (77) have ALL their CtD edges held out (551 train / 204 test); the model never sees a treats-edge for these compounds. Held-out compounds keep their other Hetionet edges."),
      bullet("Exhaustive ranking — every test edge is ranked against EVERY eligible disease (not a 50-negative sample as in the first pass), since Hetionet only has 137 disease nodes — this removes a source of evaluation noise at negligible extra cost."),
      bullet("3 seeds per condition (0, 1, 2) — every (split, variant) combination is trained and evaluated 3 times; all results are reported as mean ± SD, not a single run."),
      p("Metrics: AUROC and AUPRC over the pooled positive/negative pool, and Hits@1/3/10. Per-edge diagnostics (the test compound's non-CtD graph degree and its true disease's rank) are saved for every run, enabling the evidence-stratification analysis in Section 6."),

      h1("5. Results"),
      p("All 24 runs (3 seeds × 2 splits × 4 variants) use identical hyperparameters. Every number below is read directly from results/*.json; results/summarize_results.py computed the means and standard deviations — nothing here was estimated."),
      resultsTable(),
      caption("Table 2. Link-prediction metrics (mean ± SD over 3 seeds), full Hetionet graph."),
      image("../figures/comparison_auroc_auprc.png", 500, 220),
      caption("Figure 1. AUROC and AUPRC by split and variant, mean ± SD error bars over 3 seeds."),

      h2("5.1 What the controls reveal"),
      p("A methodological note on random_fingerprint before the comparisons below: an earlier version of its shuffle fixup had a one-compound edge case — when exactly one compound was left mapped to itself after the initial random permutation, the fixup (meant to swap it with another compound) was a no-op on a length-1 tensor, silently leaving that one compound matched to its own real fingerprint. This was caught during review; it affected 2 of the 3 seeds (seeds 0 and 2 each had exactly one residual self-match; seed 1's draw already had zero). It is fixed in src/train.py, and all 6 random_fingerprint runs below were re-trained under the corrected, verified derangement before any of these numbers were trusted.", { italics: true, size: 20, color: "555555" }),
      p("On the random split, the fingerprint variant's AUROC (0.859 ± 0.020) and its shuffled-fingerprint control's AUROC (0.837 ± 0.032) are numerically very similar and fall within the observed seed-to-seed variability. That is the first honest finding: naive concatenation's apparent benefit on the easy split is not reliably attributable to real chemical signal; it is consistent with simply adding model capacity."),
      p("On the cold-start split, fingerprint fusion underperforms the graph-only baseline on every single metric (AUROC 0.801 vs 0.837; Hits@10 0.508 vs 0.650). Its shuffled-fingerprint control underperforms baseline by a comparable margin (AUROC 0.811 vs 0.837; Hits@10 0.575 vs 0.650) and is itself numerically close to the real-fingerprint variant rather than clearly worse — so the cold-start degradation is not specific to being given the wrong (shuffled) chemistry; naive concatenation itself, real fingerprints or shuffled, hurts performance under the hard evaluation regime. This still echoes the broader concern in the 2026 pharmacology-knowledge-graph literature that explicit molecular structure is not a free improvement for topology-based repurposing models. The seed-to-seed spread on these cells is large (fingerprint Hits@10 SD = 0.117, its control's SD = 0.106; individual fingerprint seeds ranged from AUROC 0.751 to 0.858) — the kind of variance a single-seed report would have hidden entirely."),
      p("Adaptive achieves the strongest overall performance, outperforming baseline on all random-split metrics and on AUROC, AUPRC, Hits@3 and Hits@10 under cold-start, while baseline remains slightly higher on cold-start Hits@1 (0.150 vs 0.162 — the one red cell in an otherwise green row in Table 2). It also shows tighter variance than the fingerprint or random_fingerprint variants in cold-start (Hits@10 SD = 0.027 vs 0.117 and 0.106 respectively). That is the project's central, defensible result — strongest overall, not uniformly ahead on every single cell."),

      h2("5.2 Comparison to Additional Baselines"),
      p("Every number in Table 2 already compares adaptive against a graph-only R-GCN baseline and two fusion controls. This subsection adds two more points of comparison, one run ourselves and one from the literature, and is explicit about which is a verified apples-to-apples comparison and which is not."),
      p("Own additional baseline: no message passing at all. src/train.py --variant baseline --num-layers 0 decodes directly from the raw, per-node learned embedding — the two RelationalGraphConvLayer message-passing layers are simply not applied, so a compound's representation never incorporates information from its graph neighbors. Everything else is identical to every other run in this report: same data, same two splits, same negative sampling, same exhaustive ranking, same 3 seeds. This is the one comparison in this section guaranteed to satisfy \"same data and evaluation settings,\" by construction, rather than by inspection of someone else's paper."),
      noGnnTable(),
      caption("Table 5. No-GNN embedding-only baseline vs. the R-GCN baseline from Table 2 (mean ± SD over 3 seeds)."),
      p("On the random split the no-GNN baseline is nearly as strong as the full R-GCN baseline (AUROC 0.835 vs. 0.840; Hits@10 0.525 vs. 0.609) — unsurprising, since under a transductive split a held-out edge's compound still appears in many other training edges, so its embedding is trained regardless of message passing. On the cold-start split it collapses to essentially chance (AUROC 0.476, Hits@10 0.163, vs. 0.837 and 0.650 for the R-GCN baseline): a held-out compound in the cold-start split never appears as the source of a training edge under this architecture, so without message passing its embedding never leaves its random initialization. This is a clean, mechanistic demonstration of why relational message passing — not simply having more parameters — is what makes cold-start generalization possible at all here, and it sharpens the framing in Section 1: the R-GCN encoder itself, not only the gated fusion on top of it, is doing load-bearing work for the cold-start case this report is centered on."),
      p("Literature comparison, with an explicit caveat. Liu et al. (2021; workshop version 2020) report an R-GCN baseline on the same target relation. Their random split of the Compound-treats-Disease relation uses 483 train / 121 validation / 151 test triples — 755 total, and a 151-edge test set that matches this project's own Random-split test set exactly — so the underlying data is confirmed identical. Their reported R-GCN numbers are Hits@1 0.026, Hits@3 0.245, Hits@10 0.272, MRR 0.135 (journal version), noticeably lower than this project's own Random-split R-GCN baseline (Hits@3 0.362, Hits@10 0.609, Table 2). We could not confirm why: their paper does not state whether a test triple's true disease is ranked against all ~47,000 Hetionet entities or only the 137 Disease nodes, and no public code repository for this work could be located to check the implementation directly. Our own evaluation ranks exclusively against the 137 Disease nodes (Section 4), so if Liu et al. rank against all entities, their harder candidate pool — not a weaker model — would explain the gap, and the two numbers would not be measuring the same thing despite sharing the same training data. Per the explicit guidance we received when we raised this with the course instructor — that a literature citation is only a fair comparison when it uses the same data AND the same evaluation settings — we report Liu et al.'s numbers here as related work and context, not as a verified head-to-head baseline, precisely because that second condition could not be confirmed. The no-GNN baseline above is offered as the comparison that does meet both conditions."),

      h1("6. Evidence Stratification: does graph evidence predict when chemistry helps?"),
      p("The motivating hypothesis was that the adaptive gate's benefit should be concentrated in compounds with LOW non-CtD graph evidence, where the model has little else to rely on. To test this directly rather than assume it, every cold-start test edge was assigned to a Low/Medium/High graph-evidence tertile based on the non-CtD degree of its source compound (computed from that run's own train-only graph, so no leakage), and Hits@10 and mean rank were computed per bucket per variant, averaged over the 3 seeds."),
      stratTable(),
      caption("Table 3. Cold-start Hits@10 and mean rank by compound graph-evidence tertile (mean ± SD over 3 seeds; ~67–69 test edges per bucket)."),
      image("../figures/evidence_stratification.png", 460, 310),
      caption("Figure 2. Cold-start Hits@10 by graph-evidence tertile."),
      p("The result does not confirm the original hypothesis, and is reported as obtained rather than reframed to fit it. Adaptive is comparable to baseline for low and medium evidence and clearly stronger in the high-evidence tertile (low: 0.583 vs 0.569, within noise on ~68 edges; medium: 0.725 vs 0.744, baseline marginally ahead; high: 0.741 vs 0.637, a clear win) — the advantage is concentrated in the medium/high tertiles, not the low-evidence tertile the original hypothesis targeted. What the data DOES support, consistently across every tertile, is that naive fingerprint concatenation hurts relative to baseline at every evidence level (low: 0.407, medium: 0.652, high: 0.463), and its shuffled-fingerprint control is no more reliable (low: 0.461, medium: 0.686, high: 0.577) — both static-fusion variants trail baseline in all three buckets. The honest statement of the contribution is therefore narrower than \"chemistry rescues low-evidence drugs\": naive multimodal fusion is not reliably beneficial for cold-start knowledge-graph repurposing at any evidence level, and an evidence-aware gate performs at least comparably to the graph-only baseline across the evidence spectrum and clearly better where graph evidence is richest, rather than specifically rescuing the low-evidence compounds the original hypothesis targeted."),

      h1("7. Isolating the Gate from the Chemistry: A Novelty Control"),
      p("The comparison in Section 5.1 isolates whether naive concatenation benefits from real chemistry (fingerprint vs. its random_fingerprint control) — but adaptive uses a different fusion mechanism entirely (a learned, degree-conditioned gate, not concatenation), so that comparison alone cannot say whether adaptive's advantage over baseline comes from using real molecular information or simply from the gated architecture itself. A fifth variant, adaptive_random_fingerprint, answers this directly: it repeats the adaptive architecture exactly — same gate, same degree-conditioning — but is fed the same shuffled fingerprints as random_fingerprint. 6 additional runs (3 seeds × 2 splits)."),
      ablationTable(),
      caption("Table 4. Novelty-control ablation: adaptive with real vs. shuffled fingerprints (mean ± SD over 3 seeds)."),
      image("../figures/adaptive_ablation.png", 520, 165),
      caption("Figure 3. AUROC, AUPRC and Hits@10 for adaptive with real vs. shuffled fingerprints, both splits."),
      p("Two things stand out. First, the gated architecture itself accounts for a substantial share of adaptive's advantage over the static fusion variants: even fed the WRONG (shuffled) fingerprints, adaptive_random_fingerprint beats both fingerprint and random_fingerprint — the static variants — on every metric in both splits (e.g. cold-start AUROC 0.843 vs 0.801 / 0.811; cold-start Hits@10 0.613 vs 0.508 / 0.575). The gate and its per-compound, degree-conditioned blend are doing real work independent of which fingerprint they receive. Second, real chemistry still adds something on top of that: adaptive with real fingerprints is numerically ahead of adaptive with shuffled fingerprints on 9 of the 10 (split × metric) comparisons in Table 4 — most consistently, and by the largest margin, on Hits@10 in both splits (random: 0.762 ± 0.047 vs 0.687 ± 0.019; cold-start: 0.683 ± 0.027 vs 0.613 ± 0.061). Most individual gaps are within one combined standard deviation at only 3 seeds, so this should be read as suggestive rather than conclusive."),
      p("The honest answer to \"does adaptive improve because it uses meaningful molecular information, or simply because the gating architecture is better\" is therefore layered, not single-cause: most of the improvement over naive static fusion is architectural — the gating and degree-conditioning mechanism itself, which is beneficial even with scrambled chemistry — with a smaller, metric-dependent contribution from the correct compound–structure correspondence that shows up mainly in top-k retrieval (Hits@10) and not reliably in AUROC/AUPRC at this sample size (755 CtD edges, 3 seeds). A larger CtD relation, or more seeds, would be needed to state the chemistry-specific contribution with more confidence than that."),

      h1("8. Discussion and Limitations"),
      p("Three limitations are worth stating plainly. First, even with exhaustive ranking and 3 seeds, Hetionet's CtD relation is small (755 edges over 137 diseases), so per-bucket evidence-stratification cells hold only ~67–69 edges — real effects and noise are both plausible at this scale, which is why Table 3's error bars are wide. Second, 3 seeds is a minimum for a defensible mean ± SD, not a generous one; the cold-start fingerprint cell's wide spread suggests more seeds would narrow, but likely not qualitatively change, the picture. Third, the gate's two learned parameters (a, b) converged to similar values across splits and seeds (a ≈ 0.30–0.38, b ≈ 0.0–0.06 in every successful run), which keeps g_c in a fairly narrow, nearly-linear-in-log-degree range for the degrees actually observed in Hetionet — a more expressive gate (e.g. a small MLP over additional evidence features) is a natural next step but was not justified by this project's modest data scale."),
      p("Taken together, the results support a specific, bounded claim: for cold-start knowledge-graph drug repurposing on Hetionet, naively fusing molecular structure into a relational GNN is not reliably beneficial and can underperform a graph-only model, while an evidence-aware gate that performs degree-conditioned molecular feature fusion before relational message passing recovers and exceeds graph-only performance without the fusion variant's instability — most of that recovery from the gating mechanism itself (Section 7), with a smaller contribution from real chemistry visible mainly in top-k retrieval. This is a different, and in places less convenient, story than \"adding chemistry helps\" — which is exactly why it is reported in full rather than simplified."),

      h1("9. Reproducibility"),
      p("Fast demo (no download, ~15 seconds): pip install -r requirements.txt, then python src/train.py --data toy_data/graph_toy.pt --split random --variant adaptive --epochs 15 --toy --seed 0."),
      p("Full reproduction (~3–5 min per run, CPU-only): bash scripts/download_data.sh, then python src/data_prep.py, then the main 3-seed × 2-split × 4-variant loop (24 runs) plus the 3-seed × 2-split adaptive_random_fingerprint ablation (6 more runs, 30 total) in README.md, then python src/summarize_results.py, which regenerates Table 2, Table 3, Table 4, Figure 1, Figure 2, and Figure 3 exactly as shown here."),
      p("Code-availability statement: all source code, the toy dataset, every raw results JSON file (including per-edge diagnostics), every figure, and the full run logs for all 30 experiments are included in this submission exactly as produced; none of the reported numbers were hand-edited after generation."),

      h1("10. Conclusion"),
      p("This project reproduced the R-GCN + DistMult approach to KG-based drug repurposing on real Hetionet data, found and fixed a genuine normalization bug in the encoder, and replaced a single-seed, sampled-negative evaluation with a 3-seed, exhaustive one — a change that materially altered the scientific conclusion. Rather than reporting that adding molecular fingerprints to a KG-GNN helps, the corrected, controlled experiments show that naive fusion is unreliable and can underperform a graph-only model in the realistic cold-start regime, while an evidence-aware gated fusion mechanism recovers overall performance across the range of graph evidence (though not on every single metric — see Section 5.1). The stratified analysis shows this benefit is concentrated in the medium- and high-evidence tertiles rather than the low-evidence tertile originally hypothesized, and the novelty-control ablation in Section 7 shows that most of adaptive's advantage over naive fusion comes from the gating architecture itself, with a smaller, metric-dependent contribution from real chemistry visible mainly in top-k retrieval — both results reported as found rather than reframed to fit the original pitch. Every number here is runnable end-to-end from the included code."),

      h1("References"),
      p("Himmelstein, D.S., Lizee, A., Hessler, C., et al. (2017). Systematic integration of biomedical knowledge prioritizes drugs for repurposing. eLife, 6, e26726. (Hetionet v1.0 / Project Rephetio)"),
      p("Huang, K., Chandak, P., Wang, Q., et al. (2024). A foundation model for clinician-centered drug repurposing. Nature Medicine. (TxGNN)"),
      p("Schlichtkrull, M., Kipf, T.N., Bloem, P., et al. (2018). Modeling Relational Data with Graph Convolutional Networks. ESWC. (R-GCN)"),
      p("Yang, B., Yih, W., He, X., Gao, J., Deng, L. (2015). Embedding Entities and Relations for Learning and Inference in Knowledge Bases. ICLR. (DistMult)"),
      p("Yu, Y., Huang, K., Zhang, C., Glass, L., Sun, J., Xiao, C. (2021). SumGNN: Multi-typed Drug Interaction Prediction via Efficient Knowledge Graph Summarization. Bioinformatics, 37(18), 2988–2995."),
      p("Pharmacology knowledge graphs enable drug repurposing without chemical structure information (2026). https://doi.org/10.1007/s44163-026-01303-2"),
      p("Liu, Y., Hildebrandt, M., Joblin, M., Ringsquandl, M., Raissouni, R., Tresp, V. (2021). Neural Multi-Hop Reasoning With Logical Rules on Biomedical Knowledge Graphs. arXiv:2103.10367. (Workshop version: Integrating Logical Rules Into Neural Multi-Hop Reasoning for Drug Repurposing, ICML 2020 Workshop on Graph Representation Learning and Beyond, arXiv:2007.05292.)"),
      p("Course materials: CBIO 7215, Weeks 3–4 lecture slides (R-GCN message passing; TxGNN; cold-start evaluation hierarchy)."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("CBIO7215_Final_Project_Report.docx", buf);
  console.log("wrote CBIO7215_Final_Project_Report.docx");
});
