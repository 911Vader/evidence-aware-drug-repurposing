#!/usr/bin/env bash
# download_data.sh
#
# Downloads the raw data this project needs. Not shipped in the
# submission because the processed graph (data/processed/graph.pt,
# ~400MB once the dense per-compound fingerprint matrix is included)
# is far too large to hand in; run this once, then src/data_prep.py.
#
# Sources (all public, no auth required):
#   - Hetionet v1.0 nodes/edges: hetio/hetionet (Himmelstein et al.,
#     eLife 2017), https://github.com/hetio/hetionet
#   - DrugBank InChI lookup table: dhimmel/drugbank (the companion
#     dataset the Hetionet authors publish for exactly this purpose),
#     https://github.com/dhimmel/drugbank
#
# NOTE on the edges file: GitHub serves it through Git LFS. Fetching it
# from raw.githubusercontent.com returns only a ~130-byte LFS *pointer*
# file, not the actual data -- you must use the media.githubusercontent.com
# LFS-resolution endpoint instead, which this script does.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

echo "Downloading Hetionet node list..."
curl -sL -o data/raw/hetionet-v1.0-nodes.tsv \
  "https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv"

echo "Downloading Hetionet edge list (via Git-LFS media endpoint, ~12MB)..."
curl -sL -o data/raw/hetionet-v1.0-edges.sif.gz \
  "https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz"

echo "Downloading DrugBank InChI lookup table (dhimmel/drugbank)..."
curl -sL -o data/raw/drugbank_dhimmel.tsv \
  "https://raw.githubusercontent.com/dhimmel/drugbank/gh-pages/data/drugbank.tsv"

echo "Done. Files in data/raw/:"
ls -la data/raw/

echo
echo "Sanity check (edges file must be real gzip, not a ~130-byte LFS pointer):"
file data/raw/hetionet-v1.0-edges.sif.gz
