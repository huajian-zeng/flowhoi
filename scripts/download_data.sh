#!/bin/bash
# Download the demo data files into dataset/ relative to this repository.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
REPO=${1:-huajian-zeng/flowhoi-data}
hf download --repo-type dataset "$REPO" --include 'dataset/**' --local-dir .
for file in GRAB_HANDS/test_demo.txt HOT3D_HANDS/test_demo.txt grab_opt_objects.txt hot3d_opt.txt; do
  if [ ! -s "dataset/$file" ]; then
    echo "Download incomplete: dataset/$file is missing or empty." >&2
    exit 1
  fi
done
echo "Demo data downloaded to dataset/"
