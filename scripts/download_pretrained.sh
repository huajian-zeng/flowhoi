#!/bin/bash
# Download the pretrained FlowHOI checkpoints from Hugging Face into save/.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
REPO=${1:-huajian-zeng/flowhoi}
hf download "$REPO" --include 'egodex_grasp/**' --include 'grab_full/**' --include 'hot3d_full/**' --local-dir save
for file in egodex_grasp/args.json egodex_grasp/model000500000.pt \
            grab_full/args.json grab_full/model000200000.pt \
            hot3d_full/args.json hot3d_full/model000200000.pt; do
  if [ ! -s "save/$file" ]; then
    echo "Download incomplete: save/$file is missing or empty." >&2
    exit 1
  fi
done
echo "Checkpoints downloaded to save/"
