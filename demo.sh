#!/bin/bash
# FlowHOI quick-start demo: generate HOI sequences with the two-stage
# flow-matching model and render them to mp4.
#
# Prerequisites (see README):
#   - checkpoints under save/          (Hugging Face)
#   - example data under dataset/     (Hugging Face)
#   - MANO / SMPL-X models under assets/smplx/
#   - GRAB meshes and hand templates, or HOT3D object models, under assets/
#     (download separately from the dataset providers; see README)
#   - a system ffmpeg that ships ffprobe
#
# Usage:
#   bash demo.sh                # GRAB: the two best test sequences
#   DATASET=hot3d bash demo.sh  # HOT3D: scene-conditioned model
set -e

DATASET=${DATASET:-grab}
if [ "$DATASET" != "grab" ] && [ "$DATASET" != "hot3d" ]; then
  echo "Unsupported DATASET: $DATASET (choose grab or hot3d)." >&2
  exit 1
fi
if [ "$DATASET" = "hot3d" ]; then
  MODEL=${MODEL:-save/hot3d_full/model000200000.pt}
  SPLIT=${SPLIT:-demo}
  N=${N:-1}
  FPS=${FPS:-15}
  OUT=${OUT:-outputs/demo_hot3d}
else
  MODEL=${MODEL:-save/grab_full/model000200000.pt}
  SPLIT=${SPLIT:-demo}
  N=${N:-2}
  FPS=${FPS:-30}
  OUT=${OUT:-outputs/demo}
fi
GRASP_MODEL=${GRASP_MODEL:-save/egodex_grasp/model000500000.pt}

for f in "$MODEL" "$GRASP_MODEL"; do
  if [ ! -f "$f" ]; then
    echo "Missing checkpoint: $f"
    echo "Download the checkpoints from Hugging Face into save/ (see README)."
    exit 1
  fi
done
for d in dataset; do
  if [ ! -d "$d" ] || [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
    echo "$d/ is missing or empty."
    echo "Extract the Hugging Face data package in the repo root (see README)."
    exit 1
  fi
done

echo "== Stage 1+2 inference ($N sample(s), split: $SPLIT) =="
python -m sample.sample_2stage \
  --model_path "$MODEL" \
  --grasp_model_path "$GRASP_MODEL" \
  --guidance --num_samples "$N" --seed 42 \
  --split_set "$SPLIT" \
  --output_dir "$OUT"

echo
echo "== Rendering to mp4 =="
python -m visualize.visualize_sequences \
  --folder_path "$OUT" \
  --dataset "$DATASET" --output_fps "$FPS" --range_max "$N" \
  --resolution medium

echo
echo "== Done. Demo video(s): =="
find "$OUT" -name "*.mp4"
