<p align="center">
  <h1 align="center"><strong>FlowHOI: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation</strong></h1>
  <p align="center">
    <a href="https://huajian-zeng.github.io/">Huajian Zeng</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=vdo-1aYAAAAJ&hl">Lingyun Chen</a><sup>2</sup>, <a href="https://scholar.google.com/citations?hl=zh-CN&user=f7ox7CIAAAAJ">Jiaqi Yang</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=Lh2CthAAAAAJ&hl">Yuantai Zhang</a><sup>1</sup>, <a href="https://fanshi14.github.io/me/">Fan Shi</a><sup>3</sup>, <a href="https://ethliup.github.io/">Peidong Liu</a><sup>4</sup>, <a href="https://xingxingzuo.github.io/">Xingxing Zuo</a><sup>1</sup>
    <br>
    <sup>1</sup>Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), <sup>2</sup>Technical University of Munich (TUM), <sup>3</sup>National University of Singapore (NUS), <sup>4</sup>Westlake University
    <br>
  </p>
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2602.13444-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2602.13444)
[![Homepage](https://img.shields.io/badge/Homepage-%F0%9F%8C%90-blue)](https://huajian-zeng.github.io/projects/flowhoi/)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-flowhoi-yellow)](https://huggingface.co/huajian-zeng/flowhoi)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-flowhoi--data-yellow)](https://huggingface.co/datasets/huajian-zeng/flowhoi-data)

</div>

## Updates

[2026-02-13] Paper uploaded to [arXiv](https://arxiv.org/abs/2602.13444).

[2026-09-07] Two-stage inference and visualization code released, along with pretrained checkpoints and GRAB/HOT3D demo data.

## 🔥 Highlight

**FlowHOI** is a two-stage flow-matching framework that generates semantically
grounded, temporally coherent hand-object interaction (HOI) sequences -- hand
poses, object poses, and hand-object contact states -- conditioned on an
egocentric observation, a language instruction, and a 3D scene reconstruction.

A geometry-centric **Grasping** stage, pretrained on HOI data reconstructed
from large-scale egocentric videos (EgoDex), produces a contact-stable grasp;
a semantics-centric **Manipulation** stage generates the subsequent
interaction. This repository contains the inference and visualization code.

## Installation

1. Create and activate a conda environment:
    ```bash
    conda create -n flowhoi python=3.10 -y
    conda activate flowhoi
    ```
2. Install PyTorch (tested with 2.5.1 + CUDA 12.4). Install torchvision in
   the same command: `vit_pytorch` pulls it in, and letting pip fetch it from
   PyPI in the next step would replace your torch build with whatever version
   that torchvision pins. See the
   [official PyTorch website](https://pytorch.org/get-started/locally/) for
   the command matching your CUDA version:
    ```bash
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
    ```
3. Install the remaining dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Install chumpy (needed to load the MANO model files; the PyPI release is
   incompatible with recent numpy, and its setup requires disabling build
   isolation):
    ```bash
    pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
    ```

The T5-Large text encoder (about 3 GB) is downloaded automatically on first
use. It is always loaded, so prompts outside the precomputed text features
work as well.

## Download Checkpoints and Data

1. Pretrained checkpoints (grasping prior, GRAB and HOT3D manipulation
   models) from [huajian-zeng/flowhoi](https://huggingface.co/huajian-zeng/flowhoi):
    ```bash
    bash scripts/download_pretrained.sh
    ```
2. The example data package (the demo sequences with their normalization
   statistics, projection matrices, BPS encodings, text features, and HOT3D
   scene inputs) from
   [huajian-zeng/flowhoi-data](https://huggingface.co/datasets/huajian-zeng/flowhoi-data):
    ```bash
    bash scripts/download_data.sh
    ```
   The package covers the demo sequences only. The full preprocessed splits
   are not redistributed: they are rebuilt from the raw datasets with the
   preprocessing pipeline, which is not part of this release yet (see the TODO
   list below).
3. MANO / SMPL-X body models (registration required) from
   [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) into `assets/smplx/`
   (used for visualization; the path is configured in `aitvconfig.yaml`).
4. GRAB object meshes and subject hand templates (registration required) from
   [grab.is.tue.mpg.de](https://grab.is.tue.mpg.de/). Its licence does not
   permit redistribution, so they are not part of the data package. From the
   extracted `tools/` directory, copy the object meshes (`contact_meshes`,
   under `object_meshes`) into `assets/contact_meshes/`, and the per-subject
   templates (`s*_lhand.ply`, `s*_rhand.ply`, under `subject_meshes`) into
   `assets/female/` and `assets/male/`, keeping the two gender folders. Both
   are needed to render GRAB sequences; the HOT3D path uses neither.
5. HOT3D object models, for rendering HOT3D sequences. Download
   the object library with the [HOT3D toolkit](https://github.com/facebookresearch/hot3d)
   (`dataset_downloader_base_main.py -c Hot3DAssets_download_urls.json`) and put
   the `.glb` files into `assets/hot3d_assets/`. The renderer resolves them
   through `HOT3D_OBJECT_MESH_UID` in `visualize/visualize_sequences.py`.
6. `ffmpeg` and `ffprobe` on `PATH`, for writing videos. If either is
   missing, install both in the active environment:
    ```bash
    conda install -c conda-forge ffmpeg -y
    ```

## Quick Start

```bash
bash demo.sh
```

This generates the two GRAB demo sequences -- top-scoring test sequences with
distinct objects and actions (passing a mug, playing with a toy train) -- and
renders them to `outputs/demo/ours_videos/*.mp4`. Rendering needs the GRAB
meshes from step 4; generation itself does not. Likewise for the
scene-conditioned HOT3D model, whose object models come from step 5:

```bash
DATASET=hot3d bash demo.sh
```

The HOT3D demo writes its videos to `outputs/demo_hot3d/ours_videos/`.
Both demos select `--split_set demo`, the split included in the data package.

On machines without a display, the renderer falls back to an EGL context when
the installed aitviewer supports it; otherwise run `xvfb-run -a bash demo.sh`.

## Inference

Two-stage generation with explicit arguments:

```bash
python -m sample.sample_2stage \
    --model_path save/hot3d_full/model000200000.pt \
    --grasp_model_path save/egodex_grasp/model000500000.pt \
    --guidance --num_samples 1 --seed 42 --split_set demo \
    --output_dir outputs/demo_hot3d
```

The model configuration is read from the `args.json` next to each checkpoint;
arguments passed explicitly on the command line take precedence, including
`--name=value` and boolean overrides such as `--no_random_order`.
The released checkpoints use a 50-frame grasp phase (keyframe index 49);
`--max_frames_interaction` accepts integer lengths from 50 to 200.
`--guidance_param` defaults to 1, using one conditional prediction per step
as in the released sampler. Values above 1, such as `--guidance_param 2.5`,
explicitly enable classifier-free guidance and change the generated motion;
they are outside the validated demo configuration. `--guidance` separately
enables pose-gradient guidance. The grasp stage follows the released
checkpoint's sampling convention without an additional `x_start` pose condition.

The HOT3D model conditions on the reconstructed 3D scene (a global occupancy token plus
local Concerto/SceneSplat tokens); the sampler loads these scene inputs from
the data package and refuses to run without them rather than silently
dropping the conditioning.

`--num_samples N` selects up to N dataset conditions, and
`--num_repetitions R` generates R motions for each selected condition.
With `--eval_entire_set`, all conditions are generated and N becomes the batch
size; the last incomplete batch is retained. `--random_order` shuffles one shared
ID order for both stages. Saved results include the actual condition count,
`num_generated`, `requested_num_samples`, and each row's `repetition_index`.

## Visualization

```bash
python -m visualize.visualize_sequences \
    --folder_path outputs/demo \
    --dataset grab --output_fps 30 --range_max 2
```

For the HOT3D output generated above, use its dataset and frame rate:

```bash
python -m visualize.visualize_sequences \
    --folder_path outputs/demo_hot3d \
    --dataset hot3d --output_fps 15 --range_max 1
```

These commands render MANO hand meshes and the object mesh to mp4.
`--show_scene` can additionally render a scene point cloud for HOT3D.
By default it reads aligned `xyz` / `features` arrays from
`dataset/HOT3D_HANDS/local_scenes_5000/{sequence_name}.npz`; set
`--local_scenes_path` to use another directory. To render per-recording
Concerto point clouds instead, select that source explicitly:

```bash
python -m visualize.visualize_sequences \
    --folder_path outputs/demo_hot3d \
    --dataset hot3d --output_fps 15 --range_max 1 \
    --show_scene --scene_source concerto_grid
```

This mode reads `coords_grid.npy`, `coords_filtered.npy`, and
`features_grid.npy` from
`dataset/HOT3D_HANDS/scene_data/{recording_id}/concerto_features/point_cloud/`,
with the corresponding sequence anchor under `dataset/HOT3D_HANDS/anchors/`.
Neither the local scene files nor these per-recording point clouds are in
the default data package; its precomputed scene features support model
conditioning without these optional visualization inputs.

## Tests

The regression tests cover argument handling and sampler behavior on CPU
without downloading model weights or data:

```bash
python -m unittest discover -s tests -v
```

## TODO List

The current release ships **inference + visualization only**. The following
components are planned for future release:

- [ ] Evaluation code
- [ ] Training code
- [ ] Data preprocessing code

## Citation

If you find this repository useful for your research, please consider citing:

```bibtex
@article{zeng2026flowhoi,
  title   = {{FlowHOI}: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation},
  author  = {Zeng, Huajian and Chen, Lingyun and Yang, Jiaqi and Zhang, Yuantai and Shi, Fan and Liu, Peidong and Zuo, Xingxing},
  journal = {arXiv preprint arXiv:2602.13444},
  year    = {2026}
}
```

## License and Acknowledgements

This code is released under the
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license (see
`LICENSE`).

FlowHOI builds on [DiffH2O](https://github.com/diffh2o/diffh2o). Other adapted
code keeps its attribution in the file headers; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for its licences and sources.

The data package carries only data derived for FlowHOI. The GRAB, HOT3D and
MANO / SMPL-X assets it builds on are not redistributed here -- download them
from their own sites as described above, under their own licences.
