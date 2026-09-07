---
pretty_name: FlowHOI Demo Data
language:
  - en
size_categories:
  - n<1K
tags:
  - robotics
  - hand-object-interaction
  - motion-generation
  - 3d
  - flow-matching
  - numpy
  - pytorch
---

# FlowHOI Demo Data

Demo inputs for **FlowHOI**, containing **two GRAB sequences and one HOT3D sequence** for inference and visualization.

| | |
| --- | --- |
| Paper | [arXiv 2602.13444](https://arxiv.org/abs/2602.13444) |
| Project page | https://huajian-zeng.github.io/projects/flowhoi/ |
| Code | https://github.com/huajian-zeng/flowhoi |
| Checkpoints and installation | [huajian-zeng/flowhoi](https://huggingface.co/huajian-zeng/flowhoi) |

## Files

| Directory | Description |
| --- | --- |
| `dataset/GRAB_HANDS/` | Mug-passing and toy-train demos, with captions, normalization statistics, and precomputed object/text features |
| `dataset/HOT3D_HANDS/` | Cellphone demo, with captions, object/text features, occupancy, and scene inputs |
| `dataset/` shared files | Sequence lookup tables and loader compatibility files |

This package contains only the demo examples, not the full GRAB/HOT3D splits or EgoDex training data.

## Usage

Follow the [model repository's installation instructions](https://huggingface.co/huajian-zeng/flowhoi), then download from the FlowHOI code repository root:

```bash
hf download --repo-type dataset huajian-zeng/flowhoi-data \
  --include 'dataset/**' --local-dir .
```

The code repository's `demo.sh` uses these files with the released checkpoints.
MANO models, GRAB meshes and subject templates, and HOT3D object meshes are obtained separately as described in the installation guide.

The original data providers’ terms apply; see the [code repository](https://github.com/huajian-zeng/flowhoi) for data sources and access requirements.

## Citation

```bibtex
@article{zeng2026flowhoi,
  title   = {{FlowHOI}: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation},
  author  = {Zeng, Huajian and Chen, Lingyun and Yang, Jiaqi and Zhang, Yuantai and Shi, Fan and Liu, Peidong and Zuo, Xingxing},
  journal = {arXiv preprint arXiv:2602.13444},
  year    = {2026}
}
```
