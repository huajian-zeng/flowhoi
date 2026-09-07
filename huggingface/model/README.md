---
language:
  - en
library_name: pytorch
tags:
  - flow-matching
  - hand-object-interaction
  - motion-generation
  - robotics
  - arxiv:2602.13444
---

# FlowHOI: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation

Pretrained checkpoints for **FlowHOI**, a two-stage framework for hand-object interaction generation: an EgoDex grasping prior followed by GRAB or HOT3D manipulation.

| | |
| --- | --- |
| Paper | [arXiv 2602.13444](https://arxiv.org/abs/2602.13444) |
| Project page | https://huajian-zeng.github.io/projects/flowhoi/ |
| Code | https://github.com/huajian-zeng/flowhoi |
| Demo data | [huajian-zeng/flowhoi-data](https://huggingface.co/datasets/huajian-zeng/flowhoi-data) |

## Files

| File | Description |
| --- | --- |
| `egodex_grasp/model000500000.pt` | Shared grasping prior trained on EgoDex |
| `grab_full/model000200000.pt` | GRAB manipulation model |
| `hot3d_full/model000200000.pt` | HOT3D manipulation model with scene conditioning |

Each checkpoint has an accompanying `args.json` configuration file.

## Usage

Clone the [FlowHOI repository](https://github.com/huajian-zeng/flowhoi) and follow its installation instructions. Then run from the repository root:

```bash
bash scripts/download_pretrained.sh
bash scripts/download_data.sh
bash demo.sh                  # GRAB
DATASET=hot3d bash demo.sh     # HOT3D
```

This release provides inference and visualization with three prepared demo sequences. Rendering requires the separately obtained MANO/SMPL-X and dataset assets described in the code README.

[Code license](https://github.com/huajian-zeng/flowhoi/blob/main/LICENSE)

## Citation

```bibtex
@article{zeng2026flowhoi,
  title   = {{FlowHOI}: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation},
  author  = {Zeng, Huajian and Chen, Lingyun and Yang, Jiaqi and Zhang, Yuantai and Shi, Fan and Liu, Peidong and Zuo, Xingxing},
  journal = {arXiv preprint arXiv:2602.13444},
  year    = {2026}
}
```
