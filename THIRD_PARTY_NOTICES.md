# Third-party code and assets

FlowHOI's own code uses the license in `LICENSE`. The following adapted code
retains its upstream notices and license terms:

| Source | Used in FlowHOI | Included license |
| --- | --- | --- |
| [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) | `diffusion/nn.py` | [MIT](third_party/licenses/guided-diffusion-MIT.txt) |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | `utils/hfargparse.py` | [Apache 2.0](third_party/licenses/transformers-Apache-2.0.txt) |
| [Text-to-Motion](https://github.com/EricGuo5513/text-to-motion) | `data_loaders/humanml/` | [MIT](third_party/licenses/text-to-motion-MIT.txt) |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | `utils/rotation_conversions.py`, via ACTOR | [BSD](utils/PYTORCH3D_LICENSE) |

The repository also builds on [DiffH2O](https://github.com/diffh2o/diffh2o).
Existing copyright notices in adapted files are retained. FlowHOI modifies
these components for hand-object data, two-stage flow matching, configuration
loading, and inference.

Installed dependencies retain their own licenses. T5 weights, checkpoints,
and example data are downloaded separately. MANO / SMPL-X, GRAB meshes and
subject templates, and HOT3D object models must be obtained from their
respective providers under their own terms, as described in `README.md`.
