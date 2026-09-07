# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import sys
from dataclasses import dataclass, field, fields
from typing import Tuple

from utils.hfargparse import HfArgumentParser


@dataclass
class BaseOptions:
    """Common runtime options."""

    cuda: bool = field(
        default=True, metadata={"help": "Use cuda device, otherwise use CPU."})
    device: int = field(default=0, metadata={"help": "Device id to use."})
    seed: int = field(default=42, metadata={"help": "For fixing random seed."})

@dataclass
class DiffusionOptions:
    """Diffusion and flow-matching options."""

    diffusion_steps: int = field(
        default=1000,
        metadata={
            "help": "Number of diffusion steps (denoted T in the paper)"
        })
    use_flow_matching: bool = field(
        default=False,
        metadata={"help": "Use Flow Matching instead of DDPM"})
    num_sampling_steps: int = field(
        default=50,
        metadata={"help": "Number of sampling steps for Flow Matching (default: 50)"})
    sigma_min: float = field(
        default=1e-4,
        metadata={"help": "Minimum noise level for Flow Matching stability"})


@dataclass
class ModelOptions:
    """Model architecture and conditioning options."""

    arch: str = field(
        default='trans_enc',
        metadata={"help": "Architecture types as reported in the paper."})
    dit_version: str = field(
        default='base',
        metadata={
            "help": "DiT backbone: 'base' or 'scene' (hybrid scene conditioning)",
            "choices": ['base', 'scene']
        })
    emb_trans_dec: bool = field(
        default=False,
        metadata={
            "help":
            "For trans_dec architecture only, if true, will inject condition as a class token (in addition to cross-attention)."
        })
    layers: int = field(default=8, metadata={"help": "Number of layers."})
    latent_dim: int = field(default=512,
                            metadata={"help": "Transformer/GRU width."})
    ff_size: int = field(default=1024,
                         metadata={"help": "Transformer feedforward size."})
    dim_mults: Tuple[float] = field(
        default=(2, 2, 2, 2), metadata={"help": "Unet channel multipliers."})
    unet_adagn: bool = field(
        default=True, metadata={"help": "Unet adaptive group normalization."})
    unet_zero: bool = field(
        default=True, metadata={"help": "Unet zero weight initialization."})
    out_mult: bool = field(
        default=1,
        metadata={"help": "UNET/TF large variation's feature multiplier."})
    cond_mask_prob: float = field(
        default=.1,
        metadata={
            "help":
            "The probability of masking the condition during training. For classifier-free guidance learning."
        })
    lambda_vel: float = field(default=0.0,
                              metadata={"help": "Joint velocity loss."})
    unconstrained: bool = field(
        default=False,
        metadata={
            "help":
            "Model is trained unconditionally. That is, it is constrained by neither text nor action. Currently tested on HumanAct12 only."
        })

    use_initial_pos: bool = field(
        default=False,
        metadata={"help": "Use initial position (frame 0) as condition for DiT model."})
    initial_pos_dim: int = field(
        default=117,
        metadata={"help": "Dimension of initial position (117 for full representation)."})
    pos_embed_dim: int = field(
        default=512,
        metadata={"help": "Embedding dimension for initial position encoder."})

    use_scene_local: bool = field(
        default=False,
        metadata={"help": "Use scene point cloud for local cross-attention conditioning (fine-grained spatial)."})
    scene_feature_dim: int = field(
        default=771,
        metadata={"help": "Feature dimension of scene points (771 = 3 xyz + 768 langfeat for HOT3D)."})
    scene_semantic_dim: int = field(
        default=768,
        metadata={"help": "Semantic feature dimension from SceneSplat (768 for language features)."})
    use_scene_xyz: bool = field(
        default=False,
        metadata={"help": "Whether to use xyz coordinates in scene cross-attention (False = semantic only)."})
    cross_attn_layers: str = field(
        default="all",
        metadata={"help": "Which layers to apply cross-attention: 'all', 'even', 'odd', or comma-separated indices."})
    max_scene_points: int = field(
        default=10000,
        metadata={"help": "Maximum number of scene points to use."})

    use_scene_perceiver: bool = field(
        default=True,
        metadata={"help": "Use Perceiver to compress N scene points to K tokens (preserves semantics, 12x faster)."})
    perceiver_num_latents: int = field(
        default=64,
        metadata={"help": "Number of latent tokens K for scene compression (default: 64)."})
    perceiver_num_layers: int = field(
        default=2,
        metadata={"help": "Number of Perceiver layers (cross-attn + self-attn + FFN)."})
    perceiver_num_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in Perceiver."})

    scene_downsample: int = field(
        default=10000,
        metadata={"help": "Use pre-downsampled scene data with this many points. 0 = use original files."})
    use_precomputed_local_scenes: bool = field(
        default=False,
        metadata={"help": "Use pre-computed local scene point clouds."})
    local_scenes_path: str = field(
        default=None,
        metadata={"help": "Path to pre-computed local scenes (default: data_root/local_scenes_<downsample>)."})

    use_grasp_condition: bool = field(
        default=False,
        metadata={"help": "Use grasp pose (last frame) as AdaLN condition for Stage 2 (interaction model)."})
    grasp_feature_dim: int = field(
        default=108,
        metadata={"help": "Feature dimension of grasp pose (108 = hand trajectory excluding object)."})

    use_scene_global: bool = field(
        default=False,
        metadata={"help": "Use ViT-based global scene encoding on occupancy map (SceneMI style, AdaLN injection)."})
    scene_size: int = field(
        default=48,
        metadata={"help": "Occupancy map spatial resolution (default: 48x48)."})
    scene_channels: int = field(
        default=24,
        metadata={"help": "Occupancy map height channels (default: 24 Z layers)."})
    vit_dim: int = field(
        default=1024,
        metadata={"help": "ViT internal embedding dimension."})
    vit_depth: int = field(
        default=6,
        metadata={"help": "ViT number of transformer layers."})
    vit_heads: int = field(
        default=16,
        metadata={"help": "ViT number of attention heads."})
    vit_mlp_dim: int = field(
        default=2048,
        metadata={"help": "ViT MLP hidden dimension."})
    occ_maps_path: str = field(
        default=None,
        metadata={"help": "Path to pre-computed occupancy maps (default: data_root/occ_maps_<res>)."})

    use_bps: bool = field(
        default=False,
        metadata={"help": "Use BPS (Basis Point Set) encoding for object representation."})
    bps_dim: int = field(
        default=3072,
        metadata={"help": "BPS feature dimension (default: 3072 = 1024 points x 3 coords)."})
    bps_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dimension for BPS encoder."})

    pose_cond_type: str = field(
        default="full",
        metadata={"help": "Type of pose condition: 'full' (all dims), 'hands_only', 'traj_only'."})
    use_pose_cond: bool = field(
        default=False,
        metadata={"help": "Use pose conditioning (initial frame pose) for the model."})

    grasp_frame: int = field(
        default=49,
        metadata={"help": "Frame index considered as grasp frame (default: 49)."})

    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate for transformer layers."})
    heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in transformer."})

    num_text_tokens: int = field(
        default=4,
        metadata={"help": "Number of text tokens for cross-attention."})

    use_concerto_grid: bool = field(
        default=False,
        metadata={"help": "Use Concerto grid features for scene conditioning."})
    concerto_feature_dim: int = field(
        default=1536,
        metadata={"help": "Concerto feature dimension (default: 1536)."})
    max_concerto_points: int = field(
        default=25000,
        metadata={"help": "Max points for Perceiver padding (covers 99%% of sequences)."})
    concerto_fps_points: int = field(
        default=2000,
        metadata={"help": "Number of FPS sampled points for direct cross-attention."})
    concerto_use_fps: bool = field(
        default=False,
        metadata={"help": "Use FPS sampling instead of Perceiver padding."})
    concerto_perceiver_num_latents: int = field(
        default=256,
        metadata={"help": "Number of Perceiver latent tokens for Concerto encoding."})
    concerto_perceiver_num_layers: int = field(
        default=2,
        metadata={"help": "Number of Perceiver layers for Concerto encoding."})

    use_scene_fusion: bool = field(
        default=False,
        metadata={"help": "Enable gated scene fusion."})
    use_semantic_features: bool = field(
        default=False,
        metadata={"help": "Load SceneSplat semantic features (aligned_langfeat.npy)."})
    semantic_feature_dim: int = field(
        default=768,
        metadata={"help": "Semantic feature dimension (SceneSplat)."})
    semantic_features_path: str = field(
        default='./dataset/HOT3D_HANDS/semantic_features',
        metadata={"help": "Path to semantic features directory."})
    scene_fps_path: str = field(
        default=None,
        metadata={"help": "Precomputed FPS scene bundle (ships with the data package). "
                          "Replaces the raw Concerto/SceneSplat readers when set."})
    use_scene_bank: bool = field(
        default=False,
        metadata={"help": "Keep per-scene point features resident on the GPU and gather them "
                          "by index, instead of streaming them through the data loader."})
    feed_scene_inputs: bool = field(
        default=False,
        metadata={"help": "Feed scene inputs (occupancy map, Concerto/SceneSplat tokens) to the "
                          "model at sampling time. Off by default so checkpoints trained before "
                          "this flag reproduce their published numbers exactly."})
    scene_fusion_spatial_dim: int = field(
        default=1536,
        metadata={"help": "Spatial feature dimension (Concerto)."})
    scene_fusion_semantic_dim: int = field(
        default=768,
        metadata={"help": "Semantic feature dimension for fusion."})
    scene_fusion_hidden_dim: int = field(
        default=512,
        metadata={"help": "Hidden dimension for BiDirectional fusion."})
    scene_fusion_main_heads: int = field(
        default=8,
        metadata={"help": "Number of heads for main path (spatial->semantic)."})
    scene_fusion_main_layers: int = field(
        default=2,
        metadata={"help": "Number of layers for main path."})
    scene_fusion_aux_heads: int = field(
        default=4,
        metadata={"help": "Number of heads for auxiliary path (semantic->spatial)."})
    scene_fusion_aux_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for auxiliary path."})
    scene_fusion_xyz_freqs: int = field(
        default=16,
        metadata={"help": "Number of Fourier frequencies for XYZ encoding."})
    scene_fusion_xyz_enc_dim: int = field(
        default=64,
        metadata={"help": "Output dimension of XYZ positional encoding."})


@dataclass
class DataOptions:
    """Dataset and representation options."""

    dataset: str = field(default='grab',
                         metadata={
                             "help": "Dataset name (choose from list).",
                             "choices":
                             ['grab', 'hot3d', 'egodex']
                         })
    data_dir: str = field(
        default="",
        metadata={
            "help":
            "If empty, will use defaults according to the specified dataset."
        })
    data_repr: str = field(
        default="representation_full",
        metadata={
            "help":
            "If empty, will use defaults according to the specified dataset."
        })
    mean_name: str = field(
        default="Mean_grab_full.npy",
        metadata={
            "help":
            "If empty, will use defaults according to the specified dataset."
        })
    std_name: str = field(
        default="Std_grab_full.npy",
        metadata={
            "help":
            "If empty, will use defaults according to the specified dataset."
        })
    proj_matrix_name: str = field(
        default="",
        metadata={
            "help":
            "If empty, will use defaults according to the specified dataset."
        })
    mano_model_path: str = field(
        default='./assets/smplx',
        metadata={
            "help":
            "Directory holding the MANO / SMPL-X model files"
        })
    sbj_model_path: str = field(
        default='./assets',
        metadata={
            "help":
            "Directory holding the GRAB per-subject hand templates, in female/ "
            "and male/ subfolders"
        })
    obj_model_path: str = field(
        default='./assets/contact_meshes',
        metadata={
            "help":
            "Directory holding the GRAB object meshes. HOT3D object meshes are "
            "read from ./assets/hot3d_assets instead"
        })
    split_set: str = field(default='objects_unseen',
                            metadata={
                                "help":
                                "Which split to use; resolves to "
                                "<data_root>/<split>_<split_set>.txt"
                            })
    abs_3d: bool = field(default=False, metadata={"help": "Use absolute 3D."})
    hands_only: bool = field(default=False, metadata={"help": "Use hands_only to omit object"})
    obj_only: bool = field(default=False, metadata={"help": "Use obj_only to omit hands"})
    pre_grasp: bool = field(default=False, metadata={"help": "Use pre_grasp to only learn approach and grasp"})
    wrist_only: bool = field(default=False, metadata={"help": "Use wrist_only for 27D representation (wrist pos/rot + object pose)"})
    use_contacts: bool = field(default=False, metadata={"help": "Use binary contact prediction."})
    obj_enc: bool = field(default=False, metadata={"help": "Use bps based object encoding"})
    traj_only: bool = field(default=False,metadata={"help": "Use trajectory model."})
    use_pca: bool = field(default=True, metadata={"help": "Use PCA for hand representation."})

    use_random_proj: bool = field(default=False,
                                  metadata={"help": "Use random projection."})
    random_proj_scale: float = field(
        default=10.0, metadata={"help": "Random projection scale."})
    std_scale_shift: Tuple[float] = field(
        default=(1.0, 0.0),
        metadata={"help": "Adjusting the std by scale and shift."})
    drop_redundant: bool = field(
        default=False,
        metadata={"help": "Drop redundant joint information. "
                  "Keep only 4 (root) + 21*3 joints position"})
    text_detailed: bool = field(
        default=True,
        metadata={"help": "If True, use detailed text annotations instead of simple ones."})

    text_feature_dim: int = field(
        default=None,
        metadata={"help": "Dimension of the text features the model consumes; defaults to "
                  "the encoder's own width (1024 for T5-large)."})
    t5_model_name: str = field(
        default="google/flan-t5-large",
        metadata={"help": "T5 model name for 't5' encoder type."})
    text_max_length: int = field(
        default=77,
        metadata={"help": "Maximum text sequence length for T5/CLIP encoders."})

    use_tmr_alignment: bool = field(
        default=False,
        metadata={"help": "Enable TMR contrastive alignment loss."})
    tmr_align_dim: int = field(
        default=512,
        metadata={"help": "Dimension of TMR alignment embeddings."})
    tmr_temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature for TMR contrastive loss."})
    tmr_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for TMR alignment loss."})

    scene_data_root: str = field(
        default="./dataset/HOT3D_HANDS/scene_data",
        metadata={"help": "Root directory containing the per-recording scene data."})
    scene_data_max_points: int = field(
        default=10000,
        metadata={"help": "Maximum number of scene points to load per sequence."})

@dataclass
class TrainingOptions:
    """Training and evaluation-loop options."""

    save_dir: str = field(
        default=None,
        metadata={"help": "Path to save checkpoints and results."})
    overwrite: bool = field(
        default=False,
        metadata={
            "help": "If True, will enable to use an already existing save_dir."
        })
    batch_size: int = field(default=64,
                            metadata={"help": "Batch size during training."})
    train_platform_type: str = field(
        default='WandbPlatform',
        metadata={
            "help":
            "Choose platform to log results. NoPlatform means no logging.",
            "choices":
            ['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandbPlatform']
        })
    lr: float = field(default=1e-4, metadata={"help": "Learning rate."})
    weight_decay: float = field(default=0.,
                                metadata={"help": "Optimizer weight decay."})
    grad_clip: float = field(default=0, metadata={"help": "Gradient clip."})
    use_fp16: bool = field(default=False, metadata={"help": "Use fp16."})
    avg_model_beta: float = field(
        default=0, metadata={"help": "Average model beta; 0 = disabled."})
    adam_beta2: float = field(default=0.999, metadata={"help": "Adam beta2."})
    lr_anneal_steps: int = field(
        default=0, metadata={"help": "Number of learning rate anneal steps."})
    eval_batch_size: int = field(
        default=32,
        metadata={
            "help":
            "Batch size for the evaluation loop during training."
        })
    eval_during_training: bool = field(
        default=True,
        metadata={"help": "If True, will run evaluation during training."})
    eval_full_metrics: bool = field(
        default=False,
        metadata={
            "help": "If True, compute full metrics (diversity, MSE, etc.) during training evaluation. "
                    "If False (default), only compute validation loss for speed."
        })
    eval_num_samples: int = field(
        default=-1,
        metadata={
            "help": "If -1, will use all samples in the specified split."
        })
    early_stop_patience: int = field(
        default=5,
        metadata={
            "help": "Number of evaluations without improvement before stopping. Set to 0 to disable early stopping."
        })
    early_stop_metric: str = field(
        default='val_loss',
        metadata={
            "help": "Metric to monitor for early stopping. Options: val_loss, mse, diversity_gen, smoothness, gt_correlation, mse_corr_combined"
        })
    early_stop_min_delta: float = field(
        default=0.0,
        metadata={
            "help": "Minimum change in monitored metric to qualify as an improvement."
        })
    early_stop_mode: str = field(
        default='min',
        metadata={
            "help": "Whether to minimize ('min') or maximize ('max') the early_stop_metric."
        })
    log_interval: int = field(default=1_000,
                              metadata={"help": "Log losses each N steps"})
    save_interval: int = field(
        default=10_000,
        metadata={"help": "Save checkpoints and run evaluation each N steps"})
    permanent_save_interval: int = field(
        default=50_000,
        metadata={"help": "Permanently save checkpoints (not deleted) at this interval"})
    add_timestamp: bool = field(
        default=True,
        metadata={"help": "Add timestamp to save_dir to distinguish different runs."})
    num_steps: int = field(
        default=1_200_000,
        metadata={
            "help": "Training will stop after the specified number of steps."
        })
    num_frames: int = field(
        default=60,
        metadata={
            "help":
            "Limit for the maximal number of frames. In HOIDataset and KIT this field is ignored."
        })
    resume_checkpoint: str = field(
        default="",
        metadata={
            "help":
            "If not empty, will start from the specified checkpoint (path to model###.pt file)."
        })
    traj_extra_weight: float = field(
        default=1.0, metadata={"help": "Trajectory extra weight."})

@dataclass
class SamplingOptions:
    """Checkpoint sampling options."""

    model_path: str = field(
        default='',
        metadata={"help": "Path to model####.pt file to be sampled."})
    output_dir: str = field(
        default='',
        metadata={
            "help":
            "Path to results dir (auto created by the script). If empty, will create dir in parallel to checkpoint."
        })
    grasp_model_path: str = field(
        default='./save/egodex_grasp/model000500000.pt',
        metadata={"help": "Path to model####.pt file to be sampled."})
    num_samples: int = field(
        default=10,
        metadata={
            "help":
            "Maximum selected conditions; with eval_entire_set, the generation batch size (tail batches are kept)."
        })
    num_repetitions: int = field(
        default=1,
        metadata={
            "help": "Number of repetitions, per sample (text prompt/action)"
        })
    guidance_param: float = field(
        default=1.0,
        metadata={
            "help":
            "Classifier-free guidance scale. Default 1 keeps the released sampler's "
            "conditional prediction; values above 1 explicitly enable CFG and change results."
        })

@dataclass
class GenerateOptions:
    """Generation request options."""

    max_frames_grasp: int = field(
        default=50,
        metadata={
            "help":
            "Grasp phase length in frames. Released checkpoints require 50 (grasp keyframe 49)."
        })
    max_frames_interaction: int = field(
        default=200,
        metadata={
            "help":
            "Interaction length in frames, from 50 to 200 inclusive."
        })
    classifier_scale: float = field(
        default=10.0,
        metadata={
            "help":
            "A scaling factor for the gradient from the classifier. Use the same scale for both model in two-staged case"
        })
    do_inpaint: bool = field(
        default=False, metadata={"help": "If True, will perform inpainting."})
    gen_two_stages: bool = field(
        default=False, metadata={"help": "If True, generate only the grasping stage."})
    gen_mse_loss: bool = field(
        default=True, metadata={"help": "If True, use MSE loss for classifier. Otherwise, use L1 loss"})
    gen_split: str = field(default='test',
                            metadata={
                                "help":
                                "Which split to evaluate on during training.",
                                "choices": ['train', 'test']
                            })
    no_subsequence_inpainting: bool = field(
        default=False, metadata={"help": "If True, no inpainting."})
    eval_entire_set: bool = field(
        default=False, metadata={"help": "Generate all paired dataset conditions, including the final incomplete batch."})
    p2p_impute: bool = field(
        default=True, metadata={"help": "If True, use point-to-point guidance for trajectory."})
    obj_enc: bool = field(
        default=False, metadata={"help": "If True, use bps object encoding."})
    random_order: bool = field(default=False, metadata={"help": "whether or not to use shuffling of data."})
    guidance: bool = field(default=False,metadata={"help": "whether or not to use guidance."})
    inpainting: bool = field(default=False, metadata={"help": "whether or not to use inpainting during sampling. If False, inpainted_motion is only passed as condition."})
    text_detailed: bool = field(
        default=False, metadata={"help": "If True, use the new annotations of GRAB."})
    motion_enc_frames: int = field(
        default=0, metadata={"help": "If True, use bps object encoding."})
    comment: str = field(
        default='',
        metadata={
            "help": "Optional comment prefix for the output folder name (e.g., 'full_set' will create 'full_set_samples_1stage_...')"
        })
    impute_until_override: float = field(
        default=None,
        metadata={
            "help": "Override the impute_until threshold for both stages. "
                    "Values > 0.9 and < 1.0 tighten the inpainting for "
                    "out-of-distribution inputs. Default None uses "
                    "dataset-specific defaults."
        })


@dataclass
class EvaluationOptions:
    """Evaluation options."""

    model_path: str = field(
        default='',
        metadata={"help": "Path to model####.pt file to be sampled."})
    guidance_param: float = field(
        default=1.0,
        metadata={
            "help":
            "Classifier-free guidance scale. Default 1 keeps the released sampler's "
            "conditional prediction; values above 1 explicitly enable CFG and change results."
        })
    impute_until: int = field(default=None, metadata={"help": "impute until"})


@dataclass
class FullModelArgs(BaseOptions, DataOptions, ModelOptions, DiffusionOptions,
                   TrainingOptions, SamplingOptions, GenerateOptions):
    """Combined options for full-model workflows."""

    pass


@dataclass
class TrainArgs(BaseOptions, DataOptions, ModelOptions, DiffusionOptions,
                TrainingOptions):
    """Combined options for training."""

    pass


def train_args(base_cls=TrainArgs):
    """Parse training arguments with the selected options class."""
    from datetime import datetime

    parser = HfArgumentParser(base_cls)
    args: TrainArgs = parser.parse_args_into_dataclasses()[0]

    if getattr(args, 'add_timestamp', True) and args.save_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = f"{args.save_dir}_{timestamp}"

    return args


@dataclass
class GenerateArgs(BaseOptions, DataOptions, ModelOptions, DiffusionOptions,
                   TrainingOptions, SamplingOptions, GenerateOptions):
    """Combined options for generation."""

    pass


def generate_args(model_path=None) -> GenerateArgs:
    parser = HfArgumentParser(GenerateArgs)
    args = parse_and_load_from_model(parser, model_path)
    if args.max_frames_grasp != 50:
        parser.error('--max_frames_grasp must be 50 for the released checkpoints '
                     '(the grasp keyframe is index 49).')
    if not 50 <= args.max_frames_interaction <= 200:
        parser.error('--max_frames_interaction must be between 50 and 200 '
                     'to include the grasp keyframe at index 49.')
    if args.grasp_frame != 49:
        parser.error('--grasp_frame must be 49 for the released two-stage sampler.')
    return args


def parse_and_load_from_model(parser: HfArgumentParser,
                              model_path=None):
    '''Load checkpoint options with function and CLI overrides taking precedence.'''
    args: FullModelArgs = parser.parse_args()

    args_to_overwrite = []
    for cls in [DataOptions, ModelOptions, DiffusionOptions, TrainingOptions]:
        for cls_field in fields(cls):
            args_to_overwrite.append(cls_field.name)

    additional_fields = ['motion_enc_frames']
    args_to_overwrite.extend(additional_fields)

    if model_path is not None:
        print(" - model_path is given in the function call. Cmd path, if any, will be ignored.")
        args.model_path = model_path
    else:
        model_path = args.model_path

    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), f'Arguments json file was not found! {args_path}'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    explicit_fields = set()
    for token in sys.argv[1:]:
        if token == '--':
            break
        if token.startswith('--'):
            parsed_option = parser._parse_optional(token)
            if parsed_option is not None and parsed_option[0] is not None:
                explicit_fields.add(parsed_option[0].dest)
    cli_overrides = {a: getattr(args, a) for a in args_to_overwrite
                     if a in explicit_fields}

    defaulted = []
    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])
        elif 'cond_mode' in model_args:
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)
        else:
            defaulted.append(a)

    if defaulted:
        print(f'{len(defaulted)} option(s) absent from {args_path}, keeping defaults: '
              + ', '.join(defaulted))

    for a, v in cli_overrides.items():
        print(f'Command-line override: {a} = {v}')
        setattr(args, a, v)

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args
