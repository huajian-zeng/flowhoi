# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Union

import torch
from torch import nn
from data_loaders.humanml.data.dataset import MotionDataset, HOIDataset

from diffusion.flow_matching import create_flow_matching
from utils.parser_util import DataOptions, DiffusionOptions, ModelOptions, TrainingOptions
from torch.utils.data import DataLoader


def get_dit_model_class(dit_version: str):
    """Resolve the configured DiT backbone class."""
    if dit_version == 'scene':
        from model.hoi_dit_scene import SceneHOIDiT
        return SceneHOIDiT
    if dit_version == 'base':
        from model.hoi_dit import HOIDiT
        return HOIDiT
    raise ValueError(
        f"Unknown dit_version '{dit_version}': expected 'base' or 'scene'")

FullModelOptions = Union[DataOptions, ModelOptions, DiffusionOptions, TrainingOptions]
Datasets = Union[MotionDataset, HOIDataset]


TEXT_ENCODER_PREFIXES = ('clip_model.', 'text_model.')


def load_model_wo_clip(model: nn.Module, state_dict, allow_new_modules: bool = True):
    """Load weights without text encoders, optionally allowing new conditioning modules."""
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(p) for p in TEXT_ENCODER_PREFIXES)
    }
    n_filtered = len(state_dict) - len(filtered_state_dict)
    if n_filtered > 0:
        print(f'Filtered out {n_filtered} text encoder keys from checkpoint')

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict,
                                                          strict=False)
    text_missing = [k for k in missing_keys if k.startswith(TEXT_ENCODER_PREFIXES)]
    if text_missing:
        print(f'missing {len(text_missing)} text encoder keys (loaded separately)')
    print('missing', [k for k in missing_keys if k not in text_missing])

    tmr_prefixes = ('motion_encoder.', 'text_pooler.', 'tmr_loss.')
    unexpected_keys_filtered = [
        k for k in unexpected_keys
        if not any(k.startswith(p) for p in tmr_prefixes)
    ]
    n_tmr_ignored = len(unexpected_keys) - len(unexpected_keys_filtered)
    if n_tmr_ignored > 0:
        print(f'Ignored {n_tmr_ignored} TMR alignment keys (training-only)')

    assert len(unexpected_keys_filtered) == 0, f'unexpected keys: {unexpected_keys_filtered}'

    allowed_missing_prefixes = ['clip_model.', 'text_model.']
    if allow_new_modules:
        allowed_missing_prefixes.extend([
            'scene_token_encoder.',
            'scene_perceiver.',
            'grasp_encoder.',
            'embed_grasp_pose.',
            'embed_initial_pos.',
            'embed_scene_global.',
            'embed_object.',
            'text_token_encoder.',
            'adaln_cond_proj.',
            'pos_proj.',
        ])

    unexpected_missing = [
        k for k in missing_keys
        if not any(k.startswith(p) for p in allowed_missing_prefixes)
    ]

    if unexpected_missing:
        raise AssertionError(f'unexpected missing keys: {unexpected_missing}')


def create_model_and_diffusion(args: FullModelOptions, data: DataLoader):
    if args.arch.startswith('dit'):
        dit_version = getattr(args, 'dit_version', 'base')
        model_cls = get_dit_model_class(dit_version)
        print(f'Using DiT backbone: {dit_version}')
        model = model_cls(**get_model_args(args, data))
    else:
        raise ValueError(
            f"Unsupported arch '{args.arch}': this release ships only the DiT "
            f"architectures used by the published checkpoints")

    if not getattr(args, 'use_flow_matching', False):
        raise ValueError('This release supports flow-matching checkpoints only')
    diffusion = create_flow_matching_diffusion(args, data)

    return model, diffusion


def get_model_args(args: FullModelOptions, data: DataLoader):
    """Build text-conditioned model arguments for the selected dataset and representation."""
    cond_mode = 'no_cond' if args.unconstrained else 'text'

    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset in ['grab', 'hot3d', 'egodex']:
        data_rep = 'hml_vec'
        nfeats = 1
        if getattr(args, 'wrist_only', False):
            njoints = 27
        elif args.drop_redundant:
            njoints =  81
        else:
            njoints = 117
            if args.use_contacts:
                njoints += 42

    if args.traj_only:
        if args.hands_only or args.pre_grasp:
            njoints = 108
        elif args.obj_only:
            njoints = 9
        else:
            njoints = 99
        nfeats = 1

    two_head = 'two_head' in args.arch

    return {
        'modeltype': '',
        'njoints': njoints,
        'nfeats': nfeats,
        'translation': True,
        'pose_rep': 'rot6d',
        'glob': True,
        'glob_rot': True,
        'latent_dim': args.latent_dim,
        'ff_size': args.ff_size,
        'num_layers': args.layers,
        'num_heads': getattr(args, 'heads', 4),
        'dropout': getattr(args, 'dropout', 0.1),
        'activation': "gelu",
        'data_rep': data_rep,
        'cond_mode': cond_mode,
        'cond_mask_prob': args.cond_mask_prob,
        'arch': args.arch,
        'emb_trans_dec': args.emb_trans_dec,
        'dataset': args.dataset,
        'two_head': two_head,
        'dim_mults': args.dim_mults,
        'adagn': args.unet_adagn,
        'zero': args.unet_zero,
        'unet_out_mult': args.out_mult,
        'tf_out_mult': args.out_mult,
        'obj_enc': args.obj_enc,
        'obj_enc_dim': getattr(args, 'obj_enc_dim', 3072),
        'motion_enc_frames': args.motion_enc_frames,
        'text_feature_dim': getattr(args, 'text_feature_dim', None),
        'use_initial_pos': getattr(args, 'use_initial_pos', False),
        'initial_pos_dim': getattr(args, 'initial_pos_dim', 117),
        'pos_embed_dim': getattr(args, 'pos_embed_dim', 512),
        'use_scene_local': getattr(args, 'use_scene_local', False),
        'scene_feature_dim': getattr(args, 'scene_feature_dim', 771),
        'scene_semantic_dim': getattr(args, 'scene_semantic_dim', 768),
        'cross_attn_layers': getattr(args, 'cross_attn_layers', 'all'),
        'max_scene_points': getattr(args, 'max_scene_points', 10000),
        'use_scene_xyz': getattr(args, 'use_scene_xyz', False),
        'use_scene_perceiver': getattr(args, 'use_scene_perceiver', True),
        'perceiver_num_latents': getattr(args, 'perceiver_num_latents', 64),
        'perceiver_num_layers': getattr(args, 'perceiver_num_layers', 2),
        'perceiver_num_heads': getattr(args, 'perceiver_num_heads', 8),
        'use_grasp_condition': getattr(args, 'use_grasp_condition', False),
        'grasp_feature_dim': getattr(args, 'grasp_feature_dim', 108),
        'use_scene_global': getattr(args, 'use_scene_global', False),
        'scene_size': getattr(args, 'scene_size', 48),
        'scene_channels': getattr(args, 'scene_channels', 24),
        'vit_dim': getattr(args, 'vit_dim', 1024),
        'vit_depth': getattr(args, 'vit_depth', 6),
        'vit_heads': getattr(args, 'vit_heads', 16),
        'vit_mlp_dim': getattr(args, 'vit_mlp_dim', 2048),
        'use_bps': getattr(args, 'use_bps', False),
        'bps_dim': getattr(args, 'bps_dim', 3072),
        'bps_hidden_dim': getattr(args, 'bps_hidden_dim', 256),
        'use_pose_cond': getattr(args, 'use_pose_cond', False),
        'pose_cond_type': getattr(args, 'pose_cond_type', 'full'),
        'grasp_frame': getattr(args, 'grasp_frame', 49),
        'num_text_tokens': getattr(args, 'num_text_tokens', 4),
        't5_model_name': getattr(args, 't5_model_name', 'google/flan-t5-large'),
        'text_max_length': getattr(args, 'text_max_length', 77),
        'use_tmr_alignment': getattr(args, 'use_tmr_alignment', False),
        'tmr_align_dim': getattr(args, 'tmr_align_dim', 512),
        'tmr_temperature': getattr(args, 'tmr_temperature', 0.7),
        'tmr_loss_weight': getattr(args, 'tmr_loss_weight', 0.1),
        'tmr_motion_encoder_layers': getattr(args, 'tmr_motion_encoder_layers', 4),
        'use_concerto_grid': getattr(args, 'use_concerto_grid', False),
        'concerto_feature_dim': getattr(args, 'concerto_feature_dim', 1536),
        'max_concerto_points': getattr(args, 'max_concerto_points', 25000),
        'concerto_fps_points': getattr(args, 'concerto_fps_points', 2000),
        'concerto_use_fps': getattr(args, 'concerto_use_fps', False),
        'concerto_perceiver_num_latents': getattr(args, 'concerto_perceiver_num_latents', 256),
        'concerto_perceiver_num_layers': getattr(args, 'concerto_perceiver_num_layers', 2),
        'use_scene_fusion': getattr(args, 'use_scene_fusion', False),
        'scene_fusion_spatial_dim': getattr(args, 'scene_fusion_spatial_dim', 1536),
        'scene_fusion_semantic_dim': getattr(args, 'scene_fusion_semantic_dim', 768),
        'scene_fusion_hidden_dim': getattr(args, 'scene_fusion_hidden_dim', 512),
        'scene_fusion_main_heads': getattr(args, 'scene_fusion_main_heads', 8),
        'scene_fusion_main_layers': getattr(args, 'scene_fusion_main_layers', 2),
        'scene_fusion_aux_heads': getattr(args, 'scene_fusion_aux_heads', 4),
        'scene_fusion_aux_layers': getattr(args, 'scene_fusion_aux_layers', 1),
        'scene_fusion_xyz_freqs': getattr(args, 'scene_fusion_xyz_freqs', 16),
        'scene_fusion_xyz_enc_dim': getattr(args, 'scene_fusion_xyz_enc_dim', 64),
        'scene_perceiver_num_latents': getattr(args, 'scene_perceiver_num_latents', 256),
        'scene_perceiver_num_layers': getattr(args, 'scene_perceiver_num_layers', 2),
        'scene_perceiver_num_heads': getattr(args, 'scene_perceiver_num_heads', 8),
    }


def create_flow_matching_diffusion(args: FullModelOptions, data: DataLoader):
    """Create a flow-matching sampler with DDPM-compatible loss options."""
    del data
    num_sampling_steps = getattr(args, 'num_sampling_steps', 50)
    sigma_min = getattr(args, 'sigma_min', 1e-4)

    return create_flow_matching(
        num_sampling_steps=num_sampling_steps,
        sigma_min=sigma_min,
        traj_extra_weight=getattr(args, 'traj_extra_weight', 1.0),
        drop_redundant=getattr(args, 'drop_redundant', False),
        lambda_vel=getattr(args, 'lambda_vel', 0.0),
    )


def load_saved_model(model, model_path, use_avg: bool = True):
    """Load average checkpoint weights when available."""
    state_dict = torch.load(model_path, map_location='cpu')
    if use_avg and 'model_avg' in state_dict.keys():
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model')
    load_model_wo_clip(model, state_dict)
    return model




