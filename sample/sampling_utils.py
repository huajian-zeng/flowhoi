"""Shared utilities for unified sampling scripts."""
import os
import json
from datetime import datetime
from typing import Tuple, List, Dict

import numpy as np
import torch

from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.utils.data_utils import OBJECT_LIST_NEW


def is_hot3d_dataset(args) -> bool:
    """Check if the dataset is HOT3D."""
    dataset_name = getattr(args, 'dataset', 'grab')
    return 'hot3d' in dataset_name.lower()


def load_dataset(args, max_frames: int, n_frames: int, shuffle: bool = None):
    """Load a dataset, feeding scene inputs only when explicitly enabled."""
    if shuffle is None:
        shuffle = getattr(args, 'random_order', False)

    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split=args.gen_split,
        split_set=args.split_set,
        hml_mode='text_only',
        use_abs3d=args.abs_3d,
        traject_only=args.traj_only,
        use_random_projection=args.use_random_proj,
        random_projection_scale=args.random_proj_scale,
        augment_type='none',
        std_scale_shift=args.std_scale_shift,
        drop_redundant=args.drop_redundant,
        use_contacts=getattr(args, 'use_contacts', False),
        data_repr=args.data_repr,
        mean_name=args.mean_name,
        std_name=args.std_name,
        proj_matrix_name=args.proj_matrix_name,
        pre_grasp=args.pre_grasp,
        hands_only=args.hands_only,
        obj_only=args.obj_only,
        text_detailed=getattr(args, 'text_detailed', False),
        wrist_only=getattr(args, 'wrist_only', False),
        motion_enc_frames=getattr(args, 'motion_enc_frames', 0),
    )

    if getattr(args, 'feed_scene_inputs', False):
        conf.use_scene_global = getattr(args, 'use_scene_global', False)
        conf.occ_maps_path = getattr(args, 'occ_maps_path', None)
        conf.use_concerto_grid = getattr(args, 'use_concerto_grid', False)
        conf.concerto_feature_dim = getattr(args, 'concerto_feature_dim', 1536)
        conf.max_concerto_points = getattr(args, 'max_concerto_points', 25000)
        conf.concerto_fps_points = getattr(args, 'concerto_fps_points', 2000)
        conf.concerto_use_fps = getattr(args, 'concerto_use_fps', False)
        conf.use_semantic_features = getattr(args, 'use_semantic_features', False)
        conf.semantic_feature_dim = getattr(args, 'semantic_feature_dim', 768)
        conf.semantic_features_path = getattr(args, 'semantic_features_path',
                                              './dataset/HOT3D_HANDS/semantic_features')
        conf.scene_fps_path = getattr(args, 'scene_fps_path', None)
        conf.use_scene_bank = getattr(args, 'use_scene_bank', False)
        conf.use_scene_local = getattr(args, 'use_scene_local', False)
        conf.scene_data_root = getattr(args, 'scene_data_root', conf.scene_data_root)
        print(f"[Scene inputs] global={conf.use_scene_global} concerto={conf.use_concerto_grid} "
              f"semantic={conf.use_semantic_features} fps_bundle={conf.scene_fps_path}")

    data = get_dataset_loader(conf, shuffle=shuffle)
    data.fixed_length = n_frames
    return data


_SCENE_BANK_CACHE = {}


def attach_scene_inputs(args, model_kwargs, device):
    """Attach required scene features and fail if configured inputs are absent."""
    if not getattr(args, 'feed_scene_inputs', False):
        return model_kwargs

    y = model_kwargs['y']
    if getattr(args, 'use_scene_bank', False):
        bundle = getattr(args, 'scene_fps_path', None)
        if bundle not in _SCENE_BANK_CACHE:
            from data_loaders.scene_bank import SceneBank
            _SCENE_BANK_CACHE[bundle] = SceneBank(bundle, device)
        _SCENE_BANK_CACHE[bundle].fill(y)

    if getattr(args, 'use_scene_fusion', False):
        if y.get('concerto_features') is None or y.get('semantic_features') is None:
            raise RuntimeError(
                'Model is configured with use_scene_fusion but the batch carries no '
                'concerto/semantic features. The local scene branch would be skipped '
                'silently. Check scene_fps_path / use_scene_bank / use_concerto_grid.')
    if getattr(args, 'use_scene_global', False):
        if y.get('occ_map') is None:
            raise RuntimeError(
                'Model is configured with use_scene_global but the batch carries no '
                'occ_map. The global scene token would be skipped silently. Check '
                'occ_maps_path.')
    return model_kwargs


def setup_output_path(args, niter: str, stage_name: str = "1stage") -> str:
    """Setup output directory path."""
    out_path = args.output_dir

    if out_path == '':
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        is_flow = getattr(args, 'use_flow_matching', False)
        mode = 'flow' if is_flow else 'ddpm'
        folder_name = f'samples_{mode}_{stage_name}_{niter}_{timestamp}'
        if args.comment:
            folder_name = f'{args.comment}_{folder_name}'
        out_path = os.path.join(os.path.dirname(args.model_path), folder_name)

    os.makedirs(out_path, exist_ok=True)
    print(f"Output directory: {out_path}")

    return out_path


def save_args(args, out_path: str):
    """Save arguments to JSON file."""
    args_path = os.path.join(out_path, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)


def fix_ambiguous_object_names(texts_pre: List[str], texts: List[str], dataset: str = "grab") -> Tuple[List[str], List[str]]:
    """Resolve ambiguous GRAB object names across pre-grasp and post-grasp text."""
    if dataset.lower() != "grab":
        return texts_pre, texts

    for i in range(len(texts)):
        obj_name = [s for s in OBJECT_LIST_NEW if s in texts_pre[i]]
        obj_name_post = [s for s in OBJECT_LIST_NEW if s in texts[i]]

        if len(obj_name) < 1:
            continue

        if len(obj_name) > 1:
            if 'phone' in obj_name:
                obj_name = ['phone']
                obj_name_post = ['phone']
            else:
                obj_name = ['wristwatch']
                obj_name_post = ['wristwatch']

        texts_pre[i] = texts_pre[i].replace(obj_name[0], obj_name_post[0])

    return texts_pre, texts


def get_conditioning_thresholds(
    impute_until_override: float = None,
) -> Tuple[float, float, float]:
    """Return flow-matching conditioning thresholds, where t=0 is noise and t=1 is data."""
    cond_until = 0.98
    impute_until_grasp = 0.9
    impute_until_interaction = 0.9

    if impute_until_override is not None:
        print(f"[Conditioning] Overriding impute_until: "
              f"grasp {impute_until_grasp} -> {impute_until_override}, "
              f"interaction {impute_until_interaction} -> {impute_until_override}")
        cond_until = 0
        impute_until_grasp = impute_until_override
        impute_until_interaction = impute_until_override

    return cond_until, impute_until_grasp, impute_until_interaction


def sample_flow_matching(
    model,
    diffusion,
    shape: Tuple,
    model_kwargs: Dict,
    cond_fn=None,
) -> List[torch.Tensor]:
    """Run flow-matching sampling and return a list of samples."""
    sample = diffusion.p_sample_loop(
        model=model,
        shape=shape,
        noise=None,
        model_kwargs=model_kwargs,
        progress=True,
        cond_fn=cond_fn,
    )

    if isinstance(sample, list):
        return sample
    return [sample]


def save_results(
    out_path: str,
    all_motions: np.ndarray,
    all_text: List[str],
    all_lengths: np.ndarray,
    all_data_id: List,
    all_gt_kf: np.ndarray,
    args,
    n_frames_post: int,
    inference_times: List[float] = None,
    repetition_indices: List[int] = None,
):
    """Save sampled motions, metadata, text, and timing results."""
    result_file = f'results_{args.gen_split}_{args.split_set}.npy'
    npy_path = os.path.join(out_path, result_file)

    print(f"saving results file to [{npy_path}]")

    results = {
        'motion': all_motions[:, :, :n_frames_post],
        'text': all_text,
        'lengths': all_lengths,
        'num_samples': len(set(all_data_id)),
        'num_generated': len(all_motions),
        'requested_num_samples': args.num_samples,
        'num_repetitions': args.num_repetitions,
        'gt_kf': all_gt_kf[:, :n_frames_post] if len(all_gt_kf) > 0 else [],
        'data_id': all_data_id,
    }

    if repetition_indices is not None:
        results['repetition_index'] = repetition_indices
    if not (len(all_motions) == len(all_text) == len(all_lengths) == len(all_data_id)):
        raise ValueError('Generated motions and saved row metadata disagree.')

    is_flow_matching = getattr(args, 'use_flow_matching', False)
    cond_until, impute_until_grasp, impute_until_interaction = get_conditioning_thresholds(
        impute_until_override=getattr(args, 'impute_until_override', None))
    results['config'] = {
        'guidance': getattr(args, 'guidance', False),
        'inpainting': getattr(args, 'inpainting', False),
        'classifier_scale': getattr(args, 'classifier_scale', 10.0),
        'use_flow_matching': is_flow_matching,
        'cond_until': cond_until,
        'impute_until_grasp': impute_until_grasp,
        'impute_until_interaction': impute_until_interaction,
        'num_sampling_steps': getattr(args, 'num_sampling_steps', 50),
        'use_initial_pos': getattr(args, 'use_initial_pos', False),
        'use_grasp_condition': getattr(args, 'use_grasp_condition', False),
    }

    if inference_times:
        results['inference_times'] = inference_times
        results['inference_time_avg'] = np.mean(inference_times)
        results['inference_time_std'] = np.std(inference_times)
        results['throughput'] = 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0.0

    np.save(npy_path, results)

    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print("\n" + "=" * 60)
    print("SAMPLING CONFIGURATION")
    print("=" * 60)
    for key, value in results['config'].items():
        print(f"{key}: {value}")
    print("=" * 60)

    return npy_path
