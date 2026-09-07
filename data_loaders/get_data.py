# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import motion_collate
from typing import Tuple
from dataclasses import dataclass


def get_dataset_class(name):
    from data_loaders.humanml.data.dataset import GRAB, HOT3D, EGODEX
    if name == 'hot3d':
        return HOT3D
    if name == 'egodex':
        return EGODEX
    return GRAB


def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as eval_collate
        return eval_collate
    if name in ["grab", "hot3d", "egodex"]:
        return motion_collate
    else:
        return all_collate


@dataclass
class DatasetConfig:
    """Options for constructing hand-object interaction datasets and loaders."""

    name: str
    batch_size: int
    num_frames: int
    split: str = 'train'
    split_set: str = 'objects_unseen'
    hml_mode: str = 'train'
    use_abs3d: bool = False
    traject_only: bool = False
    use_random_projection: bool = False
    random_projection_scale: float = None
    augment_type: str = 'none'
    std_scale_shift: Tuple[float] = (1.0, 0.0)
    drop_redundant: bool = False
    use_pca: bool = False
    pre_grasp: bool = False
    hands_only: bool = False
    obj_only: bool = False
    use_contacts: bool = False
    obj_enc: bool = False
    motion_enc: bool = False
    motion_enc_frames: int = 0
    text_detailed: bool = False
    data_repr: str = ''
    mean_name: str = ''
    std_name: str = ''
    proj_matrix_name: str = ''
    wrist_only: bool = False
    use_scene_local: bool = False
    scene_data_root: str = './dataset/HOT3D_HANDS/scene_data'
    max_scene_points: int = 10000
    scene_feature_dim: int = 771
    scene_downsample: int = 10000
    use_precomputed_local_scenes: bool = False
    local_scenes_path: str = None
    use_scene_global: bool = False
    occ_maps_path: str = None
    use_concerto_grid: bool = False
    concerto_feature_dim: int = 1536
    max_concerto_points: int = 25000
    concerto_fps_points: int = 2000
    concerto_use_fps: bool = False
    use_semantic_features: bool = False
    semantic_feature_dim: int = 768
    semantic_features_path: str = './dataset/HOT3D_HANDS/semantic_features'
    scene_fps_path: str = None
    use_scene_bank: bool = False
    num_workers: int = 4


def get_dataset(conf: DatasetConfig):
    DATA = get_dataset_class(conf.name)

    if conf.name in ["grab", "hot3d", "egodex"]:
        dataset = DATA(split=conf.split,
                       split_set=conf.split_set,
                       num_frames=conf.num_frames,
                       mode=conf.hml_mode,
                       use_abs3d=conf.use_abs3d,
                       traject_only=conf.traject_only,
                       use_random_projection=conf.use_random_projection,
                       random_projection_scale=conf.random_projection_scale,
                       augment_type=conf.augment_type,
                       std_scale_shift=conf.std_scale_shift,
                       drop_redundant=conf.drop_redundant,
                       use_pca=conf.use_pca,
                       pre_grasp=conf.pre_grasp,
                       hands_only=conf.hands_only,
                       obj_only=conf.obj_only,
                       use_contacts=conf.use_contacts,
                       data_repr=conf.data_repr,
                       mean_name=conf.mean_name,
                       std_name=conf.std_name,
                       proj_matrix_name=conf.proj_matrix_name,
                       motion_enc_frames=conf.motion_enc_frames,
                       text_detailed=conf.text_detailed,
                       wrist_only=conf.wrist_only,
                       use_scene_points=conf.use_scene_local,
                       scene_data_root=conf.scene_data_root,
                       max_scene_points=conf.max_scene_points,
                       scene_feature_dim=conf.scene_feature_dim,
                       scene_downsample=conf.scene_downsample,
                       use_precomputed_local_scenes=getattr(conf, 'use_precomputed_local_scenes', False),
                       local_scenes_path=getattr(conf, 'local_scenes_path', None),
                       use_occ_maps=getattr(conf, 'use_scene_global', False),
                       occ_maps_path=getattr(conf, 'occ_maps_path', None),
                       use_concerto_grid=getattr(conf, 'use_concerto_grid', False),
                       concerto_feature_dim=getattr(conf, 'concerto_feature_dim', 1536),
                       max_concerto_points=getattr(conf, 'max_concerto_points', 25000),
                       concerto_fps_points=getattr(conf, 'concerto_fps_points', 2000),
                       concerto_use_fps=getattr(conf, 'concerto_use_fps', False),
                       use_semantic_features=getattr(conf, 'use_semantic_features', False),
                       semantic_feature_dim=getattr(conf, 'semantic_feature_dim', 768),
                       semantic_features_path=getattr(conf, 'semantic_features_path', './dataset/HOT3D_HANDS/semantic_features'),
                       scene_fps_path=getattr(conf, 'scene_fps_path', None),
                       use_scene_bank=getattr(conf, 'use_scene_bank', False),
                       )
    else:
        raise NotImplementedError()
    return dataset


def get_dataset_loader(conf: DatasetConfig, shuffle=True):
    """Build a loader with workers suited to preloaded scene and occupancy data."""
    dataset = get_dataset(conf)
    collate = get_collate_fn(conf.name, conf.hml_mode)

    num_workers = getattr(conf, 'num_workers', 4)
    loader = DataLoader(dataset,
                        batch_size=conf.batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        drop_last=True,
                        collate_fn=collate,
                        prefetch_factor=2 if num_workers > 0 else None,
                        persistent_workers=True if num_workers > 0 else False,
                        pin_memory=True)

    return loader
