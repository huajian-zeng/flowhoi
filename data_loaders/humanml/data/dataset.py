# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from pathlib import Path

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.utils.data_utils import (
    OBJECT_LIST_NEW, OBJECT_NEW2ORIGINAL_DICT
)

try:
    from plyfile import PlyData
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

COORD_TRANSFORM = np.array([
    [ 0,  -1,  0],
    [1,  0,  0],
    [ 0,  0,  1]
], dtype=np.float32)

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)

    return default_collate(batch)


class MotionDataset(data.Dataset):
    """Motion-text dataset with optional object and scene conditioning."""

    def __init__(self,
                 opt,
                 mean,
                 std,
                 split_file,
                 use_rand_proj=False,
                 proj_matrix_dir=None,
                 traject_only=False,
                 mode='train',
                 random_proj_scale=10.0,
                 augment_type='none',
                 std_scale_shift=(1., 0.),
                 drop_redundant=False,
                 proj_matrix_name='',
                 hands_only=False,
                 motion_enc_frames=0,
                 wrist_only=False,
                 file_names_path='dataset/file_names.txt',
                 object_list_new=None,
                 object_new2original_dict=None,
                 dataset_name='grab',
                 t5_features_path=None,
                 use_scene_points=False,
                 scene_data_root='./dataset/HOT3D_HANDS/scene_data',
                 max_scene_points=10000,
                 scene_feature_dim=771,
                 scene_downsample=10000,
                 use_precomputed_local_scenes=False,
                 local_scenes_path=None,
                 use_occ_maps=False,
                 occ_maps_path=None,
                 use_concerto_grid=False,
                 concerto_feature_dim=1536,
                 max_concerto_points=25000,
                 concerto_fps_points=2000,
                 concerto_use_fps=False,
                 use_semantic_features=False,
                 semantic_feature_dim=768,
                 semantic_features_path=None,
                 scene_fps_path=None,
                 use_scene_bank=False):
        self.opt = opt
        self.dataset_name = dataset_name
        self.t5_features_path = t5_features_path

        self.use_scene_points = use_scene_points and dataset_name == 'hot3d'
        self.scene_data_root = scene_data_root
        self.max_scene_points = max_scene_points
        self.scene_feature_dim = scene_feature_dim
        self.scene_downsample = scene_downsample
        self._scene_cache = {}
        self._anchor_cache = {}

        self.use_precomputed_local_scenes = use_precomputed_local_scenes and use_scene_points
        if local_scenes_path is None:
            local_scenes_path = os.path.join(opt.data_root, f'local_scenes_{scene_downsample}')
        self.local_scenes_path = Path(local_scenes_path) if local_scenes_path else None
        self._preloaded_local_scenes = {}

        if self.use_scene_points:
            if self.use_precomputed_local_scenes and self.local_scenes_path and self.local_scenes_path.exists():
                print(f"Using pre-computed local scenes from: {self.local_scenes_path}")
                self._preload_local_scenes()
            elif not HAS_PLYFILE:
                print("Warning: plyfile not installed. Install with 'pip install plyfile' to enable scene point loading.")
                self.use_scene_points = False
            else:
                downsample_info = f", downsample={scene_downsample}" if scene_downsample > 0 else ""
                print(f"Scene points enabled: root={scene_data_root}, max_points={max_scene_points}{downsample_info}")

        self.use_occ_maps = use_occ_maps
        self.occ_maps_path = Path(occ_maps_path) if occ_maps_path else None
        self._preloaded_occ_maps = {}
        if self.use_occ_maps and self.occ_maps_path and self.occ_maps_path.exists():
            print(f"Occupancy maps enabled: {self.occ_maps_path}")
            self._preload_occ_maps()
        elif self.use_occ_maps:
            print(f"Warning: Occupancy maps path not found: {occ_maps_path}")
            self.use_occ_maps = False

        self.use_concerto_grid = use_concerto_grid and dataset_name == 'hot3d'
        self.concerto_feature_dim = concerto_feature_dim
        self.max_concerto_points = max_concerto_points
        self.concerto_fps_points = concerto_fps_points
        self.concerto_use_fps = concerto_use_fps
        self._concerto_cache = {}
        if self.use_concerto_grid:
            mode_str = "FPS sampling" if self.concerto_use_fps else f"padding to {self.max_concerto_points}"
            print(f"Concerto grid features enabled: {mode_str}")

        self.use_semantic_features = use_semantic_features and dataset_name == 'hot3d'
        self.semantic_feature_dim = semantic_feature_dim
        if semantic_features_path is None:
            semantic_features_path = os.path.join(opt.data_root, 'semantic_features')
        self.semantic_features_path = Path(semantic_features_path) if semantic_features_path else None
        self._semantic_cache = {}
        if self.use_semantic_features:
            print(f"Semantic features enabled: path={self.semantic_features_path}, dim={semantic_feature_dim}")

        self.scene_fps_path = Path(scene_fps_path) if scene_fps_path else None
        self._scene_fps_cache = {}
        if self.scene_fps_path is not None and not self.scene_fps_path.exists():
            print(f"Warning: scene_fps_path not found: {self.scene_fps_path}")
            self.scene_fps_path = None
        if self.scene_fps_path is not None:
            print(f"Precomputed FPS scene features enabled: {self.scene_fps_path}")

        self.use_scene_bank = bool(use_scene_bank) and self.scene_fps_path is not None
        self._scene_name_to_idx = {}
        if self.use_scene_bank:
            names = sorted(p.name for p in self.scene_fps_path.iterdir() if p.is_dir())
            self._scene_name_to_idx = {name: i for i, name in enumerate(names)}
            print(f"Scene bank mode: {len(names)} scenes indexed from {self.scene_fps_path}")

        self.t5_features = None
        self.t5_masks = None
        self.use_precomputed_t5 = False
        if t5_features_path is None:
            text_dir_name = os.path.basename(opt.text_dir)
            if 'grasp' in text_dir_name:
                t5_suffix = 'grasp'
            elif 'detailed' in text_dir_name:
                t5_suffix = 'detailed'
            else:
                t5_suffix = 'simple'
            t5_features_path = os.path.join(opt.data_root, f't5_features_{t5_suffix}.pt')
        if t5_features_path is not None and os.path.exists(t5_features_path):
            print(f'Loading precomputed T5 features from {t5_features_path}')
            t5_data = torch.load(t5_features_path)
            self.t5_features = t5_data['embeddings']
            self.t5_masks = t5_data.get('masks', None)
            print(f'Loaded {len(self.t5_features)} T5 embeddings with masks')
            self.use_precomputed_t5 = True

        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 20 if not hands_only else 8

        self.use_rand_proj = use_rand_proj
        self.traject_only = traject_only
        self.hands_only = hands_only
        self.wrist_only = wrist_only

        self.object_list_new = object_list_new if object_list_new is not None else OBJECT_LIST_NEW
        self.object_new2original_dict = object_new2original_dict if object_new2original_dict is not None else OBJECT_NEW2ORIGINAL_DICT
        self.file_names_path = file_names_path

        if wrist_only:
            self.wrist_only_idcs = np.concatenate([
                np.arange(0, 6),
                np.arange(30, 36),
                np.arange(60, 66),
                np.arange(108, 117),
            ])

        self.mode = mode
        self.bps = np.load(pjoin(opt.data_root, 'bps_enc.npy'),allow_pickle=True)
        self.bps_mirrored = np.load(pjoin(opt.data_root, 'bps_enc_mirrored.npy'),allow_pickle=True)

        self.augment_type = augment_type
        self.motion_enc_frames = motion_enc_frames

        self.std_scale_shift = std_scale_shift
        self.drop_redundant = drop_redundant

        data_dict = {}
        id_list = []

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        id_dict = {}
        with open(self.file_names_path, 'r') as file:
                for line in file:
                    line = line.strip()

                    key, value = line.split(',', 1)

                    id_dict[key] = value


        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len:
                   continue
                if (len(motion) >= 200):
                    motion = motion[:200]
                text_data = []
                flag = False
                mirrored = 'M' in name
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len:
                                    continue
                                new_name = random.choice(
                                    'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice(
                                        'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {
                                    'motion': n_motion,
                                    'length': len(n_motion),
                                    'text': [text_dict]
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except Exception as err:
                                print(f'Warning: skipping segment of {name}: {err}')

                if flag:
                    data_dict[name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data,
                        'mirrored': mirrored,
                        'id': id_dict[name],
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as err:
                print(f'Warning: skipping sequence {name}: {err}')


        name_list = new_name_list
        self.max_motion_length = max(length_list) 
        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

        self.max_length = 20 if not hands_only else min(length_list)
        self.reset_max_len(self.max_length)


        if self.traject_only:
            if self.hands_only:
                self.traj_only_idcs = np.concatenate([np.arange(0,108)])
            else:
                traj_dim = motion.shape[1] if motion.shape[1] <= 118 else 117
                self.traj_only_idcs = np.concatenate([np.arange(0, traj_dim)])

        self.non_redundant_idcs = np.concatenate([np.arange(0, 3), np.arange(63, 66), np.arange(126, 186), np.arange(motion.shape[1]-15, motion.shape[1])])

        if use_rand_proj:
            self.init_random_projection(proj_matrix_dir,
                                        scale=random_proj_scale,
                                        proj_matrix_name=proj_matrix_name)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)

        self.max_length = length

    def get_std_mean(self, traject_only=None, drop_redundant=None, wrist_only=None):

        if traject_only is None:
            traject_only = self.traject_only
        if drop_redundant is None:
            drop_redundant = self.drop_redundant
        if wrist_only is None:
            wrist_only = self.wrist_only

        if wrist_only:
            std = self.std[self.wrist_only_idcs]
            mean = self.mean[self.wrist_only_idcs]
        elif traject_only:
            std = self.std[self.traj_only_idcs]
            mean = self.mean[self.traj_only_idcs]
        elif drop_redundant:
            std = self.std[self.non_redundant_idcs]
            mean = self.mean[self.non_redundant_idcs]
        else:
            std = self.std
            mean = self.mean

        std = std * self.std_scale_shift[0] + self.std_scale_shift[1]
        return std, mean

    def inv_transform(self, data, traject_only=None):
        if self.use_rand_proj:
            data = self.inv_random_projection(data)
        std, mean = self.get_std_mean(traject_only)
        return data * std + mean

    def _get_sequence_name_from_data_id(self, data_id):
        """Extract P{id}_{hash} from a HOT3D interaction identifier."""
        parts = data_id.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return None

    def _load_anchor(self, data_id):
        """Load a transformed 3-D interaction anchor, or return None."""
        anchor_path = Path(self.opt.data_root) / 'anchors' / f'{data_id}.npy'
        if anchor_path.exists():
            return np.load(str(anchor_path)).astype(np.float32)
        return None

    def _load_raw_scene(self, seq_name):
        """Load and cache transformed, uncentered scene coordinates and features."""
        if seq_name in self._scene_cache:
            return self._scene_cache[seq_name]

        seq_path = Path(self.scene_data_root) / seq_name

        if self.scene_downsample > 0:
            ply_path = seq_path / 'gaussian_generation' / 'point_cloud' / 'iteration_30000' / f'point_cloud_{self.scene_downsample}.ply'
            feat_path = seq_path / 'scene_features' / 'point_cloud' / f'pred_langfeat_{self.scene_downsample}.npy'
        else:
            ply_path = seq_path / 'gaussian_generation' / 'point_cloud' / 'iteration_30000' / 'point_cloud.ply'
            feat_path = seq_path / 'scene_features' / 'point_cloud' / 'pred_langfeat.npy'

        if not ply_path.exists() or not feat_path.exists():
            self._scene_cache[seq_name] = (None, None)
            return None, None

        try:
            plydata = PlyData.read(str(ply_path))
            xyz = np.stack([
                np.asarray(plydata['vertex']['x']),
                np.asarray(plydata['vertex']['y']),
                np.asarray(plydata['vertex']['z'])
            ], axis=-1).astype(np.float32)

            features = np.load(str(feat_path)).astype(np.float32)

            if xyz.shape[0] != features.shape[0]:
                print(f"Warning: Point cloud ({xyz.shape[0]}) and features ({features.shape[0]}) size mismatch for {seq_name}")
                min_n = min(xyz.shape[0], features.shape[0])
                xyz = xyz[:min_n]
                features = features[:min_n]

            n_points = xyz.shape[0]
            if self.scene_downsample == 0 and n_points > self.max_scene_points:
                indices = np.random.choice(n_points, self.max_scene_points, replace=False)
                xyz = xyz[indices]
                features = features[indices]

            xyz = xyz @ COORD_TRANSFORM.T

            if xyz.shape[0] < self.max_scene_points:
                pad_n = self.max_scene_points - xyz.shape[0]
                xyz = np.concatenate([xyz, np.zeros((pad_n, 3), dtype=np.float32)], axis=0)
                features = np.concatenate([features, np.zeros((pad_n, features.shape[1]), dtype=np.float32)], axis=0)

            self._scene_cache[seq_name] = (xyz, features)
            if len(self._scene_cache) <= 5:
                print(f"[Scene Cache] Loaded {seq_name}: {xyz.shape[0]} points, cache size: {len(self._scene_cache)}")
            return xyz, features

        except Exception as e:
            print(f"Error loading scene for {seq_name}: {e}")
            self._scene_cache[seq_name] = (None, None)
            return None, None

    def _preload_local_scenes(self):
        """Preload local scenes for worker-shared copy-on-write access."""
        if self.local_scenes_path is None or not self.local_scenes_path.exists():
            return

        npz_files = list(self.local_scenes_path.glob('*.npz'))
        print(f"Pre-loading {len(npz_files)} local scenes into memory...")

        loaded = 0
        for npz_path in tqdm(npz_files, desc="Loading scenes"):
            data_id = npz_path.stem
            try:
                data = np.load(str(npz_path))
                xyz = data['xyz'].astype(np.float32)
                features = data['features'].astype(np.float32)
                self._preloaded_local_scenes[data_id] = np.concatenate([xyz, features], axis=-1)
                loaded += 1
            except Exception as e:
                print(f"Error loading {npz_path}: {e}")

        if self._preloaded_local_scenes:
            sample_size = next(iter(self._preloaded_local_scenes.values())).nbytes
            total_mb = (sample_size * len(self._preloaded_local_scenes)) / (1024 * 1024)
            print(f"Pre-loaded {loaded} scenes, ~{total_mb:.1f} MB in memory")

    def _load_precomputed_local_scene(self, data_id):
        """Load a precomputed local scene from memory or disk."""
        if data_id in self._preloaded_local_scenes:
            return torch.from_numpy(self._preloaded_local_scenes[data_id].copy())

        if self.local_scenes_path is None:
            return None
        npz_path = self.local_scenes_path / f'{data_id}.npz'
        if not npz_path.exists():
            return None
        try:
            data = np.load(str(npz_path))
            xyz = data['xyz'].astype(np.float32)
            features = data['features'].astype(np.float32)
            scene_points = np.concatenate([xyz, features], axis=-1)
            return torch.from_numpy(scene_points)
        except Exception as e:
            print(f"Error loading precomputed scene for {data_id}: {e}")
            return None

    def _load_scene_points(self, data_id):
        """Load anchor-centered scene points as [N, 3 + feature_dim] tensors."""
        if self.use_precomputed_local_scenes:
            scene_points = self._load_precomputed_local_scene(data_id)
            if scene_points is not None:
                return scene_points

        seq_name = self._get_sequence_name_from_data_id(data_id)
        if seq_name is None:
            return None

        xyz, features = self._load_raw_scene(seq_name)
        if xyz is None:
            return None

        if data_id not in self._anchor_cache:
            anchor = self._load_anchor(data_id)
            if anchor is None:
                print(f"Warning: Anchor not found for {data_id}, using zero")
                anchor = np.zeros(3, dtype=np.float32)
            self._anchor_cache[data_id] = anchor
        anchor = self._anchor_cache[data_id]

        xyz_centered = xyz.copy()
        xyz_centered[:, :3] = xyz_centered[:, :3] - anchor

        scene_points = np.concatenate([xyz_centered, features], axis=-1)
        return torch.from_numpy(scene_points)

    def _preload_occ_maps(self):
        """Pre-load all occupancy maps into memory at initialization."""
        if self.occ_maps_path is None or not self.occ_maps_path.exists():
            return

        npy_files = list(self.occ_maps_path.glob('*.npy'))
        print(f"Pre-loading {len(npy_files)} occupancy maps into memory...")

        loaded = 0
        for npy_path in tqdm(npy_files, desc="Loading occ maps"):
            data_id = npy_path.stem
            try:
                occ_map = np.load(str(npy_path)).astype(np.float32)
                self._preloaded_occ_maps[data_id] = occ_map
                loaded += 1
            except Exception as e:
                print(f"Error loading {npy_path}: {e}")

        if self._preloaded_occ_maps:
            sample_size = next(iter(self._preloaded_occ_maps.values())).nbytes
            total_mb = (sample_size * len(self._preloaded_occ_maps)) / (1024 * 1024)
            print(f"Pre-loaded {loaded} occ maps, ~{total_mb:.1f} MB in memory")

    def _load_occ_map(self, data_id):
        """Load an occupancy map shaped [Z, Y, X] from memory or disk."""
        if not self.use_occ_maps:
            return None

        if data_id in self._preloaded_occ_maps:
            return torch.from_numpy(self._preloaded_occ_maps[data_id].copy())

        if self.occ_maps_path is None:
            return None

        occ_path = self.occ_maps_path / f'{data_id}.npy'
        if not occ_path.exists():
            return None

        try:
            occ_map = np.load(str(occ_path)).astype(np.float32)
            return torch.from_numpy(occ_map)
        except Exception as e:
            print(f"Error loading occ_map for {data_id}: {e}")
            return None

    def _load_concerto_grid(self, data_id):
        """Load anchor-centered Concerto data after undoing CenterShift, padded for Perceiver or raw for FPS."""
        if not self.use_concerto_grid:
            return None, None, None

        seq_name = self._get_sequence_name_from_data_id(data_id)
        if seq_name is None:
            return self._get_empty_concerto()

        if self.scene_fps_path is not None:
            bundle = self._load_scene_fps(seq_name)
            if bundle is None:
                return self._get_empty_concerto()
            anchor = self._get_anchor(data_id)
            return (bundle['features'].copy(),
                    bundle['coords'] - anchor,
                    bundle['mask'].copy())

        cache_key = f"{seq_name}_fps" if self.concerto_use_fps else seq_name
        cached = self._concerto_cache.get(cache_key)

        if cached is None:
            local_base_path = Path(self.opt.data_root) / 'concerto_features' / seq_name
            ssd_base_path = Path(self.scene_data_root) / seq_name / 'concerto_features' / 'point_cloud'

            if (local_base_path / 'features_grid.npy').exists():
                base_path = local_base_path
            else:
                base_path = ssd_base_path

            feat_path = base_path / 'features_grid.npy'
            coord_path = base_path / 'coords_grid.npy'
            filtered_path = base_path / 'coords_filtered.npy'

            if not feat_path.exists() or not coord_path.exists():
                if len(self._concerto_cache) < 5:
                    print(f"Warning: Concerto features not found for {seq_name}")
                return self._get_empty_concerto()

            try:
                raw_features = np.load(str(feat_path)).astype(np.float32)
                raw_coords = np.load(str(coord_path)).astype(np.float32)
                n_points = raw_features.shape[0]

                center_shift = np.zeros(3, dtype=np.float32)
                local_filtered_path = Path(self.opt.data_root) / 'concerto_center_shifts' / f'{seq_name}_coords_filtered.npy'
                if local_filtered_path.exists():
                    coords_filtered = np.load(str(local_filtered_path)).astype(np.float32)
                    x_min, y_min, z_min = coords_filtered.min(axis=0)
                    x_max, y_max, _ = coords_filtered.max(axis=0)
                    center_shift = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min], dtype=np.float32)
                elif filtered_path.exists():
                    coords_filtered = np.load(str(filtered_path)).astype(np.float32)
                    x_min, y_min, z_min = coords_filtered.min(axis=0)
                    x_max, y_max, _ = coords_filtered.max(axis=0)
                    center_shift = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min], dtype=np.float32)

                raw_coords = raw_coords @ COORD_TRANSFORM.T
                center_shift_transformed = center_shift @ COORD_TRANSFORM.T
                raw_coords = raw_coords + center_shift_transformed

                if self.concerto_use_fps:
                    cached = {
                        'features': raw_features,
                        'coords': raw_coords,
                    }
                else:
                    max_pts = self.max_concerto_points
                    features = np.zeros((max_pts, self.concerto_feature_dim), dtype=np.float32)
                    coords = np.zeros((max_pts, 3), dtype=np.float32)
                    mask = np.zeros(max_pts, dtype=bool)

                    if n_points <= max_pts:
                        features[:n_points] = raw_features
                        coords[:n_points] = raw_coords
                        mask[:n_points] = True
                    else:
                        features = raw_features[:max_pts]
                        coords = raw_coords[:max_pts]
                        mask[:] = True

                    cached = {
                        'features': features,
                        'coords': coords,
                        'mask': mask,
                    }

                    if len(self._concerto_cache) <= 5:
                        print(f"[Concerto Cache] Loaded {seq_name}: {n_points} points, cache size: {len(self._concerto_cache)}")

                self._concerto_cache[cache_key] = cached

            except Exception as e:
                print(f"Error loading Concerto features for {seq_name}: {e}")
                return self._get_empty_concerto()

        if data_id not in self._anchor_cache:
            anchor = self._load_anchor(data_id)
            if anchor is None:
                anchor = np.zeros(3, dtype=np.float32)
            self._anchor_cache[data_id] = anchor
        anchor = self._anchor_cache[data_id]

        features = cached['features'].copy()
        coords = cached['coords'].copy()
        mask = cached.get('mask')

        if mask is not None:
            coords[mask] = coords[mask] - anchor
        else:
            coords = coords - anchor

        if mask is not None:
            return features, coords, mask.copy()
        else:
            return features, coords, None

    def _get_anchor(self, data_id):
        """Per-interaction anchor used to centre scene coordinates."""
        if data_id not in self._anchor_cache:
            anchor = self._load_anchor(data_id)
            if anchor is None:
                anchor = np.zeros(3, dtype=np.float32)
            self._anchor_cache[data_id] = anchor
        return self._anchor_cache[data_id]

    def _load_scene_fps(self, seq_name):
        """Load and cache a float16 precomputed FPS scene bundle."""
        cached = self._scene_fps_cache.get(seq_name)
        if cached is not None:
            return cached

        base = self.scene_fps_path / seq_name
        try:
            features = np.load(base / 'features.npy')
            semantic = np.load(base / 'semantic.npy')
            coords = np.load(base / 'coords.npy').astype(np.float32)
        except OSError:
            if len(self._scene_fps_cache) < 5:
                print(f"Warning: FPS scene bundle missing for {seq_name} under {base}")
            return None

        cached = {
            'features': features,
            'semantic': semantic,
            'coords': coords,
            'mask': np.ones(coords.shape[0], dtype=bool),
        }
        self._scene_fps_cache[seq_name] = cached
        return cached

    def _get_empty_concerto(self):
        """Return empty Concerto features for the configured mode."""
        if self.concerto_use_fps:
            return (np.zeros((self.concerto_fps_points, self.concerto_feature_dim), dtype=np.float32),
                    np.zeros((self.concerto_fps_points, 3), dtype=np.float32),
                    None)
        else:
            return (np.zeros((self.max_concerto_points, self.concerto_feature_dim), dtype=np.float32),
                    np.zeros((self.max_concerto_points, 3), dtype=np.float32),
                    np.zeros(self.max_concerto_points, dtype=bool))

    def _load_semantic_features(self, data_id):
        """Load SceneSplat features aligned pointwise with Concerto features."""
        if not self.use_semantic_features:
            return None

        seq_name = self._get_sequence_name_from_data_id(data_id)
        if seq_name is None:
            return self._get_empty_semantic()

        if self.scene_fps_path is not None:
            bundle = self._load_scene_fps(seq_name)
            if bundle is None:
                return self._get_empty_semantic()
            return bundle['semantic'].copy()

        cached = self._semantic_cache.get(seq_name)

        if cached is None:
            local_path = self.semantic_features_path / seq_name / 'aligned_langfeat.npy'

            ssd_path = Path(self.scene_data_root) / seq_name / 'scene_features' / 'point_cloud' / 'aligned_langfeat.npy'

            if local_path.exists():
                feat_path = local_path
            elif ssd_path.exists():
                feat_path = ssd_path
            else:
                if len(self._semantic_cache) < 5:
                    print(f"Warning: Semantic features not found for {seq_name}")
                    print(f"  Tried: {local_path}")
                    print(f"  Tried: {ssd_path}")
                return self._get_empty_semantic()

            try:
                raw_features = np.load(str(feat_path)).astype(np.float32)
                n_points = raw_features.shape[0]

                max_pts = self.max_concerto_points
                features = np.zeros((max_pts, self.semantic_feature_dim), dtype=np.float32)

                if n_points <= max_pts:
                    features[:n_points] = raw_features
                else:
                    features = raw_features[:max_pts]

                cached = features

                if len(self._semantic_cache) <= 5:
                    print(f"[Semantic Cache] Loaded {seq_name}: {n_points} points")

                self._semantic_cache[seq_name] = cached

            except Exception as e:
                print(f"Error loading semantic features for {seq_name}: {e}")
                return self._get_empty_semantic()

        return cached.copy()

    def _get_empty_semantic(self):
        """Return empty semantic features matching Concerto dimensions."""
        return np.zeros((self.max_concerto_points, self.semantic_feature_dim), dtype=np.float32)

    def inv_transform_th(self, data, traject_only=None, use_rand_proj=None):
        use_rand_proj = self.use_rand_proj if use_rand_proj is None else use_rand_proj
        if use_rand_proj:
            data = self.inv_random_projection(data, mode="th")
        std, mean = self.get_std_mean(traject_only)
        return data * torch.from_numpy(std).to(
            data.device).to(data.dtype) + torch.from_numpy(mean).to(data.device).to(data.dtype)

    def transform_th(self, data, traject_only=None, use_rand_proj=None):
        std, mean = self.get_std_mean(traject_only)

        data = (data - torch.from_numpy(mean).to(
            data.device).to(data.dtype)) / torch.from_numpy(std).to(data.device).to(data.dtype)
        use_rand_proj = self.use_rand_proj if use_rand_proj is None else use_rand_proj
        if use_rand_proj:
            data = self.random_projection(data, mode="th")
        return data

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        """Return a 19-field motion sample whose auxiliary tensors occupy fields 9-18."""
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, mirrored, data_id = data['motion'], data['length'], data[
            'text'], data['mirrored'], data['id']

        if len(text_list) > 1:
            text_data = text_list[1]
        else:
            text_data = text_list[0]
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) == 1:
            sent_len = len(tokens)
            pass
        elif len(tokens) < self.opt.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        motion_full = motion.copy()
        if self.wrist_only:
            motion = motion[:, self.wrist_only_idcs]
        elif self.traject_only:
            motion = motion[:, self.traj_only_idcs]

        if self.drop_redundant:
            motion = motion[:, self.non_redundant_idcs]

        "Z Normalization"
        std, mean = self.get_std_mean()
        motion = (motion - mean) / std

        if (not self.mode in ["eval", "gt"]) and self.use_rand_proj:
            motion = self.random_projection(motion)
        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.tile(motion[m_length-1],(self.max_motion_length - m_length,1))
            ],axis=0)

        if self.dataset_name == 'hot3d':
            parts = data_id.split('_')
            obj_name_from_id = '_'.join(parts[2:-1])
            obj_name = [obj_name_from_id]
        elif self.dataset_name == 'egodex':
            import re
            caption_lower = caption.lower()
            match = re.search(r'pick up (?:the |a |an )?(.+?)(?:\s+from|\s+with|\s*\.)', caption_lower)
            if match:
                obj_name = [match.group(1).strip()]
            else:
                match = re.search(r'(?:grasps|grabs|takes|grips|picks up) the (.+?)(?:\s*\.)', caption_lower)
                if match:
                    obj_name = [match.group(1).strip()]
                else:
                    obj_name = ['egodex_placeholder']
        else:
            obj_name = [s for s in self.object_list_new if s in caption]

        bps_dict = self.bps_mirrored.item() if mirrored else self.bps.item()
        bps_key = self.object_new2original_dict.get(obj_name[0], obj_name[0]) if self.object_new2original_dict else obj_name[0]
        if bps_key not in bps_dict:
            if 'egodex_placeholder' in bps_dict:
                bps_key = 'egodex_placeholder'
            else:
                bps_key = list(bps_dict.keys())[0]
        obj_bps = bps_dict[bps_key]

        t5_emb = None
        t5_mask = None
        if self.use_precomputed_t5 and self.t5_features is not None:
            if caption in self.t5_features:
                t5_emb = self.t5_features[caption]
                if self.t5_masks is not None and caption in self.t5_masks:
                    t5_mask = self.t5_masks[caption]

        scene_points = None
        if self.use_scene_points:
            scene_points = self._load_scene_points(data_id)

        occ_map = None
        if self.use_occ_maps:
            occ_map = self._load_occ_map(data_id)

        concerto_features = None
        concerto_coords = None
        concerto_mask = None
        scene_idx = -1
        scene_anchor = np.zeros(3, dtype=np.float32)
        if self.use_scene_bank:
            seq_name = self._get_sequence_name_from_data_id(data_id)
            scene_idx = self._scene_name_to_idx.get(seq_name, -1)
            if scene_idx >= 0:
                scene_anchor = np.asarray(self._get_anchor(data_id), dtype=np.float32)
        elif self.use_concerto_grid:
            concerto_features, concerto_coords, concerto_mask = self._load_concerto_grid(data_id)

        semantic_features = None
        if self.use_semantic_features and not self.use_scene_bank:
            semantic_features = self._load_semantic_features(data_id)

        return caption, sent_len, motion, m_length, '_'.join(
            tokens), obj_bps, self.motion_enc_frames, motion_full, data_id, scene_points, occ_map, t5_emb, t5_mask, concerto_features, concerto_coords, concerto_mask, semantic_features, scene_idx, scene_anchor

    def init_random_projection(self, save_at, scale: float, proj_matrix_name=''):
        """Scale wrist and object-pose features at indices 0-5, 30-35, 60-65, and 108-116."""
        assert proj_matrix_name != '', "Please provide a name for the projection matrix"

        rand_proj_file = proj_matrix_name
        inv_rand_proj_file = "inv_" + proj_matrix_name

        if os.path.isfile(os.path.join(save_at, rand_proj_file)):
            print(f"Loading random projection matrix from {save_at}")
            self.proj_matrix = np.load(os.path.join(save_at, rand_proj_file))
            self.inv_proj_matrix = np.load(
                os.path.join(save_at, inv_rand_proj_file))

            self.proj_matrix_th = torch.from_numpy(self.proj_matrix)
            self.inv_proj_matrix_th = torch.from_numpy(self.inv_proj_matrix)

            if self.traject_only:
                self.proj_matrix = self.proj_matrix[self.traj_only_idcs][:,self.traj_only_idcs]
                self.inv_proj_matrix = self.inv_proj_matrix[self.traj_only_idcs][:,self.traj_only_idcs]
                self.proj_matrix_th = self.proj_matrix_th[self.traj_only_idcs][:,self.traj_only_idcs]
                self.inv_proj_matrix_th = self.inv_proj_matrix_th[self.traj_only_idcs][:,self.traj_only_idcs]
        else:
            print(f"Creating random projection matrix {scale}")

            self.proj_matrix = torch.normal(
            mean=0, std=1.0, size=(117, 117),
            dtype=torch.float)
            self.proj_matrix[[0, 1, 2, 3, 4, 5, 30, 31, 32, 33, 34, 35, 60, 61, 62, 63, 64, 65, 108, 109, 110, 111, 112, 113, 114, 115, 116], :] *= scale
            self.proj_matrix = self.proj_matrix / np.sqrt(117 - 27 + 27 * scale**2)

            self.inv_proj_matrix = torch.inverse(self.proj_matrix)

            self.proj_matrix = self.proj_matrix.detach().cpu().numpy()
            self.inv_proj_matrix = self.inv_proj_matrix.detach().cpu().numpy()

            self.proj_matrix_th = torch.from_numpy(self.proj_matrix)
            self.inv_proj_matrix_th = torch.from_numpy(self.inv_proj_matrix)

            np.save(os.path.join(save_at, rand_proj_file), self.proj_matrix)
            np.save(os.path.join(save_at, inv_rand_proj_file),self.inv_proj_matrix)

            if self.traject_only:
                self.proj_matrix = self.proj_matrix[self.traj_only_idcs][:,self.traj_only_idcs]
                self.inv_proj_matrix = self.inv_proj_matrix[self.traj_only_idcs][:,self.traj_only_idcs]
                self.proj_matrix_th = self.proj_matrix_th[self.traj_only_idcs][:,self.traj_only_idcs]
                self.inv_proj_matrix_th = self.inv_proj_matrix_th[self.traj_only_idcs][:,self.traj_only_idcs]

    def random_projection(self, motion, mode="np"):

        if mode == "th":
            proj_matrix = self.proj_matrix_th.to(device=motion.device, dtype=motion.dtype)
            return torch.matmul(motion, proj_matrix)

        return np.matmul(motion, self.proj_matrix)

    def inv_random_projection(self, data, mode="np"):
        if mode == "th":
            inv_proj_matrix = self.inv_proj_matrix_th.to(device=data.device, dtype=data.dtype)
            return torch.matmul(data, inv_proj_matrix)
        return np.matmul(data, self.inv_proj_matrix)


class HOIDataset(data.Dataset):
    """Configured wrapper around a MotionDataset instance."""

    def __init__(self,
                 mode,
                 datapath='./dataset/humanml_opt.txt',
                 split ="train",
                 split_set = "objects_unseen",
                 use_abs3d=False,
                 traject_only=False,
                 use_random_projection=False,
                 random_projection_scale=None,
                 augment_type='none',
                 std_scale_shift=(1., 0.),
                 drop_redundant=False,
                 num_frames=None,
                 data_repr='',
                 mean_name='',
                 std_name='',
                 proj_matrix_name='',
                 hands_only=False,
                 text_detailed=False,
                 motion_enc_frames=0,
                 wrist_only=False,
                 file_names_path='dataset/file_names.txt',
                 object_list_new=None,
                 object_new2original_dict=None,
                 use_scene_points=False,
                 scene_data_root='./dataset/HOT3D_HANDS/scene_data',
                 max_scene_points=10000,
                 scene_feature_dim=771,
                 scene_downsample=10000,
                 use_precomputed_local_scenes=False,
                 local_scenes_path=None,
                 use_occ_maps=False,
                 occ_maps_path=None,
                 use_concerto_grid=False,
                 concerto_feature_dim=1536,
                 max_concerto_points=25000,
                 concerto_fps_points=2000,
                 concerto_use_fps=False,
                 use_semantic_features=False,
                 semantic_feature_dim=768,
                 semantic_features_path=None,
                 scene_fps_path=None,
                 use_scene_bank=False,
                 **kwargs):
        self.mode = mode
        self.use_occ_maps = use_occ_maps
        self.occ_maps_path = occ_maps_path

        self.use_concerto_grid = use_concerto_grid
        self.concerto_feature_dim = concerto_feature_dim
        self.max_concerto_points = max_concerto_points
        self.concerto_fps_points = concerto_fps_points
        self.concerto_use_fps = concerto_use_fps

        self.use_semantic_features = use_semantic_features
        self.semantic_feature_dim = semantic_feature_dim
        self.semantic_features_path = semantic_features_path
        self.scene_fps_path = scene_fps_path
        self.use_scene_bank = use_scene_bank

        abs_base_path = '.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None

        opt = get_opt(dataset_opt_path, device, mode, data_repr, use_abs3d=use_abs3d, max_motion_length=num_frames,
                        hands_only=hands_only, text_detailed=text_detailed)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)

        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt
        self.dataset_name = opt.dataset_name
        print('Loading dataset %s ...' % opt.dataset_name)

        self.absolute_3d = use_abs3d
        self.traject_only = traject_only
        self.use_rand_proj = use_random_projection
        self.random_proj_scale = random_projection_scale
        self.augment_type = augment_type
        self.std_scale_shift = std_scale_shift
        self.drop_redundant = drop_redundant
        self.wrist_only = wrist_only

        if self.use_rand_proj:
            proj_matrix_dir = "./dataset"
            print(f'proj_matrix_dir = {proj_matrix_dir}')
        else:
            proj_matrix_dir = None


        if split_set not in ['random', 'objects_unseen']:
            mean_base, mean_ext = os.path.splitext(mean_name)
            std_base, std_ext = os.path.splitext(std_name)
            split_mean_name = f'{mean_base}_{split_set}{mean_ext}'
            split_std_name = f'{std_base}_{split_set}{std_ext}'

            split_mean_path = pjoin(opt.data_root, split_mean_name)
            split_std_path = pjoin(opt.data_root, split_std_name)

            if os.path.exists(split_mean_path) and os.path.exists(split_std_path):
                print(f'Using split-specific stats: {split_mean_name}, {split_std_name}')
                mean_name = split_mean_name
                std_name = split_std_name
            else:
                print(f'No split-specific stats for {split_set}; using {mean_name}, {std_name}')

        self.mean = np.load(pjoin(opt.data_root, mean_name))
        self.std = np.load(pjoin(opt.data_root, std_name))

        self.split_file = pjoin(opt.data_root, f'{split}_{split_set}.txt')

      
        print(
            f'dataset aug: {self.augment_type} std_scale_shift: {self.std_scale_shift}'
        )
        print(f'dataset drop redundant information: {self.drop_redundant}')
        self.motion_dataset = MotionDataset(
            self.opt,
            self.mean,
            self.std,
            self.split_file,
            use_rand_proj=self.use_rand_proj,
            proj_matrix_dir=proj_matrix_dir,
            traject_only=self.traject_only,
            mode=mode,
            random_proj_scale=self.random_proj_scale,
            augment_type=self.augment_type,
            std_scale_shift=self.std_scale_shift,
            drop_redundant=self.drop_redundant,
            proj_matrix_name=proj_matrix_name,
            hands_only=hands_only,
            motion_enc_frames=motion_enc_frames,
            wrist_only=self.wrist_only,
            file_names_path=file_names_path,
            object_list_new=object_list_new,
            object_new2original_dict=object_new2original_dict,
            dataset_name=self.dataset_name,
            use_scene_points=use_scene_points,
            scene_data_root=scene_data_root,
            max_scene_points=max_scene_points,
            scene_feature_dim=scene_feature_dim,
            scene_downsample=scene_downsample,
            use_precomputed_local_scenes=use_precomputed_local_scenes,
            local_scenes_path=local_scenes_path,
            use_occ_maps=self.use_occ_maps,
            occ_maps_path=self.occ_maps_path,
            use_concerto_grid=self.use_concerto_grid,
            concerto_feature_dim=self.concerto_feature_dim,
            max_concerto_points=self.max_concerto_points,
            concerto_fps_points=self.concerto_fps_points,
            concerto_use_fps=self.concerto_use_fps,
            use_semantic_features=self.use_semantic_features,
            semantic_feature_dim=self.semantic_feature_dim,
            semantic_features_path=self.semantic_features_path,
            scene_fps_path=self.scene_fps_path,
            use_scene_bank=self.use_scene_bank)

        assert len(self.motion_dataset) > 0, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'Only dataset text files found. Get the full preprocessed data as described ' \
                                          'in the README file.'


    def __getitem__(self, item):
        return self.motion_dataset.__getitem__(item)

    def __len__(self):
        return self.motion_dataset.__len__()

class GRAB(HOIDataset):
    """GRAB hand-object interaction dataset."""

    def __init__(self,
                mode,
                datapath='./dataset/grab_opt_objects.txt',
                split="train",
                **kwargs):
        super(GRAB, self).__init__(mode, datapath, split, **kwargs)


class HOT3D(HOIDataset):
    """HOT3D dataset deriving object names from interaction identifiers."""

    def __init__(self,
                mode,
                datapath='./dataset/hot3d_opt.txt',
                split="train",
                **kwargs):
        super(HOT3D, self).__init__(
            mode,
            datapath,
            split,
            file_names_path='dataset/file_names_hot3d.txt',
            object_list_new=None,
            object_new2original_dict=None,
            **kwargs)




class EGODEX(HOIDataset):
    """EgoDex dataset wrapper that forces the training split."""

    def __init__(self,
                mode,
                datapath='./dataset/egodex_opt.txt',
                split="train",
                **kwargs):
        if split != "train":
            print(f"Warning: EgoDex only has training data. Ignoring requested split '{split}', using 'train'.")
            split = "train"
        super(EGODEX, self).__init__(
            mode,
            datapath,
            split,
            file_names_path='dataset/EGODEX_HANDS/file_names.txt',
            object_list_new=None,
            object_new2original_dict=None,
            **kwargs)
