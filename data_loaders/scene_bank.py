"""GPU-resident bank of per-scene point features."""

from pathlib import Path

import numpy as np
import torch


class SceneBank:
    """Fixed-FPS scene tensors with an all-valid point mask."""

    def __init__(self, bundle_path, device, dtype=torch.float16):
        self.bundle_path = Path(bundle_path)
        self.device = device
        self.dtype = dtype

        seqs = sorted(p.name for p in self.bundle_path.iterdir() if p.is_dir())
        if not seqs:
            raise FileNotFoundError(f"No scene bundles under {self.bundle_path}")

        feats, sems, coords = [], [], []
        for seq in seqs:
            feats.append(np.load(self.bundle_path / seq / 'features.npy'))
            sems.append(np.load(self.bundle_path / seq / 'semantic.npy'))
            coords.append(np.load(self.bundle_path / seq / 'coords.npy'))

        n_points = {f.shape[0] for f in feats}
        if len(n_points) != 1:
            raise ValueError(f"Bundles disagree on point count: {sorted(n_points)}")

        self.seq_names = seqs
        self.name_to_idx = {name: i for i, name in enumerate(seqs)}
        self.num_points = feats[0].shape[0]

        self.features = torch.from_numpy(np.stack(feats)).to(device=device, dtype=dtype)
        self.semantic = torch.from_numpy(np.stack(sems)).to(device=device, dtype=dtype)
        self.coords = torch.from_numpy(np.stack(coords)).to(device=device, dtype=torch.float32)
        self.mask = torch.ones(len(seqs), self.num_points, dtype=torch.bool, device=device)

        mb = (self.features.numel() + self.semantic.numel()) * self.features.element_size() / 2 ** 20
        print(f"[SceneBank] {len(seqs)} scenes x {self.num_points} points from {self.bundle_path} "
              f"({mb:.0f} MiB on {device})")

    def fill(self, y):
        """Gather scenes into y in place, zeroing entries whose scene index is -1."""
        idx = y.get('scene_idx')
        if idx is None:
            return y

        idx = idx.to(self.device).long()
        valid = idx >= 0
        safe = idx.clamp(min=0)

        features = self.features[safe]
        semantic = self.semantic[safe]
        coords = self.coords[safe]
        mask = self.mask[safe]

        anchor = y.get('scene_anchor')
        if anchor is not None:
            coords = coords - anchor.to(self.device).float().unsqueeze(1)

        if not bool(valid.all()):
            keep = valid.view(-1, 1, 1)
            features = features * keep
            semantic = semantic * keep
            coords = coords * keep
            mask = mask & valid.view(-1, 1)

        y['concerto_features'] = features
        y['concerto_coords'] = coords
        y['concerto_mask'] = mask
        y['semantic_features'] = semantic
        return y


def maybe_build_scene_bank(args, device):
    """Build a SceneBank when the config asks for one, otherwise return None."""
    if not getattr(args, 'use_scene_bank', False):
        return None
    bundle = getattr(args, 'scene_fps_path', None)
    if not bundle:
        raise ValueError("use_scene_bank requires scene_fps_path")
    return SceneBank(bundle, device)
