"""Scene point cloud loading and PCA coloring utilities for HOT3D visualization."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

SCENE_DATA_ROOT = './dataset/HOT3D_HANDS/scene_data'
ANCHORS_ROOT = 'dataset/HOT3D_HANDS/anchors'


def load_anchor(seq_name: str, anchors_root: str = ANCHORS_ROOT) -> Optional[np.ndarray]:
    """Load the transition-point anchor used to center a HOT3D sequence."""
    anchor_path = Path(anchors_root) / f'{seq_name}.npy'
    if anchor_path.exists():
        return np.load(str(anchor_path)).astype(np.float32)
    return None


def compute_pca_colors(
    features: np.ndarray,
    n_components: int = 3,
    pca_params: Optional[dict] = None,
) -> np.ndarray:
    """Map point features to RGB values in [0, 1] with PCA."""
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for PCA. Install with: pip install scikit-learn")

    if pca_params is not None:
        features_centered = features - pca_params['mean']
        pca_features = features_centered @ pca_params['components'].T
    else:
        features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(features_normalized)

    pca_min = pca_features.min(axis=0, keepdims=True)
    pca_max = pca_features.max(axis=0, keepdims=True)
    colors = (pca_features - pca_min) / (pca_max - pca_min + 1e-8)

    return colors.astype(np.float32)


def load_concerto_scene_points(
    recording_id: str,
    scene_data_root: str = SCENE_DATA_ROOT,
    max_points: int = 50000,
    use_grid: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load Concerto scene points in the original SLAM frame with PCA or gray colors."""
    concerto_path = Path(scene_data_root) / recording_id / 'concerto_features' / 'point_cloud'

    if not concerto_path.exists():
        raise FileNotFoundError(f"Concerto features not found: {concerto_path}")

    if use_grid:
        coords_grid_path = concerto_path / 'coords_grid.npy'
        coords_filtered_path = concerto_path / 'coords_filtered.npy'
        features_path = concerto_path / 'features_grid.npy'

        if not coords_grid_path.exists():
            raise FileNotFoundError(f"Grid coords not found: {coords_grid_path}")
        if not coords_filtered_path.exists():
            raise FileNotFoundError(f"Filtered coords not found: {coords_filtered_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"Grid features not found: {features_path}")

        coords_grid = np.load(str(coords_grid_path)).astype(np.float32)
        coords_filtered = np.load(str(coords_filtered_path)).astype(np.float32)
        features = np.load(str(features_path)).astype(np.float32)

        x_min, y_min, z_min = coords_filtered.min(axis=0)
        x_max, y_max, _ = coords_filtered.max(axis=0)
        shift = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min], dtype=np.float32)

        coords = coords_grid + shift

        if len(coords) > max_points:
            indices = np.random.choice(len(coords), max_points, replace=False)
            coords = coords[indices]
            features = features[indices]

        colors = compute_pca_colors(features)
    else:
        coords_path = concerto_path / 'coords_filtered.npy'

        if not coords_path.exists():
            raise FileNotFoundError(f"Filtered coords not found: {coords_path}")

        coords = np.load(str(coords_path)).astype(np.float32)

        if len(coords) > max_points:
            indices = np.random.choice(len(coords), max_points, replace=False)
            coords = coords[indices]

        colors = np.full((len(coords), 3), 0.6, dtype=np.float32)

    return coords, colors


def get_recording_id_from_seq_name(seq_name: str) -> str:
    parts = seq_name.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return seq_name


def get_hot3d_to_grab_transform() -> np.ndarray:
    """Return the preprocessing 90-degree CCW Z transform from HOT3D to GRAB."""
    R_z_ccw_90 = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1],
    ], dtype=np.float32)
    return R_z_ccw_90

