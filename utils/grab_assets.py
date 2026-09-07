"""Access separately distributed GRAB meshes and hand templates."""
import os

import trimesh


def load_grab_mesh(path, process=True):
    """Load a GRAB mesh, preserving vertex order when processing is disabled."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing. The GRAB object meshes and subject hand templates "
            f"are not part of the data package: download them from "
            f"https://grab.is.tue.mpg.de/ into assets/, as described in the README.")
    return trimesh.load(path, process=process)
