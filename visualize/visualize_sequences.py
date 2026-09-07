# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import trimesh
import trimesh.registration
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.viewer import Viewer
from pathlib import Path
from utils.grab_assets import load_grab_mesh
from utils.rotation_conversions import rotation_6d_to_matrix
from scipy.spatial.transform import Rotation as R
from vis_utils import GENDER_MAP, make_mano_sequence_wt
from scene_utils import (
    load_concerto_scene_points,
    get_recording_id_from_seq_name,
    get_hot3d_to_grab_transform,
    load_anchor,
    SCENE_DATA_ROOT,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HOT3D_OBJECT_MESH_UID = {
    'aria_small': '111305855142671',
    'birdhouse_toy': '195041665639898',
    'bottle_bbq': '106434519822892',
    'bottle_mustard': '261746112525368',
    'bottle_ranch': '70709727230291',
    'bowl': '194930206998778',
    'can_parmesan': '171735388712249',
    'can_soup': '98604936546412',
    'can_tomato_sauce': '183136364331389',
    'carton_milk': '125863066770940',
    'carton_oj': '204462113746498',
    'cellphone': '5462893327580',
    'coffee_pot': '228358276546933',
    'dino_toy': '208243983021975',
    'dumbbell_5lb': '27078911029651',
    'dvd_remote': '249541253457812',
    'flask': '4111539686391',
    'food_vegetables': '96945373046044',
    'food_waffles': '253405647833885',
    'holder_black': '232501673989606',
    'holder_gray': '143541090750632',
    'keyboard': '37787722328019',
    'mouse': '265826671143948',
    'mug_patterned': '117658302265452',
    'mug_white': '223371871635142',
    'plate_bamboo': '248247003992116',
    'potato_masher': '79582884925181',
    'puzzle_toy': '106957734975303',
    'spatula_red': '270231216246839',
    'spoon_wooden': '225397651484143',
    'vase': '163340972252026',
    'whiteboard_eraser': '258906041248094',
    'whiteboard_marker': '238686662724712',
}

HOT3D_MESH_PATH = './assets/hot3d_assets'


hand_r_mesh_color = (121 / 255.0, 119 / 255.0, 158 / 255.0, 1.0)
hand_l_mesh_color = (158 / 255.0, 121 / 255.0, 119 / 255.0, 1.0)
obj_mesh_color = (121 / 255.0, 140 / 255.0, 119 / 255.0, 1.0)
keyframe_color = (50 / 255, 131 / 255, 131 / 255, 206 / 255)
pc_hand_color = (200 / 255, 85 / 255, 85 / 255, 0.6)


def placeholder_box(extent: float = 0.05):
    """Create a NumPy 2-compatible axis-aligned fallback box."""
    h = extent / 2.0
    vertices = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def mano_fk_vertices(layer, hand_pose, global_orient, trans):
    """Run MANO hand articulation directly through smplx instead of aitviewer."""
    bm = layer.bm
    dev = next(bm.parameters()).device
    n = len(hand_pose)
    with torch.no_grad():
        out = bm(
            hand_pose=torch.as_tensor(hand_pose, dtype=torch.float32, device=dev),
            global_orient=torch.as_tensor(global_orient, dtype=torch.float32, device=dev),
            transl=torch.as_tensor(trans, dtype=torch.float32, device=dev),
            betas=torch.zeros((n, bm.num_betas), dtype=torch.float32, device=dev),
        )
    return out.vertices.cpu().numpy(), bm.faces.astype(np.int64)


def render_sequence(
    pose_data_path,
    keyframe_vis=False,
    pre_grasp=False,
    save_video=False,
    vis_gt=False,
    num_reps=1,
    resolution='high',
    range_min=0,
    range_max=20,
    dataset='grab',
    output_fps=30,
    show_scene=False,
    local_scenes_path=None,
    scene_point_size=3.0,
    scene_source='local',
    scene_max_points=50000,
):
    """Render generated or ground-truth sequences from the shared 117D motion layout."""
    assert range_max > range_min, "range_max must be greater than range_min"

    if resolution == 'high':
        resolution = (4000, 3000)
    elif resolution == 'medium':
        resolution = (2000, 1500)
    else:
        resolution = (500, 375)

    if save_video:
        try:
            v = HeadlessRenderer(size=resolution)
        except Exception as err:
            import moderngl
            _orig_ctx = moderngl.create_standalone_context
            moderngl.create_standalone_context = (
                lambda **kw: _orig_ctx(**{**kw, 'backend': 'egl'}))
            try:
                v = HeadlessRenderer(size=resolution)
            except Exception:
                raise RuntimeError(
                    'Could not create a rendering context. On headless '
                    'machines without EGL, run under `xvfb-run -a`.'
                    ) from err
            finally:
                moderngl.create_standalone_context = _orig_ctx
    else:
        v = Viewer(size=resolution)

    v.playback_fps = output_fps

    v.scene.camera.fov = 40
    v.scene.camera.target = [0.0, 0.218, -0.093]
    v.scene.camera.position = [0.0, 0.65, 2.152]
    v.scene.camera.up = [0, 1.0, 0]
    v.scene.floor.position = [0.0, -0.4, 0.0]
    v.auto_set_camera_target = False
    v.auto_set_floor = False
    v.scene.remove(v.scene.lights[0])
    v.scene.remove(v.scene.origin)

    text_file = pose_data_path.replace('.npy', '.txt')
    len_file = pose_data_path.replace('.npy', '_len.txt')
    feature_vec_in = (
        np.load(pose_data_path, allow_pickle=True)
    )

    with open(text_file, "r") as f:
        texts = np.asarray([f.read().splitlines()])[0]

    with open(len_file, "r") as f:
        lengths = np.asarray([f.read().splitlines()])[0]

    range_max = min(len(texts), range_max)

    if dataset == 'hot3d':
        file_names_path = "dataset/file_names_hot3d.txt"
    elif dataset == 'egodex':
        file_names_path = "dataset/EGODEX_HANDS/file_names.txt"
    else:
        file_names_path = "dataset/file_names.txt"

    id_dict = {}
    if os.path.exists(file_names_path):
        with open(file_names_path, "r") as file:
            for line in file:
                line = line.strip()
                seq_id, seq_name = line.split(",", 1)
                id_dict[seq_name] = seq_id


    count = 0
    sample_text_mapping = {}
    if save_video:
        gt_suffix = "_gt" if vis_gt else ""
        video_folder = Path(pose_data_path).parent / ("ours_videos" + gt_suffix)
        video_folder.mkdir(parents=True, exist_ok=True)
        json_path = video_folder / "sample_text_mapping.json"
        if json_path.exists():
            with json_path.open() as f:
                sample_text_mapping = json.load(f)

    import time
    total_sequences = (range_max - range_min)
    print(f"\nStarting visualization of {total_sequences} sequences")
    print(f"num_reps={num_reps}, range_min={range_min}, range_max={range_max}")
    print(f"save_video={save_video}, vis_gt={vis_gt}, dataset={dataset}")
    vis_start_time = time.time()

    for rep in range(num_reps):
        for i in range(range_min // num_reps, range_max // num_reps):
            seq_start_time = time.time()
            seq_name = feature_vec_in.item()['data_id'][i]
            if dataset in ['hot3d', 'egodex']:
                seq_id = seq_name
            else:
                seq_id = id_dict.get(seq_name, seq_name)

            if dataset == 'hot3d':
                parts = seq_name.split('_')
                sbj_id = parts[0]
                obj_name = '_'.join(parts[2:-1])
            elif dataset == 'egodex':
                sbj_id = 'generic'
                obj_name = 'object'
            else:
                sbj_id = seq_name.split('_')[0]
                obj_name = seq_name.split('_')[1]


            print(f"\nProcessing sequence {count+1}/{total_sequences}: {seq_name}")
            print(f"obj_name={obj_name}, sbj_id={sbj_id}")

            if dataset in ['hot3d', 'egodex']:
                sbj_vtemp_lhand = None
                sbj_vtemp_rhand = None
            else:
                sbj_vtemp_lhand = load_grab_mesh(
                    os.path.join("assets", GENDER_MAP[sbj_id], sbj_id + "_lhand.ply"))
                sbj_vtemp_rhand = load_grab_mesh(
                    os.path.join("assets", GENDER_MAP[sbj_id], sbj_id + "_rhand.ply"))

            if dataset == 'hot3d':
                num_pca = 15
                use_flat_hand_mean = False
                use_pca_for_mano = args.is_pca
            else:
                num_pca = 24
                use_flat_hand_mean = True
                use_pca_for_mano = args.is_pca

            if sbj_vtemp_lhand is not None:
                smpl_layer_lhand = SMPLLayer(
                    model_type="mano",
                    use_pca=False,
                    v_template=sbj_vtemp_lhand.vertices,
                    flat_hand_mean=use_flat_hand_mean,
                    is_rhand=False,
                    num_pca_comps=num_pca,
                )
                smpl_layer_rhand = SMPLLayer(
                    model_type="mano",
                    use_pca=False,
                    v_template=sbj_vtemp_rhand.vertices,
                    flat_hand_mean=use_flat_hand_mean,
                    is_rhand=True,
                    num_pca_comps=num_pca,
                )
            else:
                smpl_layer_lhand = SMPLLayer(
                    model_type="mano",
                    use_pca=False,
                    flat_hand_mean=use_flat_hand_mean,
                    is_rhand=False,
                    num_pca_comps=num_pca,
                )
                smpl_layer_rhand = SMPLLayer(
                    model_type="mano",
                    use_pca=False,
                    flat_hand_mean=use_flat_hand_mean,
                    is_rhand=True,
                    num_pca_comps=num_pca,
                )

            if vis_gt:
                if dataset == 'hot3d':
                    gt_path = os.path.join('dataset/HOT3D_HANDS/representation_full', seq_name + '.npy')
                elif dataset == 'egodex':
                    gt_path = os.path.join('dataset/EGODEX_HANDS/representation_grasp', seq_id + '.npy')
                else:
                    gt_path = os.path.join('dataset/GRAB_HANDS/representation_full', seq_id + '.npy')
                feature_vec = np.load(gt_path)
            else:
                feature_vec = feature_vec_in.item()["motion"][
                    rep * (range_max-range_min) // num_reps + i
                ][
                    0
                ]

            sample_idx = rep * (range_max - range_min) // num_reps + i
            if sample_idx < len(lengths):
                actual_len = int(lengths[sample_idx])
                if actual_len > 0 and actual_len < feature_vec.shape[0]:
                    feature_vec = feature_vec[:actual_len]

            if pre_grasp:
                pre_grasp_len = feature_vec.shape[0] // 2
                feature_vec = feature_vec[:pre_grasp_len]

            print(f"Loading object mesh for {obj_name}...")
            mesh_load_start = time.time()
            if dataset == 'hot3d':
                if obj_name not in HOT3D_OBJECT_MESH_UID:
                    raise KeyError(
                        f"Object '{obj_name}' has no entry in HOT3D_OBJECT_MESH_UID, "
                        f"so its mesh cannot be resolved.")
                mesh_uid = HOT3D_OBJECT_MESH_UID[obj_name]
                obj_mesh_path = os.path.join(HOT3D_MESH_PATH, mesh_uid + ".glb")
                if not os.path.exists(obj_mesh_path):
                    raise FileNotFoundError(
                        f"{obj_mesh_path} is missing (object '{obj_name}'). The HOT3D "
                        f"object models are not part of the data package: download the "
                        f"object library with the HOT3D toolkit as described in the "
                        f"README, into {HOT3D_MESH_PATH}/.")
                obj_mesh = trimesh.load(obj_mesh_path, force='mesh')
            elif dataset == 'egodex':
                obj_mesh = placeholder_box()
            else:
                obj_mesh = load_grab_mesh(
                    os.path.join("assets", "contact_meshes", obj_name + ".ply"))
            mesh_load_time = time.time() - mesh_load_start
            print(f"Mesh loaded in {mesh_load_time:.2f}s ({obj_mesh.vertices.shape[0]} vertices)")

            obj_verts = obj_mesh.vertices.copy()

            obj_rot = rotation_6d_to_matrix(torch.tensor(feature_vec[:, 111:117])).reshape(-1, 3, 3).numpy()
            pos_left = torch.tensor(feature_vec[:, :3], dtype=torch.float32).to(device)
            pos_right = torch.tensor(feature_vec[:, 3:6], dtype=torch.float32).to(device)
            global_orient_l = R.from_matrix(
                rotation_6d_to_matrix(torch.tensor(feature_vec[:, 30:36])).numpy()
            ).as_rotvec()
            global_orient_r = R.from_matrix(
                rotation_6d_to_matrix(torch.tensor(feature_vec[:, 60:66])).numpy()
            ).as_rotvec()
            if dataset == 'hot3d':
                joint_rotations_l = feature_vec[:, 6:21]
                joint_rotations_r = feature_vec[:, 36:51]
            else:
                joint_rotations_l = feature_vec[:, 6:30]
                joint_rotations_r = feature_vec[:, 36:60]

            if use_pca_for_mano and joint_rotations_l.shape[1] == num_pca:
                import smplx
                from aitviewer.configuration import CONFIG as _C
                comps = {}
                for key, is_r in (('l', False), ('r', True)):
                    m = smplx.create(
                        _C.smplx_models, model_type='mano', is_rhand=is_r,
                        use_pca=True, num_pca_comps=num_pca,
                        flat_hand_mean=use_flat_hand_mean)
                    comps[key] = m.hand_components.detach().cpu().numpy()
                joint_rotations_l = joint_rotations_l @ comps['l']
                joint_rotations_r = joint_rotations_r @ comps['r']

            if not save_video:
                pos_left[..., 0] += count * 2.0
                pos_right[..., 0] += count * 2.0

            if dataset == 'egodex':
                trans_l = pos_left.detach().cpu().numpy()
                trans_r = pos_right.detach().cpu().numpy()
            else:
                trans_l = (
                    pos_left
                    - smpl_layer_lhand.bm(hand_pose=torch.zeros((1, 45)).to(device)).joints[0, 0]
                ).detach().cpu().numpy()
                trans_r = (
                    pos_right
                    - smpl_layer_rhand.bm(hand_pose=torch.zeros((1, 45)).to(device))
                    .joints[0, 0]
                ).detach().cpu().numpy()

            verts_l, faces_l = mano_fk_vertices(
                smpl_layer_lhand, joint_rotations_l, global_orient_l, trans_l)
            verts_r, faces_r = mano_fk_vertices(
                smpl_layer_rhand, joint_rotations_r, global_orient_r, trans_r)

            print(f"Transforming object vertices ({obj_verts.shape[0]} frames x {obj_verts.shape[1]} verts)...")
            transform_start = time.time()
            obj_pos_slice = slice(108, 111)
            obj_verts = np.matmul(obj_verts, obj_rot.transpose(0, 2, 1))
            obj_verts += feature_vec[:, np.newaxis, obj_pos_slice]
            print(f"Object transform done in {time.time() - transform_start:.2f}s")

            if not save_video:
                obj_verts[..., 0] += count * 2.0
                obj_verts[..., 1] += rep * 2.0

            print("Making hands watertight...")
            wt_start = time.time()
            (
                verts_hand_l,
                verts_hand_r,
                faces_hand_l,
                faces_hand_r,
            ) = make_mano_sequence_wt(verts_l, verts_r, faces_l, faces_r)
            print(f"Watertight done in {time.time() - wt_start:.2f}s")

            print("Creating mesh objects...")
            mesh_create_start = time.time()
            mesh_frame = Meshes(
                obj_verts, obj_mesh.faces, z_up=True, color=obj_mesh_color
            )

            seq_l_mesh = Meshes(
                verts_hand_l, faces_hand_l, z_up=True, color=hand_l_mesh_color
            )
            seq_r_mesh = Meshes(
                verts_hand_r, faces_hand_r, z_up=True, color=hand_r_mesh_color
            )
            print(f"Mesh objects created in {time.time() - mesh_create_start:.2f}s")

            print("Adding meshes to scene...")
            scene_add_start = time.time()
            v.scene.add(mesh_frame, seq_l_mesh, seq_r_mesh)
            print(f"Meshes added to scene in {time.time() - scene_add_start:.2f}s")

            if show_scene and dataset == 'hot3d':
                scene_xyz = None
                pca_colors = None
                scene_load_start = time.time()

                if scene_source == 'local' and local_scenes_path is not None:
                    scene_path = Path(local_scenes_path) / f"{seq_name}.npz"
                    if scene_path.exists():
                        print(f"Loading scene from local npz: {scene_path}...")
                        scene_data = np.load(str(scene_path))
                        scene_xyz = scene_data['xyz'].astype(np.float32)
                        scene_features = scene_data['features'].astype(np.float32)

                        valid_mask = np.abs(scene_xyz).sum(axis=1) > 1e-6
                        scene_xyz = scene_xyz[valid_mask]
                        scene_features = scene_features[valid_mask]

                        print(f"Loaded {len(scene_xyz)} scene points")

                        from sklearn.decomposition import PCA
                        if len(scene_features) > 3:
                            pca = PCA(n_components=3)
                            pca_colors = pca.fit_transform(scene_features)
                            pca_min = pca_colors.min(axis=0)
                            pca_max = pca_colors.max(axis=0)
                            pca_range = pca_max - pca_min
                            pca_range[pca_range < 1e-6] = 1.0
                            pca_colors = (pca_colors - pca_min) / pca_range
                        else:
                            pca_colors = np.ones((len(scene_xyz), 3)) * 0.5
                    else:
                        print(
                            f"Warning: Scene file not found: {scene_path}. "
                            "The default local scene source needs a per-sequence "
                            "npz with xyz and features. For the per-recording "
                            "point clouds under scene_data/, pass "
                            "--scene_source concerto_grid.")

                elif scene_source in ['concerto_grid', 'concerto_filtered']:
                    print(f"Loading scene from concerto features ({scene_source})...")
                    try:
                        recording_id = get_recording_id_from_seq_name(seq_name)
                        use_grid = (scene_source == 'concerto_grid')
                        scene_xyz, pca_colors = load_concerto_scene_points(
                            recording_id,
                            scene_data_root=SCENE_DATA_ROOT,
                            max_points=scene_max_points,
                            use_grid=use_grid,
                        )
                        print(f"Loaded {len(scene_xyz)} scene points from concerto")

                        transform = get_hot3d_to_grab_transform()
                        scene_xyz = scene_xyz @ transform.T

                        anchor = load_anchor(seq_name)
                        if anchor is not None:
                            scene_xyz = scene_xyz - anchor
                            print("Applied anchor centering")
                        else:
                            print(f"Warning: No anchor found for {seq_name}")

                    except FileNotFoundError as e:
                        print(f"Warning: Concerto scene not found: {e}")

                if scene_xyz is not None and pca_colors is not None:
                    alpha = np.ones((len(scene_xyz), 1), dtype=np.float32)
                    pca_colors_rgba = np.concatenate([pca_colors, alpha], axis=1)

                    scene_pc = PointClouds(
                        points=scene_xyz[np.newaxis],
                        colors=pca_colors_rgba[np.newaxis],
                        point_size=scene_point_size,
                        z_up=True,
                    )
                    v.scene.add(scene_pc)
                    print(f"Scene point cloud added in {time.time() - scene_load_start:.2f}s")

            if keyframe_vis:
                feature_vec_kf = feature_vec_in.item()["gt_kf"][
                    rep * (range_max-range_min) // num_reps + i
                ]
                obj_rot_slice_kf = slice(111, 117)
                obj_rot_kf = rotation_6d_to_matrix(
                    torch.tensor(feature_vec_kf[:, obj_rot_slice_kf])
                ).reshape(-1, 3, 3)
                obj_verts_kf = obj_mesh.vertices.copy()
                obj_verts_kf = np.tile(
                    obj_verts_kf, (feature_vec_kf.shape[0], 1, 1)
                )

                pos_left_kf = feature_vec_kf[:, np.newaxis, :3]
                pos_right_kf = feature_vec_kf[:, np.newaxis, 3:6]

                if not save_video:
                    pos_left_kf[..., 0] += count * 2.0
                    pos_right_kf[..., 0] += count * 2.0
                    pos_left_kf[..., 1] += rep * 2.0
                    pos_right_kf[..., 1] += rep * 2.0

                global_orient_l_kf = R.from_matrix(
                    rotation_6d_to_matrix(
                        torch.tensor(feature_vec_kf[:, 30:36])
                    ).numpy()
                ).as_rotvec()
                global_orient_r_kf = R.from_matrix(
                    rotation_6d_to_matrix(
                        torch.tensor(feature_vec_kf[:, 60:66])
                    ).numpy()
                ).as_rotvec()

                if dataset == 'hot3d':
                    joint_rotations_l_kf = feature_vec_kf[:, 6:21]
                    joint_rotations_r_kf = feature_vec_kf[:, 36:51]
                else:
                    joint_rotations_l_kf = feature_vec_kf[:, 6:30]
                    joint_rotations_r_kf = feature_vec_kf[:, 36:60]

                if use_pca_for_mano and joint_rotations_l_kf.shape[1] == num_pca:
                    joint_rotations_l_kf = joint_rotations_l_kf @ comps['l']
                    joint_rotations_r_kf = joint_rotations_r_kf @ comps['r']

                if dataset == 'egodex':
                    trans_l_kf = pos_left_kf[:, 0]
                    trans_r_kf = pos_right_kf[:, 0]
                else:
                    trans_l_kf = (
                        pos_left_kf[:, 0]
                        - smpl_layer_lhand.bm(hand_pose=torch.zeros((1, 45)).to(device))
                        .joints[0, 0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    trans_r_kf = (
                        pos_right_kf[:, 0]
                        - smpl_layer_rhand.bm(hand_pose=torch.zeros((1, 45)).to(device))
                        .joints[0, 0]
                        .detach()
                        .cpu()
                        .numpy()
                    )


                verts_l_kf, faces_l_kf = mano_fk_vertices(
                    smpl_layer_lhand, joint_rotations_l_kf[-1:],
                    global_orient_l_kf[-1:], trans_l_kf[-1:])
                verts_r_kf, faces_r_kf = mano_fk_vertices(
                    smpl_layer_rhand, joint_rotations_r_kf[-1:],
                    global_orient_r_kf[-1:], trans_r_kf[-1:])

                obj_verts_kf = torch.matmul(
                    torch.tensor(obj_verts_kf, dtype=torch.float32), obj_rot_kf.transpose(1, 2)
                ).numpy()

                if not save_video:
                    obj_verts_kf[..., 0] += count * 2.0
                    obj_verts_kf[..., 1] += rep * 2.0

                obj_pos_slice_kf = slice(108, 111)
                obj_verts_kf += feature_vec_kf[:, np.newaxis, obj_pos_slice_kf]
                (
                    verts_hand_l_kf,
                    verts_hand_r_kf,
                    faces_hand_l_kf,
                    faces_hand_r_kf,
                ) = make_mano_sequence_wt(
                    verts_l_kf, verts_r_kf, faces_l_kf, faces_r_kf)

                seq_l_kf_mesh = Meshes(
                    verts_hand_l_kf, faces_hand_l, z_up=True, color=keyframe_color
                )
                seq_r_kf_mesh = Meshes(
                    verts_hand_r_kf, faces_hand_r, z_up=True, color=keyframe_color
                )

                mesh_frame_kf = Meshes(
                    obj_verts_kf[-1], obj_mesh.faces, z_up=True, color=keyframe_color
                )

                v.scene.add(seq_r_kf_mesh, seq_l_kf_mesh, mesh_frame_kf)


            if save_video:
                video_stem = str(count).zfill(4) + gt_suffix
                video_path = video_folder / (video_stem + ".mp4")
                suffix = 1
                while video_path.exists() or video_path.name in sample_text_mapping:
                    video_path = video_folder / f"{video_stem}_{suffix}.mp4"
                    suffix += 1
                print(f"Rendering and saving video: {video_path.name}")
                video_save_start = time.time()
                v.save_video(
                    video_dir=str(video_path),
                    frame_dir=str(video_path.with_suffix("")),
                    quality="high",
                    output_fps=output_fps,
                    # The collision check above selects the final filename.
                    # AITViewer otherwise adds a suffix that is absent from our index.
                    ensure_no_overwrite=False,
                )
                video_save_time = time.time() - video_save_start
                print(f"Video saved in {video_save_time:.2f}s")

                sample_idx = rep * (range_max - range_min) // num_reps + i
                sample_text_mapping[video_path.name] = {
                    "text": texts[sample_idx],
                    "seq_name": seq_name,
                    "length": int(lengths[sample_idx]),
                }
                # Preserve earlier runs, and record each completed video even if a
                # later sequence fails. Replace atomically to keep the index readable.
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=video_folder, prefix=".sample_text_mapping.",
                    suffix=".tmp", delete=False,
                ) as f:
                    json.dump(sample_text_mapping, f, indent=2)
                    temporary_mapping = f.name
                os.replace(temporary_mapping, json_path)

                nodes = v.scene.nodes.copy()
                for node in nodes:
                    if node.name in ["Meshes", "PointClouds"]:
                        v.scene.remove(node)
            count += 1
            seq_time = time.time() - seq_start_time
            elapsed = time.time() - vis_start_time
            avg_time = elapsed / count if count > 0 else 0
            remaining = avg_time * (total_sequences - count)
            print(f"Sequence {count}/{total_sequences} done in {seq_time:.2f}s "
                  f"(avg: {avg_time:.2f}s/seq, ETA: {remaining/60:.1f}min)")

    if not save_video:
        v.run()

        for k in range(num_reps):
            shutil.rmtree(os.path.join(pose_data_path.split(".")[0], str(k).zfill(4)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with command line arguments")

    parser.add_argument(
        "--folder_path", type=str, required=True, help="Folder path containing results_test_*.npy (objects_unseen, random, or scenes_unseen)"
    )
    parser.add_argument("--is_pca", action="store_true", default=True, help="If PCA representation is used for the pose representation")
    parser.add_argument("--pre_grasp", action="store_true", default=False, help="If only the pre-grasp pose is used for the visualization")
    parser.add_argument("--kf_vis", action="store_true", default=False, help="If the keyframes should be visualized")
    parser.add_argument("--vis_gt", action="store_true", default=False, help="If the GT poses should be visualized")
    parser.add_argument("--save_video", action="store_true", default=True, help="If a video should be saved instead of rendering in interactive mode")
    parser.add_argument("--num_reps", default=1, type=int, help="If multiple samples were generated for the same text prompt.")
    parser.add_argument("--range_min", default=0, type=int, help="The index of the first sequence to be visualized")
    parser.add_argument("--range_max", default=1000, type=int, help="The index of the last sequence to be visualized")
    parser.add_argument("--resolution", default="low", type=str, help="The resolution of the visualization. Either 'high', 'medium', or 'low'")
    parser.add_argument("--dataset", default="grab", type=str, choices=["grab", "hot3d", "egodex"], help="Dataset type: 'grab', 'hot3d', or 'egodex'")
    parser.add_argument("--output_fps", default=30, type=int, help="Output video FPS (HOT3D: 15, GRAB: 30)")
    parser.add_argument("--show_scene", action="store_true", default=False, help="Show local scene point cloud (HOT3D only)")
    parser.add_argument("--local_scenes_path", type=str, default="dataset/HOT3D_HANDS/local_scenes_5000",
                        help="Path to pre-computed local scenes directory")
    parser.add_argument("--scene_point_size", type=float, default=3.0, help="Point size for scene visualization")
    parser.add_argument("--scene_source", type=str, default="local",
                        choices=["local", "concerto_grid", "concerto_filtered"],
                        help="Scene source: 'local' (pre-computed npz), 'concerto_grid' (grid with PCA), 'concerto_filtered' (dense, uniform)")
    parser.add_argument("--scene_max_points", type=int, default=50000,
                        help="Maximum scene points for concerto modes")

    args = parser.parse_args()

    import glob as _glob
    candidates = sorted(_glob.glob(os.path.join(args.folder_path, "results_*.npy")))
    if not candidates:
        raise FileNotFoundError(f"No results_*.npy found in {args.folder_path}")
    file_path = candidates[0]

    render_sequence(
        file_path,
        args.kf_vis,
        args.pre_grasp,
        args.save_video,
        args.vis_gt,
        args.num_reps,
        args.resolution,
        args.range_min,
        args.range_max,
        args.dataset,
        args.output_fps,
        args.show_scene,
        args.local_scenes_path,
        args.scene_point_size,
        args.scene_source,
        args.scene_max_points,
    )
