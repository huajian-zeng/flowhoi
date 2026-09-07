"""Flow-matching conditioning and inpainting utilities."""

import numpy as np
import torch


class ClassifierFreeGuidance(torch.nn.Module):
    """Apply classifier-free guidance using the checkpoint's unconditional branch."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, y=None):
        y = {} if y is None else y
        scale = torch.as_tensor(y.get('scale', 1.0), device=x.device, dtype=x.dtype)
        if scale.numel() not in (1, x.shape[0]):
            raise ValueError('CFG scale must be a scalar or contain one value per sample')
        scale = scale.reshape(-1, *([1] * (x.ndim - 1)))
        conditional = self.model(x, timesteps, y=y)
        if bool(torch.all(scale == 1)):
            return conditional
        unconditional = self.model(x, timesteps, y={**y, 'uncond': True})
        return unconditional + scale * (conditional - unconditional)


def build_interaction_target(grasp_trajectory, initial_object_pose, num_frames, feature_dim):
    """Build the physical hand and static-object target used for inpainting."""
    batch_size, hand_dim, _, grasp_frames = grasp_trajectory.shape
    object_end = hand_dim + initial_object_pose.shape[1]
    if grasp_frames > num_frames or object_end > feature_dim:
        raise ValueError('Grasp trajectory or object pose exceeds the interaction shape')
    pose = grasp_trajectory.new_zeros(batch_size, feature_dim, 1, num_frames)
    mask = torch.zeros_like(pose, dtype=torch.bool)
    pose[:, :hand_dim, :, :grasp_frames] = grasp_trajectory
    pose[:, hand_dim:object_end, 0, :grasp_frames] = initial_object_pose.unsqueeze(-1)
    mask[:, :object_end, :, :grasp_frames] = True
    return pose, mask


def make_interaction_guidance(pose, mask, motion_dataset, object_pose_slice=(108, 117), **kwargs):
    """Reproduce the released checkpoints' stable gradient surrogate.

    The original sampler omits the ill-conditioned inverse projection in this
    gradient calculation and uses zero object channels in its surrogate target.
    This is a checkpoint-compatible guidance heuristic, not a physical-pose
    reconstruction loss. Hard inpainting keeps the separate, complete object
    pose supplied by ``build_interaction_target``.
    """
    surrogate_target = pose.clone()
    object_start, object_end = object_pose_slice
    surrogate_target[:, object_start:object_end] = 0
    return CondKeyLocationsFlowMatching(
        target=surrogate_target.permute(0, 3, 2, 1),
        target_mask=mask.permute(0, 3, 2, 1),
        inv_transform=motion_dataset.inv_transform_th,
        use_rand_projection=False,
        **kwargs,
    )


def get_grasp_references(gt_motions, sampled_keyframes, dataset, feat_dim_traj=117, feat_dim_motion=117, max_len=200):
    """Build grasp targets and linearly interpolated trajectory and object-pose masks."""
    batch_size = gt_motions.shape[0]
    data_device = gt_motions.device

    target = torch.zeros([batch_size, max_len, 1, feat_dim_traj], device=data_device)
    target_mask = torch.zeros_like(target, dtype=torch.bool)

    traj_only_idcs = np.arange(0, feat_dim_traj)

    for idx in range(batch_size):
        key_posi = gt_motions[idx, 0, :, :].permute(1, 0)
        for kframe_idx, kframe in enumerate(sampled_keyframes):
            target[idx, kframe, 0] = key_posi[kframe_idx, traj_only_idcs]
            target_mask[idx, kframe, 0] = True

    inpaint_traj = torch.zeros([batch_size, feat_dim_traj, 1, max_len], device=data_device)
    inpaint_traj_mask = torch.zeros_like(inpaint_traj, dtype=torch.bool)
    inpaint_traj_points = torch.zeros_like(inpaint_traj)
    inpaint_traj_mask_points = torch.zeros_like(inpaint_traj_mask)
    inpaint_traj_points_filled = torch.zeros_like(inpaint_traj)

    inpaint_motion = torch.zeros([batch_size, feat_dim_motion, 1, max_len], device=data_device)
    inpaint_mask = torch.zeros_like(inpaint_motion, dtype=torch.bool)
    inpaint_motion_points = torch.zeros_like(inpaint_motion)
    inpaint_mask_points = torch.zeros_like(inpaint_mask)

    for idx in range(batch_size):
        key_positions = gt_motions[idx, 0, :, :].permute(1, 0)
        cur_key_pos = key_positions[0]
        last_kframe = 0

        for kframe_id, kframe_t in enumerate(sampled_keyframes):
            diff = kframe_t - last_kframe
            key_pos = key_positions[kframe_id, traj_only_idcs]

            for i in range(diff):
                inpaint_traj[idx, :, 0, last_kframe + i] = cur_key_pos + (key_pos - cur_key_pos) * i / diff
                inpaint_traj_mask[idx, :, 0, last_kframe + i] = True

                if i > 0:
                    inpaint_traj_points_filled[idx, :, 0, last_kframe + i] = cur_key_pos + (key_pos - cur_key_pos) * i / diff

            inpaint_traj_points[idx, :, 0, kframe_t] = key_pos
            inpaint_traj_mask_points[idx, :, 0, kframe_t] = True
            inpaint_traj_points_filled[idx, :, 0, kframe_t] = key_pos

            cur_key_pos = key_pos
            last_kframe = kframe_t

            if kframe_id == len(sampled_keyframes) - 1:
                inpaint_traj[idx, :, 0, kframe_t] = key_pos
                inpaint_traj_mask[idx, :, 0, kframe_t] = True
                inpaint_traj_points_filled[idx, :, 0, kframe_t] = key_pos

    inpaint_motion[:, traj_only_idcs, :, :] = inpaint_traj[:, :, :, :]
    inpaint_motion_points[:, traj_only_idcs, :, :] = inpaint_traj_points[:, :, :, :]
    inpaint_mask[:, traj_only_idcs, :, :] = inpaint_traj_mask[:, :, :, :]
    inpaint_mask_points[:, traj_only_idcs, :, :] = inpaint_traj_mask_points[:, :, :, :]

    if feat_dim_motion > feat_dim_traj and gt_motions.shape[2] > feat_dim_traj:
        obj_dim_start = feat_dim_traj
        obj_dim_end = min(feat_dim_motion, gt_motions.shape[2])
        obj_idcs = np.arange(obj_dim_start, obj_dim_end)

        for idx in range(batch_size):
            obj_positions = gt_motions[idx, 0, obj_dim_start:obj_dim_end, :].permute(1, 0)
            cur_obj_pos = obj_positions[0]
            last_kframe = 0

            for kframe_id, kframe_t in enumerate(sampled_keyframes):
                diff = kframe_t - last_kframe
                obj_pos = obj_positions[kframe_id]

                for i in range(diff):
                    interp_obj = cur_obj_pos + (obj_pos - cur_obj_pos) * i / diff
                    inpaint_motion[idx, obj_idcs, 0, last_kframe + i] = interp_obj
                    inpaint_mask[idx, obj_idcs, 0, last_kframe + i] = True

                inpaint_motion_points[idx, obj_idcs, 0, kframe_t] = obj_pos
                inpaint_mask_points[idx, obj_idcs, 0, kframe_t] = True

                cur_obj_pos = obj_pos
                last_kframe = kframe_t

                if kframe_id == len(sampled_keyframes) - 1:
                    inpaint_motion[idx, obj_idcs, 0, kframe_t] = obj_pos
                    inpaint_mask[idx, obj_idcs, 0, kframe_t] = True

    inpaint_traj = dataset.motion_dataset.transform_th(
        inpaint_traj[:, :feat_dim_traj].permute(0, 2, 3, 1),
        use_rand_proj=False
    ).permute(0, 3, 1, 2)

    inpaint_traj_points = dataset.motion_dataset.transform_th(
        inpaint_traj_points[:, :feat_dim_traj].permute(0, 2, 3, 1),
        use_rand_proj=False
    ).permute(0, 3, 1, 2)

    return (
        target, target_mask,
        inpaint_traj, inpaint_traj_mask,
        inpaint_traj_points, inpaint_traj_mask_points,
        inpaint_motion, inpaint_mask,
        inpaint_motion_points, inpaint_mask_points,
        inpaint_traj_points_filled
    )


class CondKeyLocationsFlowMatching:
    """Guide flow matching toward masked grasp targets in pose space."""

    def __init__(
        self,
        target,
        target_mask,
        inv_transform,
        classifier_scale=50.0,
        use_mse_loss=False,
        use_rand_projection=False,
        cut_frame=200,
        stop_cond_from=0.0,
    ):
        self.target = target
        self.target_mask = target_mask
        self.inv_transform = inv_transform
        self.classifier_scale = classifier_scale
        self.use_mse_loss = use_mse_loss
        self.use_rand_projection = use_rand_projection
        self.cut_frame = cut_frame
        self.stop_cond_from = stop_cond_from

    def __call__(self, x_t, t, y=None):
        """Guide [B, C, 1, T] samples until the cutoff, with t=0 noise and t=1 data."""
        t_val = t[0].item() if t.dim() > 0 else float(t)
        if t_val > self.stop_cond_from:
            return torch.zeros_like(x_t)

        if y is None:
            y = {}

        traject_only = bool(y.get("grasp_model", False))

        use_rand_proj = False if traject_only else self.use_rand_projection

        x_pose = self.inv_transform(
            x_t.permute(0, 2, 3, 1),
            traject_only=traject_only,
            use_rand_proj=use_rand_proj,
        )
        traj = x_pose.permute(0, 2, 1, 3)[:, :self.cut_frame, 0]

        target = self.target[:, :self.cut_frame, 0, :]
        mask   = self.target_mask[:, :self.cut_frame, 0, :]

        if self.use_mse_loss:
            loss = (traj - target) ** 2
        else:
            loss = (traj - target).abs()

        loss = (loss * mask).sum()

        grad = torch.autograd.grad(-loss, x_t)[0]

        return grad * self.classifier_scale
