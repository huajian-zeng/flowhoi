# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch

from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_rotation_6d


def sample_to_hand_motion(sample_list, args, model_kwargs, model, n_frames,
                     data_inv_transform_fn):

    if not isinstance(sample_list, list):
        sample_list = [sample_list]

    all_motions, all_motions_pose_space, all_lengths, all_text = [], [], [], []
    for sample in sample_list:
        sample_pose_space = data_inv_transform_fn(
            sample.cpu().permute(0, 2, 3, 1)).float()

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

        all_motions.append(sample.cpu().numpy())
        all_motions_pose_space.append(sample_pose_space.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    return all_motions_pose_space, all_motions, all_lengths, all_text

def recover_from_ric(data, object_rot_relative = True, add_obj_pos = True, num_joints=1):

        """Recover hand positions and orientations as NumPy arrays."""
        data = torch.tensor(data, dtype=torch.float32)

        obj_rot = rotation_6d_to_matrix(data[:,0,:,-6:])
        obj_pos = data[:,0,:,-9:-6]

        positions_left = data[..., : num_joints * 3].swapaxes(1,2)
        positions_right = data[..., num_joints * 3: num_joints * 3 * 2].swapaxes(1,2)

        '''Add rotation to local joint positions'''
        if object_rot_relative:
            positions_left = torch.matmul(obj_rot,positions_left.swapaxes(2,3)).permute(0,1,3,2)
            positions_right = torch.matmul(obj_rot,positions_right.swapaxes(2,3)).permute(0,1,3,2)
        if add_obj_pos:
            '''Add obj root to joints'''
            positions_left += obj_pos[:,:,np.newaxis]
            positions_right += obj_pos[:,:,np.newaxis]
        '''Concate root and joints'''

        global_orient_l = rotation_6d_to_matrix(data[:,0,:,num_joints * 3 * 2 + 24: num_joints * 3 * 2 + 30])
        global_orient_r = rotation_6d_to_matrix(data[:,0,:,num_joints * 3 * 2 + 54: num_joints * 3 * 2 + 60])

        if object_rot_relative:
            global_orient_l =  matrix_to_rotation_6d(torch.matmul(obj_rot,global_orient_l))
            global_orient_r = matrix_to_rotation_6d(torch.matmul(obj_rot,global_orient_r))
        else:
            global_orient_l =  matrix_to_rotation_6d(global_orient_l)
            global_orient_r = matrix_to_rotation_6d(global_orient_r)

        return (positions_left.numpy(), positions_right.numpy(),
                global_orient_l.numpy(), global_orient_r.numpy())


def stich_pregrasp(data, pos_left, pos_right, add_obj_pos = False, num_joints=1):

        data = torch.tensor(data, dtype=torch.float32)

        positions_left = data[..., : num_joints * 3].swapaxes(1,2)
        positions_right = data[..., num_joints * 3: num_joints * 3 * 2].swapaxes(1,2)

        global_orient_l = rotation_6d_to_matrix(data[:,0,:,num_joints * 3 * 2 + 24: num_joints * 3 * 2 + 30])
        global_orient_r = rotation_6d_to_matrix(data[:,0,:,num_joints * 3 * 2 + 54: num_joints * 3 * 2 + 60])
        
        wrist_pose_l = rotation_6d_to_matrix(data[:,0,:,num_joints * 3 * 2 + 24: num_joints * 3 * 2 + 30])
        wrist_pose_r = rotation_6d_to_matrix(data[:,0,:,num_joints * 3 * 2 + 54: num_joints * 3 * 2 + 60])

        wrist_pose_l = matrix_to_rotation_6d(torch.matmul(global_orient_l[:,-1:],wrist_pose_l))
        wrist_pose_r = matrix_to_rotation_6d(torch.matmul(global_orient_r[:,-1:],wrist_pose_r))

        wrist_pos_l = np.matmul(global_orient_l[:,-1:].numpy(),positions_left.swapaxes(2,3)).swapaxes(2,3)
        wrist_pos_r = np.matmul(global_orient_r[:,-1:].numpy(),positions_right.swapaxes(2,3)).swapaxes(2,3)

        wrist_pos_l += pos_left[:,-1:]
        wrist_pos_r += pos_right[:,-1:]

        if add_obj_pos:
            obj_pos = data[:,0,:,-9:-6]
            wrist_pos_l += obj_pos[:,:, np.newaxis]
            wrist_pos_r += obj_pos[:,:, np.newaxis]

        return wrist_pos_l.numpy(), wrist_pos_r.numpy(), wrist_pose_l.numpy(), wrist_pose_r.numpy()

