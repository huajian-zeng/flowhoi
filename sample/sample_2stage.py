"""Usage:
    python -m sample.sample_2stage \\
        --model_path save/hot3d_full/model.pt \\
        --grasp_model_path save/egodex_grasp/model.pt \\
        --guidance --num_samples 1
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import time
import numpy as np
import torch

from data_loaders.humanml.utils.data_utils import REPRESENTATION_IDCS
from utils import dist_util
from utils.fixseed import fixseed
from utils.generation_template import get_template
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.output_util import sample_to_hand_motion, stich_pregrasp, recover_from_ric
from utils.parser_util import generate_args
from sample.batching import align_stage_loaders, iter_stage_batches
from sample.condition_hands import (
    ClassifierFreeGuidance,
    build_interaction_target,
    make_interaction_guidance,
)

from sample.sampling_utils import (
    load_dataset,
    attach_scene_inputs,
    setup_output_path,
    save_args,
    fix_ambiguous_object_names,
    get_conditioning_thresholds,
    sample_flow_matching,
    save_results,
)


def get_grasp_references_unified(is_flow_matching: bool):
    """Get the grasp references function (flow matching only in this release)."""
    if not is_flow_matching:
        raise ValueError('This release supports flow-matching checkpoints only')
    from sample.condition_hands import get_grasp_references
    return get_grasp_references


def get_guidance_class(is_flow_matching: bool):
    """Get the guidance class (flow matching only in this release)."""
    if not is_flow_matching:
        raise ValueError('This release supports flow-matching checkpoints only')
    from sample.condition_hands import CondKeyLocationsFlowMatching
    return CondKeyLocationsFlowMatching


def load_grasp_model(data, args_traj):
    """Load grasp model for trajectory prediction."""
    grasp_model, grasp_diffusion = create_model_and_diffusion(args_traj, data)

    print(f"Loading grasp model checkpoints from [{args_traj.grasp_model_path}]...")
    load_saved_model(grasp_model, args_traj.grasp_model_path)

    grasp_model.to(dist_util.dev())
    grasp_model.eval()
    return grasp_model, grasp_diffusion


def load_stage_datasets(grasp_args, interaction_args, grasp_frames, interaction_frames):
    """Keep the full-motion loader's cap independent of the grasp prefix."""
    return (load_dataset(grasp_args, grasp_frames, grasp_frames, shuffle=False),
            load_dataset(interaction_args, interaction_frames, interaction_frames, shuffle=False))


def main():
    """Run two-stage inference with unprojected grasp and projected interaction inpainting."""
    args = generate_args()
    print(f"Architecture: {args.arch}")

    is_flow_matching = getattr(args, 'use_flow_matching', False)
    print(f"Guidance: {args.guidance} | Inpainting: {args.inpainting}")

    args = get_template(args, guidance=args.guidance)

    args_interaction = args
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '').replace('_best', '_best')

    args = generate_args(model_path=args.grasp_model_path)

    dataset_attrs = [
        'dataset', 'data_dir', 'motion_representation', 'split_file',
        'mean_name', 'std_name', 'proj_matrix_name',
        'split_set',
    ]
    mean_std_mapping = {
        'Mean_grab_full.npy': 'Mean_grab_grasp.npy',
        'Std_grab_full.npy': 'Std_grab_grasp.npy',
        'rand_proj_grab.npy': 'rand_proj_grab_grasp.npy',
        'rand_proj_grab_full.npy': 'rand_proj_grab_grasp.npy',
        'Mean_hot3d_full.npy': 'Mean_hot3d_grasp.npy',
        'Std_hot3d_full.npy': 'Std_hot3d_grasp.npy',
        'rand_proj_hot3d_full.npy': 'rand_proj_hot3d_grasp.npy',
        'Mean_egodex_full.npy': 'Mean_egodex_grasp.npy',
        'Std_egodex_full.npy': 'Std_egodex_grasp.npy',
        'rand_proj_egodex_full.npy': 'rand_proj_egodex_grasp.npy',
    }
    for attr in dataset_attrs:
        if hasattr(args_interaction, attr):
            old_val = getattr(args, attr, None)
            new_val = getattr(args_interaction, attr)
            if attr in ['mean_name', 'std_name', 'proj_matrix_name'] and new_val in mean_std_mapping:
                new_val = mean_std_mapping[new_val]
            if old_val != new_val:
                print(f"Overriding grasp model {attr}: {old_val} -> {new_val}")
                setattr(args, attr, new_val)

    if is_flow_matching:
        if not getattr(args, 'use_flow_matching', False):
            print("Warning: use_flow_matching not set in grasp model args, enabling it")
            args.use_flow_matching = True
        if not getattr(args_interaction, 'use_flow_matching', False):
            print("Warning: use_flow_matching not set in interaction model args, enabling it")
            args_interaction.use_flow_matching = True

    fixseed(args.seed)
    out_path = args_interaction.output_dir

    get_grasp_references = get_grasp_references_unified(is_flow_matching)
    CondKeyLocations = get_guidance_class(is_flow_matching)

    n_frames = args.max_frames_interaction if not args.pre_grasp else args.max_frames_grasp
    n_frames_post = args.max_frames_interaction

    dist_util.setup_dist(args.device)

    out_path = setup_output_path(args_interaction, niter, "2stage")

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'

    args.batch_size = args.num_samples
    args_interaction.batch_size = args.num_samples

    print('Loading dataset...')
    data, data_interaction = load_stage_datasets(args, args_interaction, n_frames, n_frames_post)

    data, data_interaction = align_stage_loaders(
        data, data_interaction, random_order=args.random_order, seed=args.seed)
    if args.num_repetitions < 1:
        raise ValueError('num_repetitions must be positive.')
    num_batches = len(data_interaction) if args.eval_entire_set else 1
    print(f"DATA BATCHES: {num_batches}; REPETITIONS PER CONDITION: {args.num_repetitions}")

    impute_override = getattr(args_interaction, 'impute_until_override', None)
    cond_until, impute_until_grasp, impute_until_interaction = get_conditioning_thresholds(
        impute_until_override=impute_override)

    all_motions = []
    all_gt_kf = []
    all_lengths = []
    all_data_id = []
    all_text = []

    inference_times = []
    batch_sizes = []
    repetition_indices = []

    pos_left_idcs = REPRESENTATION_IDCS['pos_left']
    pos_right_idcs = REPRESENTATION_IDCS['pos_right']
    glob_orient_l_idcs = REPRESENTATION_IDCS['global_orient_l']
    glob_orient_r_idcs = REPRESENTATION_IDCS['global_orient_r']
    obj_pose_idcs = REPRESENTATION_IDCS['object_pose']

    batches = iter_stage_batches(data, data_interaction, args.num_repetitions, args.eval_entire_set)
    for rep_i, (batch_index, repetition, grasp_batch, interaction_batch) in enumerate(batches):
        print(f"BATCH {batch_index}; REP {repetition}")
        model_device = dist_util.dev()
        motion_gt_grasp, model_kwargs_grasp = grasp_batch
        motion_gt, model_kwargs = interaction_batch
        texts_pre = model_kwargs_grasp['y']['text']
        texts = model_kwargs['y']['text']
        batch_size = motion_gt.shape[0]
        grasp_batch_args, interaction_batch_args = copy.copy(args), copy.copy(args_interaction)
        for batch_args in (grasp_batch_args, interaction_batch_args):
            batch_args.num_samples = batch_args.batch_size = batch_size

        texts_pre, texts = fix_ambiguous_object_names(texts_pre, texts, dataset=args.dataset)
        model_kwargs_grasp['y']['text'] = texts_pre
        model_kwargs['y']['text'] = texts

        attach_scene_inputs(args_interaction, model_kwargs, model_device)

        key_frames = [0, 49]

        motion_gt = data_interaction.dataset.motion_dataset.inv_transform_th(
            motion_gt.cpu().permute(0, 2, 3, 1)).float().permute(0, 1, 3, 2)
        motion_gt_kf = motion_gt[..., key_frames]

        save_args(args_interaction, out_path)

        (target, target_mask,
         inpaint_traj_p2p, inpaint_traj_mask_p2p,
         inpaint_traj_points, inpaint_traj_mask_points,
         inpaint_motion_p2p, inpaint_mask_p2p,
         inpaint_motion_points, inpaint_mask_points,
         inpaint_traj_points_filled) = get_grasp_references(
            motion_gt_kf, key_frames, data.dataset,
            max_len=n_frames, feat_dim_traj=motion_gt_grasp.shape[1], feat_dim_motion=117)

        model_kwargs_grasp['y']['grasp_model'] = args.traj_only
        model_kwargs['y']['grasp_model'] = args_interaction.traj_only

        grasp_model, grasp_diffusion = load_grasp_model(data, args)
        grasp_model_kwargs = copy.deepcopy(model_kwargs_grasp)
        grasp_model_kwargs['y']['log_name'] = out_path
        grasp_model_kwargs['y']['grasp_model'] = True
        args.do_inpaint = True

        if getattr(args, 'use_initial_pos', False):
            initial_pos = motion_gt_kf[:, 0, :, 0]
            grasp_model_kwargs['y']['initial_pos'] = initial_pos.to(model_device)
            print(f"[Initial Pos Conditioning] Set initial_pos shape: {initial_pos.shape}, "
                  f"range: [{initial_pos.min().item():.4f}, {initial_pos.max().item():.4f}]")

        target = target[..., :motion_gt_grasp.shape[1]].to(model_device)
        target_mask = target_mask[..., :motion_gt_grasp.shape[1]].to(model_device)
        model_kwargs['y']['target'] = target
        model_kwargs['y']['target_mask'] = target_mask

        # Keep the grasp checkpoint's original sampling conditions: its pose
        # branch receives no x_start; initial-state guidance is applied below.
        model_kwargs['y']['scale'] = args_interaction.guidance_param
        grasp_model_kwargs['y']['scale'] = args.guidance_param

        print("##### Stage 1: Generating grasp trajectory #####")
        grasp_model_kwargs['y']['log_id'] = 0

        grasp_model_kwargs['y']['inpainted_motion'] = inpaint_traj_p2p[:, :motion_gt_grasp.shape[1]].to(model_device)
        grasp_model_kwargs['y']['inpainting_mask'] = inpaint_traj_mask_p2p[:, :motion_gt_grasp.shape[1]].to(model_device)
        grasp_model_kwargs['y']['do_inpainting'] = args.inpainting

        grasp_model_kwargs['y']['cond_until'] = cond_until
        grasp_model_kwargs['y']['impute_until'] = impute_until_grasp
        grasp_model_kwargs['y']['impute_until_second_stage'] = cond_until
        grasp_model_kwargs['y']['inpainted_motion_second_stage'] = inpaint_traj_points[:, :motion_gt_grasp.shape[1]].to(model_device)
        grasp_model_kwargs['y']['inpainting_mask_second_stage'] = inpaint_traj_mask_points[:, :motion_gt_grasp.shape[1]].to(model_device)

        grasp_diffusion.data_transform_fn = data.dataset.motion_dataset.transform_th
        grasp_diffusion.data_inv_transform_fn = data.dataset.motion_dataset.inv_transform_th
        grasp_diffusion.data_get_mean_fn = data.dataset.motion_dataset.get_std_mean

        if args.guidance:
            CondKeyLocationsClass = get_guidance_class(is_flow_matching)
            if is_flow_matching:
                cond_fn_traj = CondKeyLocationsClass(
                    target=target,
                    target_mask=target_mask,
                    inv_transform=data.dataset.motion_dataset.inv_transform_th,
                    classifier_scale=args.classifier_scale,
                    use_mse_loss=args.gen_mse_loss,
                    use_rand_projection=False,
                    cut_frame=n_frames,
                    stop_cond_from=grasp_model_kwargs['y'].get('cond_until', 0),
                )
            else:
                cond_fn_traj = CondKeyLocations(
                    target=target,
                    target_mask=target_mask,
                    transform=data.dataset.motion_dataset.transform_th,
                    inv_transform=data.dataset.motion_dataset.inv_transform_th,
                    abs_3d=args.abs_3d,
                    classifiler_scale=args.classifier_scale,
                    use_mse_loss=args.gen_mse_loss,
                    use_rand_projection=False,
                )
        else:
            cond_fn_traj = None

        shape_grasp = (batch_size, grasp_model.njoints, grasp_model.nfeats, n_frames)
        inference_start_time = time.time()
        grasp_sample = sample_flow_matching(
            ClassifierFreeGuidance(grasp_model), grasp_diffusion, shape_grasp,
            grasp_model_kwargs, cond_fn=cond_fn_traj)
        stage1_time = time.time() - inference_start_time
        print(f"Stage 1 inference time: {stage1_time:.4f}s")

        gen_eff_len = min(grasp_sample[0].shape[-1], n_frames)
        print(f'cut the grasp motion length to {gen_eff_len}')
        for j in range(len(grasp_sample)):
            grasp_sample[j] = grasp_sample[j][:, :, :, :gen_eff_len]

        grasp_inv_transform = lambda x: data.dataset.motion_dataset.inv_transform_th(
            x, use_rand_proj=False)
        cur_motions_grasp_pose_space, _, _, _ = sample_to_hand_motion(
            grasp_sample, grasp_batch_args, grasp_model_kwargs, grasp_model, gen_eff_len,
            grasp_inv_transform)

        print("##### Stage 2: Generating full interaction #####")
        model, diffusion = create_model_and_diffusion(args_interaction, data_interaction)
        print(f"Loading checkpoints from [{args_interaction.model_path}]...")
        load_saved_model(model, args_interaction.model_path)

        model.to(dist_util.dev())
        model.eval()

        model_kwargs['y']['do_inpainting'] = args_interaction.inpainting

        grasp_traj_tensor = torch.tensor(
            np.array(cur_motions_grasp_pose_space)[-1]).permute(0, 3, 1, 2)[:, :obj_pose_idcs[0], :, -n_frames:].to(model_device)
        interaction_target_pose, interaction_target_mask = build_interaction_target(
            grasp_traj_tensor,
            motion_gt[:, 0, obj_pose_idcs[0]:obj_pose_idcs[1], 0].to(model_device),
            n_frames_post,
            data_interaction.dataset.motion_dataset.mean.shape[0],
        )
        model_kwargs['y']['inpainting_mask'] = interaction_target_mask
        model_kwargs['y']['inpainted_motion'] = data_interaction.dataset.motion_dataset.transform_th(
            interaction_target_pose.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        model_kwargs['y']['inpainted_in_pose_space'] = False

        if getattr(args_interaction, 'use_grasp_condition', False):
            grasp_pose = grasp_traj_tensor[:, :, 0, -1]
            model_kwargs['y']['grasp_pose'] = grasp_pose
            print(f"[Grasp Conditioning] Using grasp pose (last frame) shape: {grasp_pose.shape}")

        if getattr(args_interaction, 'use_pose_cond', False):
            grasp_frame = getattr(args_interaction, 'grasp_frame', 49)
            feature_dim = data_interaction.dataset.motion_dataset.mean.shape[0]

            x_start_pose = torch.zeros(
                (batch_size, feature_dim, 1, n_frames_post),
                dtype=torch.float32).to(model_device)

            x_start_pose[:, :, 0, 0] = motion_gt[:, 0, :, 0]

            x_start_pose[:, :obj_pose_idcs[0], 0, grasp_frame] = grasp_traj_tensor[:, :, 0, -1]
            x_start_pose[:, obj_pose_idcs[0]:obj_pose_idcs[1], 0, grasp_frame] = motion_gt[:, 0, obj_pose_idcs[0]:obj_pose_idcs[1], 0]

            x_start_normalized = data_interaction.dataset.motion_dataset.transform_th(
                x_start_pose.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            model_kwargs['y']['x_start'] = x_start_normalized
            print("[Pose Conditioning] Using x_start for mask-based conditioning")
            print(f"  x_start shape: {x_start_normalized.shape}")
            print(f"  Conditioned frames: 0 and {grasp_frame}")

        if args_interaction.pre_grasp:
            model_kwargs['y']['inpainted_motion'] = inpaint_traj_p2p.to(model_device)
            model_kwargs['y']['inpainting_mask'] = inpaint_traj_mask_p2p.to(model_device)
            model_kwargs['y']['inpainted_in_pose_space'] = False
            args.do_inpaint = True

        if not args.do_inpaint and "inpainted_motion" in model_kwargs['y'].keys():
            del model_kwargs['y']['inpainted_motion']
            del model_kwargs['y']['inpainting_mask']

        model_kwargs['y']['log_id'] = rep_i
        model_kwargs['y']['cond_until'] = cond_until
        model_kwargs['y']['impute_until'] = impute_until_interaction

        diffusion.data_get_mean_fn = data_interaction.dataset.motion_dataset.get_std_mean
        diffusion.data_transform_fn = data_interaction.dataset.motion_dataset.transform_th
        diffusion.data_inv_transform_fn = data_interaction.dataset.motion_dataset.inv_transform_th

        if args.guidance:
            cond_fn = make_interaction_guidance(
                interaction_target_pose,
                interaction_target_mask,
                data_interaction.dataset.motion_dataset,
                object_pose_slice=obj_pose_idcs,
                classifier_scale=args_interaction.classifier_scale,
                use_mse_loss=args_interaction.gen_mse_loss,
                cut_frame=n_frames_post,
                stop_cond_from=model_kwargs['y'].get('cond_until', 0),
            )
        else:
            cond_fn = None

        shape_interaction = (batch_size, model.njoints, model.nfeats, n_frames_post)
        stage2_start_time = time.time()
        sample = sample_flow_matching(
            ClassifierFreeGuidance(model), diffusion, shape_interaction,
            model_kwargs, cond_fn=cond_fn)
        stage2_time = time.time() - stage2_start_time
        print(f"Stage 2 inference time: {stage2_time:.4f}s")

        batch_inference_time = stage1_time + stage2_time
        per_sample_time = batch_inference_time / batch_size
        inference_times.append(per_sample_time)
        batch_sizes.append(batch_size)
        print(f"Total batch inference time: {batch_inference_time:.4f}s "
              f"({per_sample_time:.4f}s per sample)")

        gen_eff_len = min(sample[0].shape[-1], n_frames_post)
        print(f'cut the interaction motion length to {gen_eff_len}')
        for j in range(len(sample)):
            sample[j] = sample[j][:, :, :, :gen_eff_len]

        args_interaction.num_dump_step = 1

        interaction_inv_transform = data_interaction.dataset.motion_dataset.inv_transform_th
        cur_motions_pose_space, _, cur_lengths, cur_texts = sample_to_hand_motion(
            sample, interaction_batch_args, model_kwargs, model, gen_eff_len,
            interaction_inv_transform)

        cur_motions_pose_space = np.array(cur_motions_pose_space)[0]

        positions_left, positions_right, global_orient_l, global_orient_r = recover_from_ric(
            cur_motions_pose_space, object_rot_relative=False, add_obj_pos=False)
        cur_motions_pose_space[..., :pos_left_idcs[1]] = positions_left.swapaxes(1, 2)
        cur_motions_pose_space[..., pos_right_idcs[0]:pos_right_idcs[1]] = positions_right.swapaxes(1, 2)
        cur_motions_pose_space[..., glob_orient_l_idcs[0]:glob_orient_l_idcs[1]] = global_orient_l[:, np.newaxis]
        cur_motions_pose_space[..., glob_orient_r_idcs[0]:glob_orient_r_idcs[1]] = global_orient_r[:, np.newaxis]

        if args.no_subsequence_inpainting:
            motions_grasp = np.zeros((batch_size, 1, n_frames, data_interaction.dataset.motion_dataset.mean.shape[0]))
            motions_grasp[..., [111, 115]] = 1.0
            motions_grasp[..., :obj_pose_idcs[0]] = np.array(cur_motions_grasp_pose_space)[-1]
            positions_left, positions_right, global_orient_l, global_orient_r = stich_pregrasp(
                cur_motions_pose_space, positions_left, positions_right, add_obj_pos=True)
            cur_motions_pose_space[..., :pos_left_idcs[1]] = positions_left.swapaxes(1, 2)
            cur_motions_pose_space[..., pos_right_idcs[0]:pos_right_idcs[1]] = positions_right.swapaxes(1, 2)
            cur_motions_pose_space[..., glob_orient_l_idcs[0]:glob_orient_l_idcs[1]] = global_orient_l[:, np.newaxis]
            cur_motions_pose_space[..., glob_orient_r_idcs[0]:glob_orient_r_idcs[1]] = global_orient_r[:, np.newaxis]
            motions_full = np.concatenate((motions_grasp, cur_motions_pose_space), axis=2, dtype=np.float32)
        else:
            motions_full = cur_motions_pose_space

        all_motions.extend(motions_full[:, np.newaxis])
        all_lengths.extend(cur_lengths)
        all_text.extend(cur_texts)
        all_data_id.extend(model_kwargs['y']['data_id'])
        repetition_indices.extend([repetition] * batch_size)

        all_gt_kf.extend(motion_gt_kf.swapaxes(2, 3))

    all_motions = np.concatenate(all_motions, axis=0)
    all_gt_kf = np.concatenate(all_gt_kf, axis=0) if all_gt_kf else np.array([])
    all_lengths = np.concatenate(all_lengths, axis=0)

    if inference_times:
        per_row_times = np.repeat(inference_times, batch_sizes)
        avg_inference_time = np.mean(per_row_times)
        std_inference_time = np.std(per_row_times)
        total_inference_time = float(np.dot(inference_times, batch_sizes))

        print("=" * 50)
        print("INFERENCE TIME STATISTICS (Two-Stage)")
        print("=" * 50)
        print(f"Total inference time: {total_inference_time:.4f}s")
        print(f"Average time per sample: {avg_inference_time:.4f}s")
        print(f"Std time per sample: {std_inference_time:.4f}s")
        print(f"Throughput: {1.0 / avg_inference_time:.2f} samples/s")
        print("=" * 50)
    else:
        avg_inference_time = 0.0
        std_inference_time = 0.0

    save_results(
        out_path, all_motions, all_text, all_lengths, all_data_id,
        all_gt_kf, args_interaction, n_frames_post,
        inference_times=np.repeat(inference_times, batch_sizes).tolist(),
        repetition_indices=repetition_indices)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()
