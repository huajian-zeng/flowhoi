# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch

def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]

    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    """Collate motions, using the first 108 channels at the final encoded frame as the grasp pose."""
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)

    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1)

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'object_bps' in notnone_batches[0]:
        object_bps_batch = [b['object_bps'] for b in notnone_batches]
        cond['y'].update({'object_bps': object_bps_batch})

    if 'motion_enc_frames' in notnone_batches[0] and notnone_batches[0]['motion_enc_frames']>0:
        databatch_full = [b['motion_full'] for b in notnone_batches]
        motion_full = collate_tensors(databatch_full)
        motion_enc = motion_full[...,:notnone_batches[0]['motion_enc_frames']]

        cond['y'].update({'motion_enc_gt': motion_enc})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    if 'data_id' in notnone_batches[0]:
        data_ids = [b['data_id'] for b in notnone_batches]
        cond['y'].update({'data_id': data_ids})

    if 't5_emb' in notnone_batches[0] and notnone_batches[0]['t5_emb'] is not None:
        t5_embs = [b['t5_emb'] for b in notnone_batches]
        if all(emb is not None for emb in t5_embs):
            t5_embs_tensor = torch.stack(t5_embs, dim=0)
            cond['y'].update({'t5_emb': t5_embs_tensor})
            if 't5_mask' in notnone_batches[0] and notnone_batches[0]['t5_mask'] is not None:
                t5_masks = [b['t5_mask'] for b in notnone_batches]
                if all(m is not None for m in t5_masks):
                    t5_masks_tensor = torch.stack(t5_masks, dim=0)
                    cond['y'].update({'t5_mask': t5_masks_tensor})

    if 'motion_full' in notnone_batches[0]:
        motion_full_batch = [b['motion_full'] for b in notnone_batches]
        motion_full_tensor = collate_tensors(motion_full_batch)
        initial_pos = motion_full_tensor[:, :, :, 0]
        initial_pos = initial_pos.reshape(initial_pos.shape[0], -1)
        cond['y'].update({'initial_pos': initial_pos})

        if 'motion_enc_frames' in notnone_batches[0] and notnone_batches[0]['motion_enc_frames'] > 0:
            grasp_frame_idx = notnone_batches[0]['motion_enc_frames'] - 1
            grasp_pose = motion_full_tensor[:, :108, :, grasp_frame_idx]
            grasp_pose = grasp_pose.reshape(grasp_pose.shape[0], -1)
            cond['y'].update({'grasp_pose': grasp_pose})

    if 'scene_points' in notnone_batches[0] and notnone_batches[0]['scene_points'] is not None:
        scene_points_list = [b['scene_points'] for b in notnone_batches]
        if all(sp is not None for sp in scene_points_list):
            scene_points = torch.stack(scene_points_list, dim=0)
            scene_mask = (scene_points[..., :3].abs().sum(dim=-1) > 1e-6)
            cond['y'].update({
                'scene_points': scene_points,
                'scene_mask': scene_mask,
            })

    if 'occ_map' in notnone_batches[0] and notnone_batches[0]['occ_map'] is not None:
        occ_map_list = [b['occ_map'] for b in notnone_batches]
        if all(om is not None for om in occ_map_list):
            occ_map = torch.stack(occ_map_list, dim=0)
            cond['y'].update({'occ_map': occ_map})

    if 'concerto_features' in notnone_batches[0] and notnone_batches[0]['concerto_features'] is not None:
        concerto_features_list = [b['concerto_features'] for b in notnone_batches]
        if all(cf is not None for cf in concerto_features_list):
            concerto_features = torch.stack(concerto_features_list, dim=0)
            cond['y'].update({'concerto_features': concerto_features})

            if 'concerto_coords' in notnone_batches[0] and notnone_batches[0]['concerto_coords'] is not None:
                concerto_coords_list = [b['concerto_coords'] for b in notnone_batches]
                if all(cc is not None for cc in concerto_coords_list):
                    concerto_coords = torch.stack(concerto_coords_list, dim=0)
                    cond['y'].update({'concerto_coords': concerto_coords})

            if 'concerto_mask' in notnone_batches[0] and notnone_batches[0]['concerto_mask'] is not None:
                concerto_mask_list = [b['concerto_mask'] for b in notnone_batches]
                if all(cm is not None for cm in concerto_mask_list):
                    concerto_mask = torch.stack(concerto_mask_list, dim=0)
                    cond['y'].update({'concerto_mask': concerto_mask})

    if 'scene_idx' in notnone_batches[0] and notnone_batches[0]['scene_idx'] is not None:
        scene_idx_list = [b['scene_idx'] for b in notnone_batches]
        if all(si is not None for si in scene_idx_list):
            cond['y'].update({'scene_idx': torch.tensor(scene_idx_list, dtype=torch.long)})
            anchor_list = [b['scene_anchor'] for b in notnone_batches]
            if all(a is not None for a in anchor_list):
                cond['y'].update({'scene_anchor': torch.stack(anchor_list, dim=0)})

    if 'semantic_features' in notnone_batches[0] and notnone_batches[0]['semantic_features'] is not None:
        semantic_features_list = [b['semantic_features'] for b in notnone_batches]
        if all(sf is not None for sf in semantic_features_list):
            semantic_features = torch.stack(semantic_features_list, dim=0)
            cond['y'].update({'semantic_features': semantic_features})

    return motion, cond

def motion_collate(batch):
    """Adapt 19-field HOIDataset tuples to the generic motion collator."""
    def to_tensor(x):
        """Convert to tensor, handling both numpy arrays and existing tensors."""
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.clone().detach().float()
        return torch.tensor(x).float()

    adapted_batch = [{
        'text': b[0],
        'inp': torch.tensor(b[2].T).float().unsqueeze(1),
        'lengths': b[3],
        'tokens': b[4],
        'object_bps': to_tensor(b[5]),
        'motion_enc_frames': b[6],
        'motion_full': torch.tensor(b[7].T).float().unsqueeze(1),
        'data_id': b[8],
        'scene_points': b[9] if len(b) > 9 else None,
        'occ_map': b[10] if len(b) > 10 else None,
        't5_emb': b[11] if len(b) > 11 else None,
        't5_mask': b[12] if len(b) > 12 else None,
        'concerto_features': to_tensor(b[13]) if len(b) > 13 and b[13] is not None else None,
        'concerto_coords': to_tensor(b[14]) if len(b) > 14 and b[14] is not None else None,
        'concerto_mask': torch.tensor(b[15]).bool() if len(b) > 15 and b[15] is not None else None,
        'semantic_features': to_tensor(b[16]) if len(b) > 16 and b[16] is not None else None,
        'scene_idx': int(b[17]) if len(b) > 17 and b[17] is not None else None,
        'scene_anchor': to_tensor(b[18]) if len(b) > 18 and b[18] is not None else None,
    } for b in batch]

    return collate(adapted_batch)
