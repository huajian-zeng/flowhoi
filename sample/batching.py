"""Pair two-stage conditions by identity and repeat complete batches."""
import copy

import torch
from torch.utils.data import DataLoader


def sequence_ids(dataset):
    motion = dataset.motion_dataset
    names = motion.name_list[motion.pointer:]
    ids = [motion.data_dict[name]['id'] for name in names]
    if len(ids) != len(dataset) or not ids:
        raise ValueError('Two-stage sampling requires a nonempty dataset with one identity per sample.')
    if any(not isinstance(identity, str) or not identity for identity in ids) or len(set(ids)) != len(ids):
        raise ValueError('Two-stage sampling requires unique nonempty data_id values.')
    return ids


def _ordered_loader(loader, indices):
    # Keep two iterators and their normal base-seed draws. A single paired
    # DataLoader would change the published first-batch model/noise RNG state.
    ordered = DataLoader(
        loader.dataset, batch_size=loader.batch_size, sampler=indices,
        num_workers=loader.num_workers, collate_fn=loader.collate_fn,
        drop_last=False, pin_memory=loader.pin_memory,
        prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers,
        worker_init_fn=loader.worker_init_fn, generator=loader.generator,
    )
    if hasattr(loader, 'fixed_length'):
        ordered.fixed_length = loader.fixed_length
    return ordered


def align_stage_loaders(grasp, interaction, random_order=False, seed=0):
    """Use the full-motion ID order and one isolated random permutation."""
    grasp_ids, interaction_ids = sequence_ids(grasp.dataset), sequence_ids(interaction.dataset)
    if set(grasp_ids) != set(interaction_ids):
        missing_grasp = sorted(set(interaction_ids) - set(grasp_ids))
        missing_full = sorted(set(grasp_ids) - set(interaction_ids))
        raise ValueError(f'Two-stage data_id mismatch: missing grasp={missing_grasp[:8]}, '
                         f'missing interaction={missing_full[:8]}')
    if grasp.batch_size != interaction.batch_size:
        raise ValueError('Two-stage loaders must use the same batch size.')
    order = list(range(len(interaction_ids)))
    if random_order:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(order), generator=generator).tolist()
    lookup = {identity: index for index, identity in enumerate(grasp_ids)}
    grasp_order = [lookup[interaction_ids[index]] for index in order]
    return _ordered_loader(grasp, grasp_order), _ordered_loader(interaction, order)


def iter_stage_batches(grasp, interaction, repetitions, entire_set=False):
    """Load each selected condition once, then generate its requested repetitions."""
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError('num_repetitions must be a positive integer.')
    grasp_iterator = iter(grasp)
    interaction_iterator = iter(interaction)
    for batch_index in range(len(interaction)):
        grasp_batch = next(grasp_iterator)
        interaction_batch = next(interaction_iterator)
        if list(grasp_batch[1]['y']['data_id']) != list(interaction_batch[1]['y']['data_id']):
            raise ValueError('Two-stage batch data_id mismatch after loading.')
        for repetition in range(repetitions):
            # Sampling mutates conditioning dictionaries; repetitions must start
            # from the same observations, lengths, captions and scene inputs.
            yield batch_index, repetition, copy.deepcopy(grasp_batch), copy.deepcopy(interaction_batch)
        if not entire_set:
            break
