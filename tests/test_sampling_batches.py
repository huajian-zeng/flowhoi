"""Real prepared motion readers and a bounded two-stage main-loop harness."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_loaders.humanml.data.dataset import MotionDataset
from data_loaders.tensors import motion_collate
from sample.batching import align_stage_loaders, iter_stage_batches
from utils.parser_util import GenerateArgs


class PreparedMotion(Dataset):
    """Expose the production reader with small on-disk prepared representations."""
    def __init__(self, root, ids, grasp, projected):
        phase = 'grasp' if grasp else 'full'
        directory = root / phase
        directory.mkdir(parents=True, exist_ok=True)
        split = root / f'{phase}_split.txt'
        split.write_text(''.join(f'{identity}\n' for identity in ids))
        frames = 50 if grasp else 70
        for identity in ids:
            motion = np.zeros((frames, 117), np.float32)
            motion[:, 0] = int(identity) + 1
            motion[:, 3] = int(identity) + 2
            for start in (30, 60, 111):
                motion[:, start:start + 6] = [1, 0, 0, 0, 1, 0]
            np.save(directory / f'{identity}.npy', motion)
            (directory / f'{identity}.txt').write_text('move mug#move/OTHER mug/OTHER#0.0#0.0\n')
        names = root / 'file_names.txt'
        names.write_text(''.join(f'{identity},s6_mug_pass_{int(identity) + 1}\n' for identity in sorted(set(ids))))
        for name in ('bps_enc.npy', 'bps_enc_mirrored.npy'):
            np.save(root / name, {'mug': np.ones((1, 1024, 3), np.float32)})
        projection = np.diag(np.linspace(2, 4, 117, dtype=np.float32))
        np.save(root / 'projection.npy', projection)
        np.save(root / 'inv_projection.npy', np.linalg.inv(projection))
        opt = SimpleNamespace(data_root=str(root), motion_dir=str(directory), text_dir=str(directory),
                              max_motion_length=frames, max_text_len=20)
        self.motion_dataset = MotionDataset(
            opt, np.zeros(117, np.float32), np.ones(117, np.float32), str(split),
            mode='text_only', hands_only=grasp, traject_only=grasp,
            use_rand_proj=projected, proj_matrix_dir=str(root), proj_matrix_name='projection.npy',
            file_names_path=str(names), object_list_new=['mug'], object_new2original_dict={'mug': 'mug'})

    def __len__(self):
        return len(self.motion_dataset)

    def __getitem__(self, index):
        return self.motion_dataset[index]


class DummyModel(torch.nn.Module):
    def __init__(self, joints):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.randn(()))
        self.njoints, self.nfeats = joints, 1


class SamplingBatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='flowhoi-sampling-batches-')
        self.root = Path(self.temporary.name)
        self.quiet = contextlib.ExitStack()
        self.quiet.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.quiet.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def tearDown(self):
        self.quiet.close()
        self.temporary.cleanup()

    def loaders(self, size=3, batch_size=2, reversed_grasp=False, projected=True):
        ids = [str(index) for index in range(size)]
        grasp = PreparedMotion(self.root / 'grasp-data', ids[::-1] if reversed_grasp else ids, True, projected)
        full = PreparedMotion(self.root / 'full-data', ids, False, projected)
        return tuple(DataLoader(dataset, batch_size=batch_size, collate_fn=motion_collate,
                                num_workers=0, drop_last=True) for dataset in (grasp, full))

    def test_random_order_is_shared_by_id_and_does_not_draw_global_rng(self):
        grasp, full = self.loaders(size=7, reversed_grasp=True)
        torch.manual_seed(12)
        before = torch.get_rng_state().clone()
        ordered = align_stage_loaders(grasp, full, random_order=True, seed=1)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        batches = list(iter_stage_batches(*ordered, repetitions=1, entire_set=True))
        seen = []
        for _, _, first, second in batches:
            self.assertEqual(first[1]['y']['data_id'], second[1]['y']['data_id'])
            seen.extend(second[1]['y']['data_id'])
        self.assertCountEqual(seen, [f's6_mug_pass_{i + 1}' for i in range(7)])
        self.assertNotEqual(seen, [f's6_mug_pass_{i + 1}' for i in range(7)])
        self.assertEqual(len(batches[-1][3][1]['y']['data_id']), 1)

    def test_default_first_batch_preserves_two_loader_rng_draws_and_values(self):
        grasp, full = self.loaders(size=2)
        torch.manual_seed(42)
        before = iter(grasp), iter(full)
        expected = next(before[0]), next(before[1])
        expected_noise = torch.randn(16)
        torch.manual_seed(42)
        after = align_stage_loaders(grasp, full)
        _, _, actual_grasp, actual_full = next(iter_stage_batches(*after, repetitions=1))
        torch.testing.assert_close(actual_grasp[0], expected[0][0], rtol=0, atol=0)
        torch.testing.assert_close(actual_full[0], expected[1][0], rtol=0, atol=0)
        torch.testing.assert_close(torch.randn(16), expected_noise, rtol=0, atol=0)

    def test_mismatched_or_duplicate_ids_fail_before_sampling(self):
        grasp, full = self.loaders()
        record = grasp.dataset.motion_dataset.data_dict['0']
        record['id'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'data_id mismatch'):
            align_stage_loaders(grasp, full)
        record['id'] = grasp.dataset.motion_dataset.data_dict['1']['id']
        with self.assertRaisesRegex(ValueError, 'unique'):
            align_stage_loaders(grasp, full)

    def test_repetitions_reuse_observations_without_mutation(self):
        ordered = align_stage_loaders(*self.loaders())
        iterator = iter_stage_batches(*ordered, repetitions=2)
        _, _, first, _ = next(iterator)
        expected = first[0].clone()
        first[0].fill_(999)
        first[1]['y']['text'][0] = 'changed'
        _, repetition, second, _ = next(iterator)
        self.assertEqual(repetition, 1)
        torch.testing.assert_close(second[0], expected)
        self.assertEqual(second[1]['y']['text'][0], 'move mug')
        with self.assertRaises(StopIteration):
            next(iterator)

    def run_main(self, size=3, batch_size=2, repetitions=2, entire_set=True,
                 random_order=False, projected=True):
        import sample.sample_2stage as sampling
        grasp_loader, full_loader = self.loaders(size, batch_size, reversed_grasp=True, projected=projected)
        interaction = GenerateArgs()
        for key, value in dict(dataset='grab', num_samples=batch_size, batch_size=64,
                               num_repetitions=repetitions, eval_entire_set=entire_set,
                               random_order=random_order, seed=1, guidance=False, inpainting=True,
                               pre_grasp=False, traj_only=False, use_flow_matching=True,
                               use_pose_cond=True, no_subsequence_inpainting=False,
                               use_random_proj=projected, max_frames_grasp=50, max_frames_interaction=70,
                               model_path='unused/full.pt', grasp_model_path='unused/grasp.pt',
                               output_dir=str(self.root / 'result'), split_set='fixture').items():
            setattr(interaction, key, value)
        grasp = copy.deepcopy(interaction)
        grasp.pre_grasp = grasp.traj_only = grasp.hands_only = True
        calls, expected_physical = [], []

        def create(args, data):
            return DummyModel(108 if args.pre_grasp else 117), SimpleNamespace()

        def sample_flow(model, diffusion, shape, model_kwargs, cond_fn=None):
            y = model_kwargs['y']
            self.assertEqual(shape[0], len(y['data_id']))
            self.assertEqual(y['inpainted_motion'].shape, shape)
            calls.append((shape[1], list(y['data_id'])))
            if shape[1] == 108:
                return [y['inpainted_motion'].clone()]
            # Execute all production conditioning and conversion, replacing only
            # the expensive network/sampler with a known physical prediction.
            physical = torch.zeros(shape[0], 1, shape[-1], 117)
            for start in (30, 60, 111):
                physical[..., start:start + 6] = torch.tensor([1, 0, 0, 0, 1, 0])
            physical[..., 0] = torch.rand(shape[0], 1, 1)
            expected_physical.extend(physical[:, 0, 0, 0].tolist())
            encoded = full_loader.dataset.motion_dataset.transform_th(physical)
            return [encoded.permute(0, 3, 1, 2)]

        with patch.object(sampling, 'generate_args', side_effect=[interaction, grasp]), \
             patch.object(sampling, 'get_template', side_effect=lambda args, **kwargs: args), \
             patch.object(sampling, 'load_stage_datasets', return_value=(grasp_loader, full_loader)), \
             patch.object(sampling, 'create_model_and_diffusion', side_effect=create), \
             patch.object(sampling, 'load_saved_model', side_effect=lambda model, path: model), \
             patch.object(sampling, 'sample_flow_matching', side_effect=sample_flow), \
             patch.object(sampling.dist_util, 'setup_dist'), \
             patch.object(sampling.dist_util, 'dev', return_value=torch.device('cpu')):
            sampling.main()
        path = Path(interaction.output_dir) / f'results_{interaction.gen_split}_fixture.npy'
        result = np.load(path, allow_pickle=True).item()
        for stage_one, stage_two in zip(calls[::2], calls[1::2]):
            self.assertEqual(stage_one[1], stage_two[1])
        np.testing.assert_allclose(result['motion'][:, 0, 0, 0], expected_physical, atol=1e-6)
        self.assertEqual(len(result['text']), len(result['motion']))
        self.assertEqual(len(result['lengths']), len(result['motion']))
        self.assertEqual(len(result['gt_kf']), len(result['motion']))
        self.assertEqual(len(result['inference_times']), len(result['motion']))
        return result

    def test_main_repeats_one_selected_condition_instead_of_advancing_data(self):
        result = self.run_main(batch_size=1, repetitions=3, entire_set=False)
        self.assertEqual(result['data_id'], ['s6_mug_pass_1'] * 3)
        self.assertEqual(result['repetition_index'], [0, 1, 2])
        self.assertEqual(result['num_samples'], 1)
        self.assertEqual(result['num_generated'], 3)

    def test_main_entire_set_keeps_tail_and_repeats_every_condition(self):
        result = self.run_main()
        self.assertEqual(result['data_id'], ['s6_mug_pass_1', 's6_mug_pass_2'] * 2 + ['s6_mug_pass_3'] * 2)
        self.assertEqual(result['repetition_index'], [0, 0, 1, 1, 0, 1])
        self.assertEqual(result['num_samples'], 3)
        self.assertEqual(result['num_generated'], 6)

    def test_main_random_subset_smaller_than_batch_and_unprojected_inverse(self):
        result = self.run_main(size=2, batch_size=3, random_order=True, projected=False)
        self.assertEqual(result['num_samples'], 2)
        self.assertEqual(result['num_generated'], 4)
        self.assertEqual(result['requested_num_samples'], 3)
        self.assertCountEqual(result['data_id'], ['s6_mug_pass_1', 's6_mug_pass_2'] * 2)
        self.assertEqual(result['num_repetitions'], 2)


if __name__ == '__main__':
    unittest.main()
