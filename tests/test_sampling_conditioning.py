"""CPU regressions for checkpoint-compatible sampling and optional CFG.

These tests preserve numerical behavior; they do not measure generation quality.
"""

import unittest

import torch

from diffusion.flow_matching import FlowMatching
from sample.condition_hands import (
    ClassifierFreeGuidance,
    build_interaction_target,
    make_interaction_guidance,
)


class ProjectedDataset:
    """An invertible feature mix which moves object channels into hand channels."""

    use_rand_proj = True

    def __init__(self):
        self.mean = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
        self.std = torch.tensor([2., 4., 2., 4.], dtype=torch.float64)
        self.projection = torch.eye(4, dtype=torch.float64)[[2, 3, 0, 1]]

    def transform_th(self, pose, use_rand_proj=None):
        result = (pose - self.mean.to(pose)) / self.std.to(pose)
        if self.use_rand_proj if use_rand_proj is None else use_rand_proj:
            result = result @ self.projection.to(pose)
        return result

    def inv_transform_th(self, x, traject_only=False, use_rand_proj=None):
        if self.use_rand_proj if use_rand_proj is None else use_rand_proj:
            x = x @ self.projection.T.to(x)
        return x * self.std.to(x) + self.mean.to(x)


class SamplingConditioningTests(unittest.TestCase):
    def setUp(self):
        self.dataset = ProjectedDataset()
        grasp = torch.tensor([[[[1., 2., 3.]], [[4., 5., 6.]]]], dtype=torch.float64)
        self.object_pose = torch.tensor([[7., 8.]], dtype=torch.float64)
        self.target, self.mask = build_interaction_target(grasp, self.object_pose, 5, 4)

    def make_guidance(self):
        return make_interaction_guidance(
            self.target, self.mask, self.dataset,
            object_pose_slice=(2, 4),
            use_mse_loss=True, classifier_scale=1., stop_cond_from=.98, cut_frame=5,
        )

    def test_gradient_matches_original_surrogate_fixture(self):
        x = torch.zeros_like(self.target, requires_grad=True)
        grad = self.make_guidance()(x, torch.tensor([0.5]), {'grasp_model': False})
        expected = torch.tensor([
            [0., 4., 8., 0., 0.], [16., 24., 32., 0., 0.],
            [-12., -12., -12., 0., 0.], [-32., -32., -32., 0., 0.],
        ], dtype=torch.float64).reshape_as(x)
        torch.testing.assert_close(grad, expected, atol=0., rtol=0.)

    def test_complete_flow_sample_matches_original_sampler_fixture(self):
        # Captured from eb4402e's CondKeyLocationsFlowMatching and its original
        # hand-only surrogate target, with the same 50-step Euler sampler.
        expected = torch.tensor([
            0.3156752545277588, 0.33373893565895874, 0.34465647376665465,
            0.28572799771723245, 0.2966681354309003, 0.3774579688976532,
            0.38833930319645293, 0.3991597444066037, 0.3408840716431582,
            0.35194679251815286, 0.29285209323121614, 0.30398831150844774,
            0.3151293052897139, 0.3955873009252007, 0.406243553282836,
            0.34836177522957484, 0.35930481368842326, 0.37014727843218936,
            0.44734082771063144, 0.4571688755934084,
        ], dtype=torch.float64).reshape(1, 4, 1, 5)
        model = NonlinearPrediction()
        result = FlowMatching(num_sampling_steps=50, sigma_min=.0001).p_sample_loop(
            ClassifierFreeGuidance(model), (1, 4, 1, 5),
            noise=torch.linspace(-1, 1, 20, dtype=torch.float64).reshape(1, 4, 1, 5),
            model_kwargs={'y': {'grasp_model': False, 'scale': 1.}},
            cond_fn=self.make_guidance(),
        )
        torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(model.calls, 50)

    def test_hard_inpainting_keeps_physical_object_pose(self):
        normalized = self.dataset.transform_th(self.target.permute(0, 2, 3, 1))
        normalized = normalized.permute(0, 3, 1, 2)
        result = FlowMatching(num_sampling_steps=4, sigma_min=0.).p_sample_loop(
            NonlinearPrediction(), normalized.shape, noise=torch.zeros_like(normalized),
            model_kwargs={'y': {
                'inpainted_motion': normalized, 'inpainting_mask': self.mask,
                'do_inpainting': True, 'impute_until': 1., 'grasp_model': False,
            }}, cond_fn=self.make_guidance(),
        )
        physical_result = self.dataset.inv_transform_th(result.permute(0, 2, 3, 1))
        expected_object = self.object_pose[:, None, :].expand(-1, 3, -1)
        torch.testing.assert_close(physical_result[:, 0, :3, 2:], expected_object, atol=0., rtol=0.)
        # Building the zero-object surrogate must not mutate the inpainting goal.
        torch.testing.assert_close(self.target[:, 2:, 0, :3], expected_object.transpose(1, 2))


class NonlinearPrediction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, x, timesteps, y=None):
        self.calls += 1
        return .4 * x + .15 * torch.sin(x) + .2 * timesteps[:, None, None, None]


class TextPrediction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, x, timesteps, y):
        self.calls.append(y)
        text_value = 1. if y.get('uncond', False) else 3.
        return torch.full_like(x, text_value + y['observed_pose'].item())


class ClassifierFreeGuidanceTests(unittest.TestCase):
    def test_cfg_changes_flow_samples_and_keeps_observed_conditions(self):
        model = TextPrediction()
        observed = torch.tensor(7.)
        scene = torch.ones(1, 2, 3)
        y = {'scale': torch.tensor([1., 2.5]), 'observed_pose': observed, 'scene_points': scene}
        sampler = FlowMatching(num_sampling_steps=2, sigma_min=0.)
        result = sampler.p_sample_loop(
            ClassifierFreeGuidance(model), (2, 4, 1, 5),
            noise=torch.zeros(2, 4, 1, 5), model_kwargs={'y': y},
        )
        torch.testing.assert_close(result[0], torch.full_like(result[0], 10.))
        torch.testing.assert_close(result[1], torch.full_like(result[1], 13.))
        self.assertEqual(len(model.calls), 4)
        self.assertNotIn('uncond', y)
        for call in model.calls:
            self.assertIs(call['observed_pose'], observed)
            self.assertIs(call['scene_points'], scene)

    def test_scale_one_preserves_single_forward_behavior(self):
        model = TextPrediction()
        x = torch.zeros(1, 4, 1, 5)
        result = ClassifierFreeGuidance(model)(x, torch.tensor([0.5]), {'observed_pose': torch.tensor(7.)})
        torch.testing.assert_close(result, torch.full_like(result, 10.))
        self.assertEqual(len(model.calls), 1)


if __name__ == '__main__':
    unittest.main()
