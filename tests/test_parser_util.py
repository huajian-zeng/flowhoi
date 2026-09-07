import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.parser_util import GenerateArgs, generate_args


class GenerateArgsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model_path = Path(self.tmp.name) / 'model.pt'
        self.checkpoint_options = asdict(GenerateArgs())
        self.write_checkpoint_options()

    def write_checkpoint_options(self, **updates):
        self.checkpoint_options.update(updates)
        (self.model_path.parent / 'args.json').write_text(
            json.dumps(self.checkpoint_options))

    def parse(self, *options, model_path=None):
        argv = ['sample.sample_2stage', '--model_path', str(self.model_path), *options]
        with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return generate_args(model_path=model_path)

    def test_checkpoint_configuration_is_loaded_without_overrides(self):
        self.write_checkpoint_options(split_set='random', use_scene_global=True,
                                      use_scene_perceiver=False, num_sampling_steps=80)
        args = self.parse()
        self.assertEqual(args.split_set, 'random')
        self.assertTrue(args.use_scene_global)
        self.assertFalse(args.use_scene_perceiver)
        self.assertEqual(args.num_sampling_steps, 80)
        self.assertTrue(args.cuda)

    def test_default_cfg_preserves_conditional_sampling(self):
        self.write_checkpoint_options(guidance_param=2.5)
        self.assertEqual(self.parse().guidance_param, 1.0)
        self.assertEqual(self.parse('--guidance').guidance_param, 1.0)

    def test_cfg_scale_requires_an_explicit_sampling_override(self):
        self.write_checkpoint_options(guidance_param=3.0)
        for options in [('--guidance_param', '2.5'), ('--guidance_param=2.5',)]:
            with self.subTest(options=options):
                self.assertEqual(self.parse(*options).guidance_param, 2.5)

    def test_space_and_equals_cli_options_override_checkpoint(self):
        self.write_checkpoint_options(split_set='objects_unseen',
                                      scene_fps_path='/old/bundle',
                                      num_sampling_steps=80, data_dir='/old/data')
        for options in [
            ['--split_set', 'demo', '--scene_fps_path', './scene',
             '--num_sampling_steps', '7', '--data_dir', ''],
            ['--split_set=demo', '--scene_fps_path=./scene',
             '--num_sampling_steps=7', '--data_dir='],
        ]:
            with self.subTest(options=options):
                args = self.parse(*options)
                self.assertEqual(args.split_set, 'demo')
                self.assertEqual(args.scene_fps_path, './scene')
                self.assertEqual(args.num_sampling_steps, 7)
                self.assertEqual(args.data_dir, '')

    def test_boolean_false_overrides_checkpoint_for_both_default_values(self):
        self.write_checkpoint_options(use_scene_global=True, use_scene_perceiver=True)
        for options in [
            ['--no_use_scene_global', '--no_use_scene_perceiver'],
            ['--use_scene_global=false', '--use_scene_perceiver=false'],
            ['--use_scene_global', 'false', '--use_scene_perceiver', 'false'],
        ]:
            with self.subTest(options=options):
                args = self.parse(*options)
                self.assertFalse(args.use_scene_global)
                self.assertFalse(args.use_scene_perceiver)

    def test_boolean_true_overrides_checkpoint_for_both_default_values(self):
        self.write_checkpoint_options(use_scene_global=False, use_scene_perceiver=False)
        args = self.parse('--use_scene_global', '--use_scene_perceiver')
        self.assertTrue(args.use_scene_global)
        self.assertTrue(args.use_scene_perceiver)

    def test_last_boolean_override_wins(self):
        self.write_checkpoint_options(use_scene_global=True)
        args = self.parse('--use_scene_global', '--no_use_scene_global')
        self.assertFalse(args.use_scene_global)
        args = self.parse('--no_use_scene_global', '--use_scene_global')
        self.assertTrue(args.use_scene_global)

    def test_function_checkpoint_path_still_keeps_cli_overrides(self):
        grasp_dir = self.model_path.parent / 'grasp'
        grasp_dir.mkdir()
        (grasp_dir / 'args.json').write_text(json.dumps({
            **self.checkpoint_options, 'dataset': 'egodex', 'split_set': 'random'}))
        model_path = str(grasp_dir / 'grasp.pt')
        args = self.parse('--split_set=demo', model_path=model_path)
        self.assertEqual(args.model_path, model_path)
        self.assertEqual(args.dataset, 'egodex')
        self.assertEqual(args.split_set, 'demo')

    def test_frame_lengths_are_integers_at_valid_boundaries(self):
        for frame_count in [50, 200]:
            with self.subTest(frame_count=frame_count):
                args = self.parse('--max_frames_grasp=50',
                                  f'--max_frames_interaction={frame_count}')
                self.assertIs(type(args.max_frames_grasp), int)
                self.assertIs(type(args.max_frames_interaction), int)
                self.assertEqual(args.max_frames_interaction, frame_count)

    def test_unsupported_frame_lengths_fail_before_loading_models(self):
        for option in ['--max_frames_grasp=49', '--max_frames_grasp=51',
                       '--max_frames_grasp=50.0', '--max_frames_interaction=49',
                       '--max_frames_interaction=201', '--max_frames_interaction=50.5',
                       '--grasp_frame=48']:
            with self.subTest(option=option), self.assertRaises(SystemExit) as error:
                self.parse(option)
            self.assertEqual(error.exception.code, 2)

    def test_invalid_boolean_value_is_rejected(self):
        with self.assertRaises(SystemExit) as error:
            self.parse('--use_scene_global=perhaps')
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
