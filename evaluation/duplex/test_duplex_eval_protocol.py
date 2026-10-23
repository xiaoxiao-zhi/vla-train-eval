import sys as _sys
for _p in ('/apdcephfs/private_chengzhixu/evaluation', '/apdcephfs/private_chengzhixu/JarvisVLA'):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import importlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from omegaconf import OmegaConf

ROOT = Path('/apdcephfs/private_chengzhixu/JarvisVLA')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/apdcephfs/private_chengzhixu/evaluation')
from duplex import duplex_interact_v4 as duplex
from duplex import minecraft_eval_protocol as protocol
from duplex import duplex_eval_group as group


class DuplexProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = ROOT / 'probe'
        cls.probe.mkdir(exist_ok=True)

    def test_shared_entrypoints_and_real_configs(self):
        baseline = importlib.import_module('nonduplex.run_eval_task')
        self.assertIs(baseline.build_env, duplex.build_shared_env)
        self.assertIs(baseline.prepare_episode, duplex.prepare_episode)
        self.assertIs(baseline.load_task_cfg, duplex.load_task_cfg)
        self.assertIs(baseline.build_task_instruction, duplex.build_task_instruction)
        self.assertEqual(duplex.TASKS, {
            'kill/kill_zombie': 300, 'mine/mine_stone': 500,
            'craft/craft_crafting_table': 400,
        })
        for task in duplex.TASKS:
            a, b = baseline.load_task_cfg(task), duplex.load_task_cfg(task)
            self.assertEqual(OmegaConf.to_container(a, resolve=True),
                             OmegaConf.to_container(b, resolve=True))
        craft = protocol.load_task_cfg('craft/craft_crafting_table')
        self.assertTrue(craft.need_gui)
        self.assertEqual(craft.worker, 'craft')
        self.assertEqual(craft.init_inventory[0]['type'], 'oak_planks')

    def test_instructions_are_user_fixed_for_all_tasks_and_episodes(self):
        baseline = importlib.import_module('nonduplex.run_eval_task')
        expected = {
            'craft/craft_crafting_table': 'Craft a crafting table',
            'kill/kill_zombie': 'Kill the zombie.',
            'mine/mine_stone': 'Gather stone by using a pickaxe.',
            'smelt/iron_ingot': 'Use the furnace to make an iron ingot.',
        }
        for task, text in expected.items():
            task_text = protocol.load_task_cfg(task).task_conf[0]['text']
            for episode in range(3):
                state, numpy_state = random.getstate(), np.random.get_state()
                with patch.object(random, 'choice', side_effect=AssertionError('Random instruction forbidden')):
                    self.assertEqual(duplex.build_task_instruction(task_text, episode), text)
                    self.assertEqual(baseline.build_task_instruction(task_text, episode, 123), text)
                    self.assertEqual(baseline.FIXED_TASK_AGENT.create_instruction(
                        SimpleNamespace(), task_text, 'inventory'), text)
                    self.assertEqual(baseline.FLAT3_AGENT.create_instruction(
                        SimpleNamespace(), task_text, 'inventory'), text)
                self.assertEqual(random.getstate(), state)
                np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        with self.assertRaisesRegex(ValueError, 'No fixed instruction'):
            protocol.build_task_instruction('unknown_task:item')

    def test_flat3_episode_uses_shared_instruction_and_saves_it(self):
        baseline = importlib.import_module('nonduplex.run_eval_task')
        with tempfile.TemporaryDirectory(dir=self.probe) as directory:
            cfg = protocol.load_task_cfg('craft/craft_crafting_table')
            agent = Mock(log_inputs=[], log_outputs=[])
            env = Mock()
            frame = {'pov': np.zeros((2, 2, 3), dtype=np.uint8)}
            env.step.return_value = ('obs', 0, False, False, frame)
            with patch.object(baseline, 'prepare_episode', return_value=('obs', frame, False)), \
                 patch.object(baseline, 'finish_episode_recording'), \
                 patch.object(baseline, 'FLAT3_AGENT', return_value=agent):
                baseline.run_episode(env, Mock(), cfg,
                                     {'flat3': True, 'ckpt': 'unused', 'base_url': 'unused'},
                                     directory, 1, 'recipe', 4, episode=1)
            expected = duplex.build_task_instruction(cfg.task_conf[0]['text'], 1)
            self.assertEqual(agent._flat_instruction, expected)
            saved = json.loads((Path(directory) / 'instruction.json').read_text())
            self.assertEqual(saved['instruction'], expected)
            self.assertEqual(saved['episode_seed'], 43)
            self.assertEqual(saved['selection'], 'user_fixed')

    def test_qa_rng_isolated_from_environment(self):
        pool = duplex.load_qa_pool(duplex.QA_POOL_JSON)
        self.assertEqual(len(pool), 200)
        for episode in range(3):
            protocol.seed_host(42 + episode)
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            qa, frame = duplex.choose_qa(pool, 42, episode)
            self.assertEqual((qa, frame), duplex.choose_qa(pool, 42, episode))
            self.assertEqual(random.getstate(), python_state)
            after = np.random.get_state()
            self.assertEqual(after[0], numpy_state[0])
            np.testing.assert_array_equal(after[1], numpy_state[1])
            self.assertEqual(after[2:], numpy_state[2:])
            self.assertTrue(20 <= frame <= 40)

    def test_fast_reset_commands_reproducible(self):
        from minestudio.simulator.callbacks import FastResetCallback
        cfg = protocol.load_task_cfg('mine/mine_stone')
        for episode in range(3):
            observed = []
            for mode in ('baseline', 'duplex'):
                protocol.seed_host(42 + episode)
                if mode == 'duplex':
                    duplex.choose_qa([{'q': 'test', 'a': 'test'}], 42, episode)
                sim = SimpleNamespace(already_reset=True, env=Mock())
                sim.env.execute_cmd.return_value = (None, 0, False, {})
                callback = FastResetCallback(cfg.candidate_preferred_spawn_biome,
                                             cfg.random_tp_range, cfg.start_time)
                self.assertFalse(callback.before_reset(sim, True))
                observed.append([call.args[0] for call in sim.env.execute_cmd.call_args_list])
            self.assertEqual(observed[0], observed[1])

    def test_prepare_episode_seed_and_craft_gui(self):
        cfg = protocol.load_task_cfg('craft/craft_crafting_table')
        with tempfile.TemporaryDirectory(dir=self.probe) as directory:
            for episode in range(3):
                env = Mock()
                env.reset.return_value = ('initial_obs', {'pov': np.zeros((2, 2, 3), dtype=np.uint8)})
                record = SimpleNamespace(record_path=None)
                worker = Mock()
                worker.obs = 'gui_obs'
                worker.info = {'pov': np.full((2, 2, 3), episode, dtype=np.uint8), 'isGuiOpen': False}
                with patch('jarvisvla.evaluate.env_helper.craft_agent.CraftWorker', return_value=worker) as craft:
                    folder = Path(directory) / ('ep%d' % episode)
                    obs, info, need_table = protocol.prepare_episode(env, record, cfg, folder, episode)
                craft.assert_called_once_with(env, if_discrete=True)
                worker._null_action.assert_called_once_with(1)
                worker._call_func.assert_called_once_with('inventory')
                self.assertEqual(obs, 'gui_obs')
                self.assertFalse(need_table)
                self.assertEqual(env.action_type, 'agent')
                meta = json.loads((folder / 'environment.json').read_text())
                self.assertEqual(meta['python_seed'], 42 + episode)
                self.assertEqual(meta['numpy_seed'], 42 + episode)
                self.assertEqual(meta['world_seed'], cfg.seed)
                np.testing.assert_array_equal(env.np_random.integers(100, size=3),
                                              np.random.default_rng(42 + episode).integers(100, size=3))

    def test_observation_text_and_image_count(self):
        image = Image.new('RGB', (2, 2))
        session = Mock()
        session.post.return_value.json.return_value = {'choices': [{'message': {'content': 'test'}}]}
        prompt = '<|obs_user|>task\nobservation: \n<image><|obs_asst|><|silence|><|obs_user|>question\nobservation: \n<image>'
        _, request = duplex.vllm_chat('http://127.0.0.1:8100/v1', 'model', prompt, [image, image], session)
        rebuilt = ''.join(item['text'] if item['type'] == 'text' else '<image>'
                          for item in request['messages'][0]['content'])
        self.assertEqual(rebuilt, prompt)
        self.assertEqual(request['top_k'], -1)
        with self.assertRaises(ValueError):
            duplex.vllm_chat('http://127.0.0.1:8100/v1', 'model', prompt, [image], session)

    def test_default_plan_has_nine_episodes(self):
        args = duplex.parse_args([])
        self.assertEqual(args.task, 'all')
        self.assertEqual(args.repeats, 3)
        with tempfile.TemporaryDirectory(dir=self.probe) as directory:
            tokenizer = Mock()
            tokenizer.return_value = {'input_ids': [204]}
            record = Mock()
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                 patch('jarvisvla.inference.action_mapping.OneActionTokenizer'), \
                 patch('minestudio.simulator.callbacks.RecordCallback', new=Mock), \
                 patch.object(duplex, 'build_shared_env', side_effect=lambda *a: (Mock(), [record])) as build, \
                 patch.object(duplex, 'run_episode', return_value=(True, 10)) as run:
                for task in duplex.TASKS:
                    duplex.main(['--task', task, '--out-dir', directory, '--tokenizer', 'unused'])
            self.assertEqual(build.call_count, 3)
            self.assertEqual(run.call_count, 9)
            calls = [(call.args[4], call.args[5], call.args[7]) for call in run.call_args_list]
            self.assertEqual(calls, [(task, ep, budget) for task, budget in duplex.TASKS.items() for ep in range(3)])
            self.assertFalse((Path(directory) / 'summary.json').exists())
            for task in duplex.TASKS:
                summary = json.loads((Path(directory) / task.replace('/', '_') / 'summary.json').read_text())
                self.assertEqual(summary['total'], 3)
                self.assertEqual([ep['seed'] for ep in summary['tasks'][task]['episodes']], [42, 43, 44])

    def test_post_success_twenty_steps_and_per_frame_text(self):
        with tempfile.TemporaryDirectory(dir=self.probe) as directory:
            args = duplex.parse_args(['--task', 'kill/kill_zombie'])
            env = Mock()
            frame = {'pov': np.zeros((2, 2, 3), dtype=np.uint8)}
            env.step.return_value = ('obs', 1, False, False, frame)
            tokenizer = Mock()
            tokenizer.return_value = {'input_ids': [178, 219, 240, 179]}
            action_tokenizer = Mock()
            action_tokenizer.decode.return_value = [{'buttons': [0], 'camera': [0, 0]}]
            action = '<|reserved_special_token_178|><|reserved_special_token_219|><|reserved_special_token_240|><|reserved_special_token_179|>'
            with patch.object(duplex, 'prepare_episode', return_value=('obs', frame, False)), \
                 patch.object(duplex, 'finish_episode_recording'), \
                 patch.object(duplex, 'vllm_chat', return_value=(action + '<|silence|>', {})) as chat_request:
                cfg = protocol.load_task_cfg('kill/kill_zombie')
                result = duplex.run_episode(args, env, Mock(), cfg,
                                            'kill/kill_zombie', 0, directory, 300, tokenizer,
                                            action_tokenizer, Mock(), Mock(), [{'q': 'q', 'a': 'a', 'source_id': 'id'}])
            expected = protocol.build_task_instruction(cfg.task_conf[0]['text'])
            self.assertEqual(chat_request.call_args_list[0].args[2],
                             duplex.OBS_USER + expected + '\nobservation: \n<image>')
            self.assertIn(duplex.OBS_ASST + '<|silence|>' + duplex.OBS_USER + '<|silence|>',
                          chat_request.call_args_list[1].args[2])
            saved = json.loads((Path(directory) / 'instruction.json').read_text())
            self.assertEqual(saved['instruction'], expected)
            self.assertEqual(result, (True, 0))
            self.assertEqual(env.step.call_count, 21)
            payload = json.loads(next(Path(directory).glob('outputs_*.json')).read_text())
            self.assertEqual(len(payload['outputs']), 21)
            self.assertTrue(all(item['text_raw'] == '<|silence|>' and item['is_silence'] for item in payload['outputs']))
            chat = json.loads(next(Path(directory).glob('chat_*.json')).read_text())
            self.assertEqual(len([row for row in chat['chat'] if row['role'] == 'assistant']), 21)


class DuplexTaskSchedulingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = ROOT / 'probe'
        cls.probe.mkdir(exist_ok=True)

    def write_task_results(self, args, task):
        directory = Path(args.out_dir) / task.replace('/', '_')
        episodes = []
        budget = args.max_frames if args.max_frames is not None else duplex.TASKS[task]
        for episode in range(args.repeats):
            folder = directory / ('ep%d' % episode)
            folder.mkdir(parents=True)
            result = {'ok': True, 'steps': 10, 'episode': episode,
                      'max_frames': budget, 'seed': args.seed + episode}
            (folder / 'result.json').write_text(json.dumps(result))
            (folder / 'environment.json').write_text(json.dumps({
                'python_seed': args.seed + episode, 'numpy_seed': args.seed + episode}))
            episodes.append({'episode': episode, 'ok': True, 'steps': 10, 'seed': args.seed + episode})
        summary = {'model': args.model_name, 'base_seed': args.seed, 'hist_mode': args.hist_mode,
                   'repeats': args.repeats, 'qa_pool': args.qa_pool,
                   'protocol': 'shared_minecraft_eval_v1', 'total': args.repeats,
                   'successes': args.repeats, 'tasks': {task: {
                       'successes': args.repeats, 'total': args.repeats,
                       'max_frames': budget, 'episodes': episodes}}}
        (directory / 'summary.json').write_text(json.dumps(summary))

    def test_all_dispatches_without_loading_model_or_environment(self):
        with patch.object(group, 'run_tasks', return_value='scheduled') as run, \
             patch.object(duplex, 'build_shared_env') as build:
            self.assertEqual(duplex.main([]), 'scheduled')
        build.assert_not_called()
        args, tasks = run.call_args.args
        self.assertEqual(args.workers, 3)
        self.assertEqual(args.stagger, 10)
        self.assertEqual(tasks, duplex.TASKS)

    def test_three_tasks_start_before_wait_and_arguments_match(self):
        with tempfile.TemporaryDirectory(dir=self.probe) as directory:
            urls = ['http://127.0.0.1:%d/v1' % port for port in (19100, 19200, 19300)]
            args = duplex.parse_args(['--out-dir', directory, '--stagger', '0',
                                      '--base-urls', ','.join(urls)])
            started = []
            def start(command, **kwargs):
                task = command[command.index('--task') + 1]
                self.write_task_results(args, task)
                started.append((command, kwargs))
                process = Mock(pid=1000 + len(started), args=command)
                def poll():
                    self.assertEqual(len(started), 3)
                    return 0
                process.poll.side_effect = poll
                return process
            with patch.object(group.subprocess, 'Popen', side_effect=start), \
                 patch.object(group.time, 'sleep'):
                summary = group.run_tasks(args, duplex.TASKS)
            self.assertEqual(summary['total'], 9)
            for command, kwargs in started:
                self.assertNotEqual(command[command.index('--task') + 1], 'all')
                self.assertEqual(command[command.index('--repeats') + 1], '3')
                self.assertEqual(command[command.index('--seed') + 1], '42')
                self.assertEqual(command[command.index('--qa-pool') + 1], args.qa_pool)
                self.assertTrue(kwargs['start_new_session'])
            assigned = [command[command.index('--base-url') + 1] for command, _ in started]
            self.assertEqual(assigned, urls)
            self.assertTrue(all(command[command.index('--base-urls') + 1] == '' for command, _ in started))
            routes = json.loads((Path(directory) / '_logs/task_servers.json').read_text())
            self.assertEqual(list(routes.values()), urls)
            self.assertTrue((Path(directory) / 'summary.json').exists())

    def test_task_starts_staggered_with_overlapping_lifetimes(self):
        with tempfile.TemporaryDirectory(dir=self.probe) as directory:
            args = duplex.parse_args(['--out-dir', directory])
            clock, starts = [0.0], []
            def start(command, **kwargs):
                task = command[command.index('--task') + 1]
                self.write_task_results(args, task)
                starts.append(clock[0])
                process = Mock(pid=2000 + len(starts), args=command)
                process.poll.side_effect = lambda: 0 if clock[0] >= 125 else None
                return process
            def sleep(seconds):
                clock[0] += seconds
            with patch.object(group.subprocess, 'Popen', side_effect=start), \
                 patch.object(group.time, 'monotonic', side_effect=lambda: clock[0]), \
                 patch.object(group.time, 'sleep', side_effect=sleep):
                group.run_tasks(args, duplex.TASKS)
            self.assertEqual(starts, [0, 10, 20])

    def test_child_failure_does_not_write_success_summary(self):
        with tempfile.TemporaryDirectory(dir=self.probe) as directory:
            args = duplex.parse_args(['--out-dir', directory, '--stagger', '0'])
            children = []
            def start(command, **kwargs):
                process = Mock(pid=3000 + len(children), args=command)
                process.poll.return_value = 1 if not children else None
                children.append(process)
                return process
            def cleanup(running):
                self.assertEqual(len(running), 2)
                for _, _, log in running:
                    log.close()
            with patch.object(group.subprocess, 'Popen', side_effect=start), \
                 patch.object(group, 'stop_owned_processes', side_effect=cleanup), \
                 self.assertRaises(group.subprocess.CalledProcessError):
                group.run_tasks(args, duplex.TASKS)
            self.assertFalse((Path(directory) / 'summary.json').exists())

    def test_summary_rejects_wrong_episode_seed(self):
        with tempfile.TemporaryDirectory(dir=self.probe) as directory:
            args = duplex.parse_args(['--out-dir', directory])
            for task in duplex.TASKS:
                self.write_task_results(args, task)
            path = Path(directory) / 'mine_mine_stone/ep1/result.json'
            result = json.loads(path.read_text())
            result['seed'] = 42
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'seed differs'):
                group.collect_results(args, duplex.TASKS)
            self.assertFalse((Path(directory) / 'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
