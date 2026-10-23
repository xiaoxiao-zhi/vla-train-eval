import sys as _sys
for _p in ('/apdcephfs/private_chengzhixu/evaluation', '/apdcephfs/private_chengzhixu/JarvisVLA'):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/apdcephfs/private_chengzhixu/evaluation')
from duplex import vllm_eval_pool as launcher
from duplex.eval_server_urls import URLSlots, parse_base_urls


class ServerInterfacesTests(unittest.TestCase):
    def test_url_slots_release_only_completed_task(self):
        slots = URLSlots(['http://a:1111/v1', 'http://b:2222/v1'], 2)
        first = slots.acquire('a')
        second = slots.acquire('b')
        with self.assertRaises(RuntimeError):
            slots.acquire('c')
        slots.release('b')
        self.assertEqual(slots.acquire('c'), second)
        self.assertEqual(slots.owners['a'], first)

    def test_invalid_or_duplicate_urls_rejected(self):
        self.assertEqual(parse_base_urls(' http://host:123/v1/ ,https://host:456/v1'),
                         ['http://host:123/v1', 'https://host:456/v1'])
        for value in ('http://host:123/v1,http://host:123/v1/', 'not-a-url', 'http://host:123'):
            with self.assertRaises(ValueError):
                parse_base_urls(value)

    def test_all_servers_started_before_readiness_and_no_client_launched(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(gpus=[0, 1, 2], ports=[19100, 19200, 19300],
                                   logs=directory, model='/shared/model', model_name='model',
                                   vllm='vllm', timeout=30)
            started = []
            def start(command, **kwargs):
                started.append((command, kwargs))
                process = Mock(pid=1000 + len(started))
                process.poll.return_value = None
                return process
            def ready(url, name):
                self.assertEqual(len(started), 3)
                return True
            with patch.object(launcher.subprocess, 'check_output', return_value='0'), \
                 patch.object(launcher.socket, 'socket'), \
                 patch.object(launcher.subprocess, 'Popen', side_effect=start), \
                 patch.object(launcher, 'service_ready', side_effect=ready), \
                 patch.object(launcher, 'terminate_owned') as cleanup:
                urls = launcher.run_pool(args)
            cleanup.assert_not_called()
            self.assertEqual(len(started), 3)
            self.assertEqual(urls, ['http://127.0.0.1:%d/v1' % port for port in args.ports])
            self.assertEqual(json.loads((Path(directory) / 'base_urls.json').read_text()), urls)
            for index, (command, kwargs) in enumerate(started):
                self.assertEqual(command[:2], ['vllm', 'serve'])
                self.assertEqual(command[command.index('--data-parallel-size') + 1], '1')
                self.assertEqual(command[command.index('--tensor-parallel-size') + 1], '1')
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], str(index))
                self.assertTrue(kwargs['start_new_session'])
                self.assertTrue(kwargs['stdout'].closed)

    def test_startup_failure_cleans_only_owned_servers(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(gpus=[0, 1], ports=[19100, 19200], logs=directory,
                                   model='/shared/model', model_name='model', vllm='vllm', timeout=30)
            failed = Mock(pid=1000)
            failed.poll.return_value = 1
            other = Mock(pid=1001)
            other.poll.return_value = None
            with patch.object(launcher.subprocess, 'check_output', return_value='0'), \
                 patch.object(launcher.socket, 'socket'), \
                 patch.object(launcher.subprocess, 'Popen', side_effect=[failed, other]), \
                 patch.object(launcher, 'terminate_owned') as cleanup, \
                 self.assertRaisesRegex(RuntimeError, 'before readiness'):
                launcher.run_pool(args)
            cleanup.assert_called_once_with([failed, other])
            self.assertFalse((Path(directory) / 'base_urls.json').exists())

    def test_nonduplex_parent_assigns_input_urls_to_four_children(self):
        baseline = importlib.import_module('nonduplex.eval_4tasks_group')
        with tempfile.TemporaryDirectory() as directory:
            urls = ['http://127.0.0.1:%d/v1' % port for port in (19100, 19200, 19300, 19400)]
            started = []
            def start(command, **kwargs):
                self.assertTrue(command[1].endswith('run_eval_task.py'))
                self.assertEqual(command[5], '0,1,2')
                tag, task, out = command[2:5]
                for episode in range(3):
                    folder = Path(out) / tag / task.replace('/', '_') / ('ep%d' % episode)
                    folder.mkdir(parents=True)
                    (folder / 'result.json').write_text(json.dumps({'ok': True, 'steps': 1}))
                started.append((command, kwargs))
                process = Mock(pid=2000 + len(started), args=command, returncode=0)
                def poll():
                    self.assertEqual(len(started), 4)
                    return 0
                process.poll.side_effect = poll
                return process
            argv = ['eval_4tasks_group.py', '--out', directory, '--models', 'test',
                    '--workers', '4', '--stagger', '0', '--base-urls', ','.join(urls)]
            with patch.object(sys, 'argv', argv), \
                 patch.object(baseline.subprocess, 'Popen', side_effect=start), \
                 patch.object(baseline.time, 'sleep'):
                baseline.main()
            assigned = [kwargs['env']['EVAL_BASE_URL'] for _, kwargs in started]
            self.assertEqual(assigned, urls)
            self.assertEqual(len(set(assigned)), 4)
            self.assertTrue(all(kwargs['start_new_session'] for _, kwargs in started))
            routes = json.loads((Path(directory) / '_logs/test_task_servers.json').read_text())
            self.assertEqual(list(routes.values()), urls)
            summary = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertTrue(all(row['test'] == '3/3' for row in summary.values()))

    def test_eval_shells_do_not_launch_or_fix_server_addresses(self):
        files = [ROOT / 'duplex/run_duplex500_eval.sh',
                 Path('/apdcephfs/private_chengzhixu/evaluation/nonduplex/restart_eval_flat3_exp_v3_1936.sh'),
                 Path('/apdcephfs/private_chengzhixu/auto_eval_flat3_exp_v3.sh')]
        for path in files:
            text = path.read_text()
            self.assertIn('--base-urls', text)
            self.assertNotIn('vllm_eval_pool.py', text)
            self.assertNotIn('CUDA_VISIBLE_DEVICES=', text)
            self.assertNotIn('http://127.0.0.1:8100', text)


if __name__ == '__main__':
    unittest.main()
