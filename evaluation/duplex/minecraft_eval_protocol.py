import sys as _sys
for _p in ('/apdcephfs/private_chengzhixu/evaluation', '/apdcephfs/private_chengzhixu/JarvisVLA'):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

CONFIG_DIR = Path('/apdcephfs/private_chengzhixu/JarvisVLA/jarvisvla/evaluate/config')
TASKS = {
    'kill/kill_zombie': ('normal', 2, 300),
    'mine/mine_stone': ('normal', 2, 500),
    'craft/craft_crafting_table': ('recipe', 4, 400),
    'smelt/iron_ingot': ('recipe', 4, 400),
}
DEFAULT_SEED = 42
REPEATS = 3
FIXED_TASK_INSTRUCTIONS = {
    'craft item crafting_table': 'Craft a crafting table',
    'kill_entity:zombie': 'Kill the zombie.',
    'mine_block:stone': 'Gather stone by using a pickaxe.',
    'craft_item:iron_ingot': 'Use the furnace to make an iron ingot.',
}


def seed_host(seed):
    if not 0 <= seed < 2 ** 32:
        raise ValueError('Host seed must be within NumPy uint32 range')
    random.seed(seed)
    np.random.seed(seed)


def build_task_instruction(task_text, episode=0, base_seed=DEFAULT_SEED, ep_dir=None):
    seed = base_seed + episode
    if not 0 <= seed < 2 ** 32:
        raise ValueError('Instruction seed must be within NumPy uint32 range')
    if task_text not in FIXED_TASK_INSTRUCTIONS:
        raise ValueError('No fixed instruction configured for task: %s' % task_text)
    instruction = FIXED_TASK_INSTRUCTIONS[task_text]
    if ep_dir is not None:
        directory = Path(ep_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'instruction.json').open('x') as file:
            json.dump({'task_text': task_text, 'instruction': instruction,
                       'instruction_type': 'normal', 'episode_seed': seed,
                       'selection': 'user_fixed'}, file, ensure_ascii=False, indent=2)
    return instruction


def load_task_cfg(task):
    import hydra
    if task not in TASKS:
        raise ValueError('Unsupported task: %s' % task)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    task_path = Path(task)
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR / task_path.parent), version_base='1.3'):
        return hydra.compose(config_name=task_path.name)


def build_env(cfg, record_dir, base_seed=DEFAULT_SEED):
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.entry import CameraConfig
    from minestudio.simulator.callbacks import (
        SpeedTestCallback, RecordCallback, RewardsCallback, TaskCallback,
        FastResetCallback, SummonMobsCallback, CommandsCallback,
    )
    from jarvisvla.evaluate.evaluate import _create_init_inventory_callback

    seed_host(base_seed)
    callbacks = [
        FastResetCallback(biomes=cfg.candidate_preferred_spawn_biome,
                          random_tp_range=cfg.random_tp_range, start_time=cfg.start_time),
        SpeedTestCallback(50),
        TaskCallback(getattr(cfg, 'task_conf', None)),
        RewardsCallback(getattr(cfg, 'reward_conf', None)),
        _create_init_inventory_callback(cfg),
        CommandsCallback(getattr(cfg, 'command', [])),
        RecordCallback(record_path=Path(record_dir), fps=30, show_actions=False),
    ]
    if cfg.mobs:
        callbacks.append(SummonMobsCallback(cfg.mobs))
    env = MinecraftSim(
        action_type='env', seed=cfg.seed, obs_size=cfg.origin_resolution,
        render_size=cfg.resize_resolution, camera_config=CameraConfig(**cfg.camera_config),
        preferred_spawn_biome=getattr(cfg, 'preferred_spawn_biome', None), callbacks=callbacks,
    )
    return env, callbacks


def prepare_episode(env, record_callback, cfg, ep_dir, episode, base_seed=DEFAULT_SEED):
    from omegaconf import OmegaConf

    episode_seed = base_seed + episode
    seed_host(episode_seed)
    env.np_random = np.random.default_rng(episode_seed)
    record_callback.record_path = Path(ep_dir)
    env.action_type = 'env'
    obs, info = env.reset()
    pre_agent = None
    worker_type = getattr(cfg, 'worker', None)
    if worker_type == 'craft':
        from jarvisvla.evaluate.env_helper.craft_agent import CraftWorker
        pre_agent = CraftWorker(env, if_discrete=True)
    elif worker_type == 'smelt':
        from jarvisvla.evaluate.env_helper.smelt_agent import SmeltWorker
        pre_agent = SmeltWorker(env, if_discrete=True)
    need_crafting_table = False
    if getattr(cfg, 'need_gui', False):
        if pre_agent is None:
            raise ValueError('GUI task requires a craft or smelt initialization worker')
        need_crafting_table = getattr(cfg, 'need_crafting_table', False)
        if need_crafting_table:
            pre_agent.open_crating_table_wo_recipe()
        elif getattr(cfg, 'need_furnace', False):
            pre_agent.open_furnace_wo_recipe()
        else:
            pre_agent._null_action(1)
            if not pre_agent.info['isGuiOpen']:
                pre_agent._call_func('inventory')
        obs, info = pre_agent.obs, pre_agent.info
    env.action_type = 'agent'
    protocol = {
        'protocol': 'shared_minecraft_eval_v1', 'episode': episode,
        'world_seed': int(cfg.seed), 'base_seed': base_seed,
        'python_seed': episode_seed, 'numpy_seed': episode_seed, 'sim_np_random_seed': episode_seed,
        'reset': 'same env per task; first hard reset then fast resets in episode order',
        'config_dir': str(CONFIG_DIR), 'config': OmegaConf.to_container(cfg, resolve=True),
        'initial_pov_sha256': hashlib.sha256(np.asarray(info['pov']).tobytes()).hexdigest(),
        'initial_gui_open': bool(info.get('isGuiOpen', False)),
        'worker': worker_type, 'need_crafting_table': need_crafting_table,
    }
    Path(ep_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(ep_dir, 'environment.json'), 'w') as file:
        json.dump(protocol, file, ensure_ascii=False, indent=2)
    return obs, info, need_crafting_table


def finish_episode_recording(record_callback):
    record_callback._save_episode()
    record_callback.episode_id += 1
