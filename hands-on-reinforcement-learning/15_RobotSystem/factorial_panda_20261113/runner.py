"""Pre-registered Panda factorial experiment. Default action is dry-run.

Training/evaluation run only via --run, or explicit --smoke for tiny checks.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter, sleep

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = Path.home() / 'Documents' / 'ChatGPT' / 'EIP' / 'panda-factorial-20261113'
SCENE = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData' / 'Local'))) / 'EmbodiedAI' / 'PandaSmoke' / 'menagerie-98a70c5' / 'franka_emika_panda' / 'scene.xml'
SEEDS = (1103, 1129, 1181, 1213, 1277)
ARMS = {'sac': (False, False, False), 'her': (True, False, False),
        'her_shape': (True, True, False), 'her_smooth': (True, False, True),
        'her_both': (True, True, True)}
STEPS = 50000
MILESTONES = (10000, 20000, 30000, 40000, 50000)


def write(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def log(path, data):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def generate_cases(seed, base_id, count_per_quadrant=12):
    rng = random.Random(seed)
    rows = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        for _ in range(count_per_quadrant):
            rows.append([base_id + len(rows), [round(sx * rng.uniform(.04, .16), 6),
                                               round(sy * rng.uniform(.04, .16), 6)]])
    return rows


def protocol(smoke=False, device='cpu'):
    count = 1 if smoke else 12
    validation = generate_cases(2026111311, 50101, count)
    test = generate_cases(2026111312, 60101, count)
    validate_goal_exclusions(validation, test)
    return {'seeds': [1103] if smoke else list(SEEDS), 'arms': ARMS, 'training_device': device,
            'evaluation_device': 'cpu', 'concurrent_workers': 5,
            'steps': 320 if smoke else STEPS, 'milestones': [320] if smoke else list(MILESTONES),
            'validation_goal_seed': 2026111311, 'test_goal_seed': 2026111312,
            'validation_cases': validation, 'test_cases': test,
            'goal_distribution': 'equal quadrant counts, independent uniform magnitudes .04 to .16 rad',
            'warmup': 256, 'batch': 64, 'smooth_alpha': .5,
            'smooth_definition': 'actor-only behavior: executed=0.5*raw+0.5*previous, reset previous to 0 each episode; warmup unfiltered',
            'shaping_definition': 'Phi=-max(abs(q-goal)); r_train=r_base+gamma*Phi(next)-Phi(state); Phi(next)=0 on terminated, not truncated',
            'success': 'no physics failure, max position error < .01 rad and max speed < .05 rad/s',
            'baseline': 'existing position_target controller; evaluated once per split',
            'script_sha256': digest(Path(__file__)), 'smoke_only': smoke}


def validate_goal_exclusions(validation, test):
    """Check exact goal overlap, not minimum separation or distribution shift."""
    v = [tuple(row[1]) for row in validation]
    t = [tuple(row[1]) for row in test]
    if len(set(v)) != len(v) or len(set(t)) != len(t) or set(v) & set(t):
        raise ValueError('Duplicated validation/test goal')
    earlier = {(x, y) for x in (-.16, -.08, .08, .16) for y in (-.16, -.08, .08, .16)}
    earlier.update({(.10,.10),(-.10,.10),(.12,.14),(-.14,.12),(.14,-.12),(-.12,-.14)})
    old = ROOT.parent
    precision = old/'panda-sac-her-precision-20261112-222442'/'protocol.json'
    multiseed = old/'panda-warmup-multiseed-20261113'/'protocol.json'
    for f, keys in ((precision, ('diagnostic_cases', 'fresh_cases')), (multiseed, ('cases',))):
        data = json.loads(f.read_text(encoding='utf-8'))
        for key in keys:
            earlier.update(tuple(row[1]) for row in data[key])
    duplicates = (set(v) | set(t)) & earlier
    if duplicates:
        raise ValueError(f'Goal overlaps with historical set: {duplicates}')


def environment_snapshot(scene):
    import numpy as np
    import torch
    import mujoco
    files = [Path(__file__), PROJECT/'panda_reach.py', PROJECT/'panda_her.py',
             PROJECT/'panda_two_joint_env.py', PROJECT/'panda_sac_smoke.py',
             PROJECT.parent/'11_SAC'/'Pendulum'/'agents.py',
             PROJECT.parent/'11_SAC'/'Pendulum'/'networks.py',
             PROJECT.parent/'11_SAC'/'Pendulum'/'rl_utils.py', scene, scene.with_name('panda.xml')]
    return {'sources_sha256': {str(f.resolve()): digest(f) for f in files},
            'versions': {'numpy': np.__version__, 'torch': str(torch.__version__),
                         'torch_cuda': torch.version.cuda, 'mujoco': mujoco.__version__, 'python': sys.version},
            'cuda_device': ({'name': torch.cuda.get_device_name(0),
                             'capability': list(torch.cuda.get_device_capability(0)),
                             'architectures': torch.cuda.get_arch_list()}
                            if torch.cuda.is_available() else None)}


def verify_environment(p, scene):
    if p['environment'] != environment_snapshot(scene):
        raise RuntimeError('Source, scene, model, or dependency changed since protocol freeze')


def check_device(device):
    import torch
    if device != 'cuda':
        return
    if not torch.version.cuda or not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but this Python environment has no usable CUDA GPU')
    major, minor = torch.cuda.get_device_capability(0)
    expected = f'sm_{major}{minor}'
    architectures = torch.cuda.get_arch_list()
    if expected not in architectures and f'compute_{major}{minor}' not in architectures:
        raise RuntimeError(f'CUDA GPU {expected} is unsupported by this PyTorch build: {architectures}')


def shaped_rewards(states, rewards, next_states, dones, gamma):
    import numpy as np
    # HER has already updated both goal slices and reward/terminated in the sampled batch.
    phi = -np.max(np.abs(states[:, :2] - states[:, 4:6]), axis=1)
    phi_next = -np.max(np.abs(next_states[:, :2] - next_states[:, 4:6]), axis=1)
    phi_next = np.where(np.asarray(dones, dtype=bool), 0.0, phi_next)
    return np.asarray(rewards) + gamma * phi_next - phi


def smooth_behavior(raw, previous):
    return .5 * raw + .5 * previous


def sample_her_with_flags(replay, batch_size):
    """Same uniform source sample and relabel rule as PandaHERReplayBuffer.sample.

    Adds a per-sample relabel flag for diagnostics, without changing RNG use.
    """
    import numpy as np
    experiences = random.sample(replay.buffer, batch_size)
    counts = {'attempts': batch_size // 2, 'relabeled': 0, 'relabeled_success': 0,
              'no_future_goal': 0, 'already_at_goal': 0, 'source_failed': 0}
    rows, flags = [], []
    for i, experience in enumerate(experiences):
        if i < counts['attempts']:
            transition, result = replay._relabel(experience)
            counts[result] += 1
            if result == 'relabeled' and transition[-1]:
                counts['relabeled_success'] += 1
        else:
            transition, result = experience.transition, 'original'
        rows.append(transition)
        flags.append(result == 'relabeled')
    counts['original'] = batch_size - counts['relabeled']
    return tuple(np.asarray(values) for values in zip(*rows)), counts, flags


def train(root, seed, arm, scene):
    sys.path.insert(0, str(PROJECT))
    import numpy as np
    import torch
    from panda_her import PandaHERReplayBuffer
    from panda_two_joint_env import PandaTwoJointEnv
    from panda_sac_smoke import CONFIG, ReplayBuffer, SAC, checkpoint
    import panda_reach as original

    p = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    verify_environment(p, scene)
    check_device(p['training_device'])
    config = {**CONFIG, 'device': p['training_device']}
    her, shape, smooth = p['arms'][arm]
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    goal_rng = np.random.default_rng(seed)
    action_rng = np.random.default_rng(seed + 1)
    env = PandaTwoJointEnv(scene, max_steps=250)
    agent = SAC(**config)
    replay = PandaHERReplayBuffer(100000, env.lower, env.upper, seed + 2) if her else ReplayBuffer(100000)
    base = root / f'seed-{seed}' / arm
    base.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), PROJECT/'panda_reach.py', PROJECT/'panda_her.py',
               PROJECT/'panda_two_joint_env.py', PROJECT/'panda_sac_smoke.py']
    import panda_sac_smoke
    sources += [panda_sac_smoke.SAC_DIR / x for x in ('agents.py', 'networks.py', 'rl_utils.py')]
    write(base / 'config.json', {'seed': seed, 'arm': arm, 'her': her, 'shape': shape, 'smooth': smooth,
          'config': config, 'protocol_sha256': digest(root/'protocol.json'),
          'xml_sha256': {'scene': digest(scene), 'panda': digest(scene.with_name('panda.xml'))},
          'source_sha256': {str(f): digest(f) for f in sources}, 'numpy': np.__version__,
          'torch': str(torch.__version__), 'steps': p['steps']})
    obs, _ = env.reset(goal=original.TRAIN_GOALS[goal_rng.integers(len(original.TRAIN_GOALS))])
    previous = np.zeros(2, dtype=float)
    episode = 1
    outcomes = Counter()
    her_counts = Counter()
    actor_previous = None
    episode_stats = {'actor_action_tv': 0., 'actor_action_pairs': 0,
                     'joint_path_rad': 0., 'velocity_delta_rad_s': 0., 'steps': 0}
    started = perf_counter()
    episodes_file = base / 'episodes.jsonl'
    steps_file = base / 'steps.jsonl'
    updates_file = base / 'updates.jsonl'
    for step in range(1, p['steps']+1):
        warmup = step <= p['warmup']
        raw = action_rng.uniform(-1, 1, 2) if warmup else agent.take_action(obs)
        action = smooth_behavior(raw, previous) if smooth and not warmup else raw
        if smooth and not warmup:
            previous = action.copy()
        action = original.normalized_action(action)
        next_obs, reward, terminated, truncated, info = env.step(original.ACTION_SCALE * action)
        if info['outcome'] == 'failure' or not np.isfinite(next_obs).all():
            raise RuntimeError(f'Physical failure at step {step}: {info}')
        if her:
            replay.add(obs, action, reward, next_obs, terminated, truncated=truncated, failed=False)
        else:
            replay.add(obs, action, reward, next_obs, terminated)
        episode_stats['steps'] += 1
        episode_stats['joint_path_rad'] += float(np.linalg.norm(next_obs[:2]-obs[:2]))
        episode_stats['velocity_delta_rad_s'] += float(np.linalg.norm(next_obs[2:4]-obs[2:4]))
        if not warmup:
            if actor_previous is not None:
                episode_stats['actor_action_tv'] += float(np.linalg.norm(action-actor_previous))
                episode_stats['actor_action_pairs'] += 1
            actor_previous = action.copy()
        if step in (1, p['steps']) or step % 250 == 0:
            log(steps_file, {'step': step, 'episode': episode, 'source': 'warmup' if warmup else 'actor',
                             'observation': original.vector(obs), 'next_observation': original.vector(next_obs),
                             'raw_action': original.vector(raw), 'executed_action': original.vector(action),
                             'reward_base': reward, 'terminated': terminated, 'truncated': truncated})
        if not warmup:
            if her:
                sample, counts, relabeled_flags = sample_her_with_flags(replay, p['batch'])
                her_counts.update(counts)
            else:
                sample = replay.sample(p['batch'])
                relabeled_flags = [False]*p['batch']
            sample_states, _, sample_rewards, _, _ = sample
            if shape:
                s, a, r, ns, done = sample
                shaped = shaped_rewards(s, r, ns, done, config['gamma'])
                adjustment = shaped - r
                sample = (s, a, shaped, ns, done)
            diagnostic = (step == p['steps'] or step % 1000 == 0)
            metrics = agent.update(sample, diagnostics=diagnostic)
            if diagnostic:
                log(updates_file, {'step': step, **metrics, 'her_total': dict(her_counts),
                    'shaping_mean': float(np.mean(adjustment)) if shape else 0.,
                    'shaping_min': float(np.min(adjustment)) if shape else 0.,
                    'shaping_max': float(np.max(adjustment)) if shape else 0.,
                    'sample_goals': sample_states[:, 4:6].tolist(),
                    'sample_relabeled': relabeled_flags,
                    'sample_base_rewards': sample_rewards.tolist(),
                    'sample_shaping': adjustment.tolist() if shape else [0.]*p['batch']})
        if terminated or truncated:
            outcomes[info['outcome']] += 1
            log(episodes_file, {'step': step, 'episode': episode, 'episode_steps': env.steps,
                                'outcome': info['outcome'], 'goal': original.vector(env.goal),
                                **episode_stats})
            obs, _ = env.reset(goal=original.TRAIN_GOALS[goal_rng.integers(len(original.TRAIN_GOALS))])
            previous = np.zeros(2, dtype=float)
            actor_previous = None
            episode_stats = {'actor_action_tv': 0., 'actor_action_pairs': 0,
                             'joint_path_rad': 0., 'velocity_delta_rad_s': 0., 'steps': 0}
            episode += 1
        else:
            obs = next_obs
        if step in p['milestones']:
            for name in ('actor', 'critic_1', 'critic_2', 'target_critic_1', 'target_critic_2'):
                if not all(torch.isfinite(v).all() for v in getattr(agent, name).parameters()):
                    raise RuntimeError(f'Non-finite parameters in {name} at step {step}')
            target = base / f'checkpoint-{step}.pt'
            torch.save({**checkpoint(agent), 'config': config, 'run_config': original.model_info(scene),
                        'step': step, 'seed': seed, 'arm': arm}, target)
    write(base / 'train-summary.json', {'status': 'PASS', 'steps': p['steps'],
          'updates': p['steps']-p['warmup'], 'outcomes': dict(outcomes),
          'her_counts': dict(her_counts), 'training_seconds': perf_counter()-started})


def evaluate(root, scene, split, out, checkpoint_path=None):
    sys.path.insert(0, str(PROJECT))
    import numpy as np
    import torch
    import panda_reach as original
    from panda_two_joint_env import PandaTwoJointEnv
    from panda_sac_smoke import SAC
    p = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    verify_environment(p, scene)
    torch.set_num_threads(1)
    cases = p[split + '_cases']
    if checkpoint_path:
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if saved['run_config']['xml_sha256'] != original.model_info(scene)['xml_sha256']:
            raise ValueError('Checkpoint/scene model mismatch')
        # Evaluation is CPU-only and does not load training optimizer states.
        eval_config = {**saved['config'], 'device': 'cpu'}
        agent = SAC(**eval_config)
        agent.actor.load_state_dict(saved['networks']['actor'])
        policy = lambda obs: original.deterministic_action(agent, obs)
        name = 'sac'
    else:
        policy = original.command_controller
        name = 'position_target'
    env = PandaTwoJointEnv(scene, max_steps=250)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    with (out/'steps.jsonl').open('x', encoding='utf-8') as f:
        for case_id, goal in cases:
            result = original.run_episode(env, policy, goal, case_id, 0, f)
            rows.append(result)
    write(out/'config.json', {'controller': name, 'split': split, 'cases': cases,
          'checkpoint_sha256': digest(checkpoint_path) if checkpoint_path else None,
          'protocol_sha256': digest(root/'protocol.json'), 'max_episode_steps': 250})
    write(out/'summary.json', {'controller': name, 'episodes': len(rows),
          'successes': sum(x['outcome']=='success' for x in rows), 'per_episode': rows})
    if len(rows) != len(cases) or [[r['seed'], r['goal_rad']] for r in rows] != cases:
        raise RuntimeError('Incomplete or mismatched evaluation cases')


def run(root, scene, smoke=False, device='cpu'):
    import numpy as np
    import torch
    import mujoco
    if root.exists():
        raise FileExistsError(f'Result directory exists: {root}')
    if not scene.is_file() or not scene.with_name('panda.xml').is_file():
        raise FileNotFoundError(f'Panda scene or model missing: {scene}')
    check_device(device)
    p = protocol(smoke, device)
    p['environment'] = environment_snapshot(scene)
    root.mkdir(parents=True)
    write(root/'protocol.json', p)
    status = root/'status.jsonl'
    protocol_sha = digest(root/'protocol.json')
    log(status, {'stage': 'protocol', 'status': 'frozen', 'sha256': protocol_sha,
                 'utc': datetime.now(timezone.utc).isoformat()})
    active = {}
    def launch_batch(jobs):
        if digest(root/'protocol.json') != protocol_sha:
            raise RuntimeError('Protocol changed after freezing')
        env = dict(__import__('os').environ)
        env.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'NUMEXPR_NUM_THREADS': '1', 'PYTHONIOENCODING': 'utf-8'})
        try:
            for stage, args in jobs:
                stdout = (root / f'{stage}.stdout.log').open('xb')
                try:
                    stderr = (root / f'{stage}.stderr.log').open('xb')
                except BaseException:
                    stdout.close()
                    raise
                try:
                    proc = subprocess.Popen(args, stdout=stdout, stderr=stderr, env=env,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except BaseException:
                    stdout.close(); stderr.close()
                    raise
                active[stage] = (proc, stdout, stderr, perf_counter())
                log(status, {'stage': stage, 'status': 'running', 'pid': proc.pid,
                             'utc': datetime.now(timezone.utc).isoformat(), 'command': args})
            while active:
                for stage, (proc, stdout, stderr, started) in list(active.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    stdout.close(); stderr.close()
                    del active[stage]
                    log(status, {'stage': stage, 'status': 'complete' if code == 0 else 'failed',
                                 'exit_code': code, 'seconds': perf_counter()-started,
                                 'utc': datetime.now(timezone.utc).isoformat()})
                    if code:
                        raise RuntimeError(f'{stage} failed: see {root / (stage + ".stderr.log")}')
                if active:
                    sleep(.2)
        finally:
            for stage, (proc, stdout, stderr, started) in list(active.items()):
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                stdout.close(); stderr.close()
                log(status, {'stage': stage, 'status': 'terminated', 'exit_code': proc.returncode,
                             'seconds': perf_counter()-started, 'utc': datetime.now(timezone.utc).isoformat()})
                del active[stage]
    def verify_result(out, split, ckpt):
        cfg = json.loads((out/'config.json').read_text(encoding='utf-8'))
        summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
        expected = p[split+'_cases']
        expected_sha = digest(ckpt) if ckpt else None
        if (cfg['cases'] != expected or cfg['checkpoint_sha256'] != expected_sha
                or cfg['protocol_sha256'] != protocol_sha or summary['episodes'] != len(expected)
                or len(summary['per_episode']) != len(expected)):
            raise RuntimeError(f'Evaluation integrity check failed: {out}')
    common = [sys.executable, str(Path(__file__).resolve())]
    try:
        launch_batch([('baseline-validation', common+['--internal-eval', str(root), str(scene), 'validation',
                                                      str(root/'baseline'/'validation'), '-'])])
        verify_result(root/'baseline'/'validation', 'validation', None)
        for seed in p['seeds']:
            launch_batch([(f'train-{seed}-{arm}', common+['--internal-train', str(root), str(seed), arm, str(scene)])
                          for arm in p['arms']])
            for arm in p['arms']:
                base = root/f'seed-{seed}'/arm
                summary = json.loads((base/'train-summary.json').read_text(encoding='utf-8'))
                if (summary['status'] != 'PASS' or summary['steps'] != p['steps']
                        or summary['updates'] != p['steps']-p['warmup']):
                    raise RuntimeError(f'Invalid training summary: {base}')
            jobs = []
            for arm in p['arms']:
                base = root/f'seed-{seed}'/arm
                for n in p['milestones']:
                    jobs.append((f'val-{seed}-{arm}-{n}', common+['--internal-eval', str(root), str(scene),
                                'validation', str(base/'validation'/str(n)), str(base/f'checkpoint-{n}.pt')]))
            for i in range(0, len(jobs), 5):
                launch_batch(jobs[i:i+5])
            for arm in p['arms']:
                base = root/f'seed-{seed}'/arm
                for n in p['milestones']:
                    verify_result(base/'validation'/str(n), 'validation', base/f'checkpoint-{n}.pt')
        # Only test after all training and validation have finished.
        launch_batch([('baseline-test', common+['--internal-eval', str(root), str(scene), 'test',
                                                str(root/'baseline'/'test'), '-'])])
        verify_result(root/'baseline'/'test', 'test', None)
        jobs = []
        for seed in p['seeds']:
            for arm in p['arms']:
                base = root/f'seed-{seed}'/arm
                jobs.append((f'test-{seed}-{arm}', common+['--internal-eval', str(root), str(scene), 'test',
                             str(base/'test'), str(base/f"checkpoint-{p['steps']}.pt")]))
        for i in range(0, len(jobs), 5):
            launch_batch(jobs[i:i+5])
        for seed in p['seeds']:
            for arm in p['arms']:
                base = root/f'seed-{seed}'/arm
                verify_result(base/'test', 'test', base/f"checkpoint-{p['steps']}.pt")
        log(status, {'stage': 'job', 'status': 'complete', 'utc': datetime.now(timezone.utc).isoformat()})
    except BaseException as e:
        log(status, {'stage': 'job', 'status': 'failed', 'error': str(e),
                     'utc': datetime.now(timezone.utc).isoformat()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='tiny 320-step smoke, one seed, all arms')
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--scene', type=Path, default=SCENE)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu', help='training device; evaluation remains CPU')
    parser.add_argument('--internal-train', nargs=4, metavar=('ROOT','SEED','ARM','SCENE'), help=argparse.SUPPRESS)
    parser.add_argument('--internal-eval', nargs=5, metavar=('ROOT','SCENE','SPLIT','OUT','CKPT'), help=argparse.SUPPRESS)
    a = parser.parse_args()
    if a.internal_train:
        root, seed, arm, scene = a.internal_train
        train(Path(root), int(seed), arm, Path(scene)); return
    if a.internal_eval:
        root, scene, split, out, ckpt = a.internal_eval
        evaluate(Path(root), Path(scene), split, Path(out), None if ckpt == '-' else Path(ckpt)); return
    if a.run and a.smoke:
        parser.error('choose --run or --smoke')
    if not a.run and not a.smoke:
        print(json.dumps({'mode': 'DRY RUN: no training, no output directory', 'output': str(a.output),
                          'scene': str(a.scene), 'protocol': protocol(device=a.device)}, indent=2))
        return
    run(a.output, a.scene, a.smoke, a.device)


if __name__ == '__main__':
    main()
