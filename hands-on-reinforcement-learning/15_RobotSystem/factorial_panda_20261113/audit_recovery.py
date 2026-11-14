"""Read-only, post-completion integrity and safety audit for the recovery experiment."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def finite_vector(value, length):
    return (isinstance(value, list) and len(value) == length and
            all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value))


def eval_audit(folder, split, protocol, protocol_sha, checkpoint):
    config, summary = read(folder / 'config.json'), read(folder / 'summary.json')
    cases = protocol[split + '_cases']
    expected_hash = digest(checkpoint) if checkpoint else None
    expected_controller = 'sac' if checkpoint else 'position_target'
    require(config['cases'] == cases and config['split'] == split, f'Cases/split mismatch: {folder}')
    require(config['protocol_sha256'] == protocol_sha and config['checkpoint_sha256'] == expected_hash,
            f'Protocol/checkpoint hash mismatch: {folder}')
    require(config['controller'] == expected_controller and summary['controller'] == expected_controller,
            f'Controller mismatch: {folder}')
    episodes = summary['per_episode']
    require(summary['episodes'] == len(cases) == len(episodes), f'Episode count mismatch: {folder}')
    require([[e['seed'], e['goal_rad']] for e in episodes] == cases, f'Per-case mismatch: {folder}')
    outcomes = Counter(e['outcome'] for e in episodes)
    require(set(outcomes) <= {'success', 'timeout', 'failure'}, f'Invalid outcome: {folder}')
    require(summary['successes'] == outcomes['success'], f'Success count mismatch: {folder}')
    by_case = {e['seed']: e for e in episodes}
    require(len(by_case) == len(cases), f'Duplicate case ID: {folder}')
    by_outcome = defaultdict(lambda: {'episodes': 0, 'physics_warning_episodes': 0,
                                      'limit_violation_episodes': 0, 'nonfinite_final_episodes': 0,
                                      'nonfinite_step_rows': 0, 'steps': 0,
                                      'action_change_means': []})
    case_actions = defaultdict(list)
    case_steps = Counter()
    step_warnings, step_limits, step_bad = set(), set(), set()
    for row in lines(folder / 'steps.jsonl'):
        case_id = row['seed']
        require(case_id in by_case, f'Unknown step case: {folder}')
        case_steps[case_id] += 1
        require(row['step'] == case_steps[case_id], f'Nonconsecutive step: {folder}')
        outcome = by_case[case_id]['outcome']
        by_outcome[outcome]['steps'] += 1
        if row.get('physics_warning'):
            step_warnings.add(case_id)
        if row.get('actual_limit_violation'):
            step_limits.add(case_id)
        valid = (finite_vector(row.get('observation'), 8) and finite_vector(row.get('next_observation'), 8)
                 and finite_vector(row.get('action_normalized'), 2))
        if not valid:
            by_outcome[outcome]['nonfinite_step_rows'] += 1
            step_bad.add(case_id)
        case_actions[case_id].append(row.get('action_normalized'))
    for e in episodes:
        case_id, outcome = e['seed'], e['outcome']
        require(case_steps[case_id] == e['steps'], f'Incomplete step log: {folder} case {case_id}')
        slot = by_outcome[outcome]
        slot['episodes'] += 1
        slot['physics_warning_episodes'] += bool(e.get('physics_warning') or case_id in step_warnings)
        slot['limit_violation_episodes'] += bool(e.get('actual_limit_violation') or case_id in step_limits)
        slot['nonfinite_final_episodes'] += not (finite_vector(e.get('final_abs_error_rad'), 2)
                                                   and finite_vector(e.get('final_abs_speed_rad_s'), 2))
        actions = case_actions[case_id]
        if len(actions) >= 2 and all(finite_vector(x, 2) for x in actions):
            slot['action_change_means'].append(mean(math.hypot(b[0]-a[0], b[1]-a[1])
                                                     for a, b in zip(actions, actions[1:])))
    for slot in by_outcome.values():
        values = slot.pop('action_change_means')
        slot['episodes_with_smoothness'] = len(values)
        slot['mean_action_change_per_step'] = mean(values) if values else None
    return {'outcomes': dict(outcomes), 'by_outcome': dict(by_outcome),
            'cases_with_nonfinite_steps': len(step_bad)}


def audit(root):
    import torch
    protocol = read(root / 'protocol.json')
    status = list(lines(root / 'status.jsonl'))
    require(status and status[-1]['stage'] == 'job' and status[-1]['status'] == 'complete',
            'Experiment is not complete; do not audit a partial run')
    proto_sha = digest(root / 'protocol.json')
    frozen = [s for s in status if s.get('stage') == 'protocol' and s.get('status') == 'frozen']
    require(len(frozen) == 1 and frozen[0]['sha256'] == proto_sha, 'Frozen protocol hash mismatch')
    require(not any(x.get('status') in ('failed', 'terminated') for x in status), 'Failed/terminated job stage')
    seeds = protocol['seeds']
    arms = ['sac', 'her', 'her_shape', 'her_smooth', 'her_both']
    milestones = [10000, 20000, 30000, 40000, 50000]
    require(seeds == [1103, 1129, 1181, 1213, 1277] and protocol['milestones'] == milestones
            and protocol['steps'] == 50000 and not protocol['smoke_only'], 'Unexpected seed/budget')
    require(set(protocol['arms']) == set(arms) and protocol['training_device'] == 'cuda'
            and protocol['evaluation_device'] == 'cpu'
            and protocol['failure_training_base_reward'] == -250.0, 'Unexpected protocol')
    require(protocol['validation_goal_seed'] == 2026111311 and protocol['test_goal_seed'] == 2026111313
            and len(protocol['validation_cases']) == len(protocol['test_cases']) == 48, 'Unexpected test/validation')
    expected_stages = {'baseline-validation', 'baseline-test'}
    for seed in seeds:
        for arm in arms:
            expected_stages.add(f'train-{seed}-{arm}')
            expected_stages.add(f'test-{seed}-{arm}')
            expected_stages.update(f'val-{seed}-{arm}-{step}' for step in milestones)
    for stage in expected_stages:
        completed = [s for s in status if s.get('stage') == stage and s.get('status') == 'complete']
        require(len(completed) == 1 and completed[0].get('exit_code') == 0, f'Incomplete stage: {stage}')
    va = {tuple(x[1]) for x in protocol['validation_cases']}
    te = {tuple(x[1]) for x in protocol['test_cases']}
    require(len(va) == len(te) == 48 and not va.intersection(te), 'Duplicated/overlapping goals')
    environment = protocol['environment']
    frozen_sources = environment['sources_sha256']
    versions = environment['versions']
    frozen_code = {path: h for path, h in frozen_sources.items()
                   if Path(path).name.lower() not in ('scene.xml', 'panda.xml')}
    scene_hashes = {Path(path).name.lower(): h for path, h in frozen_sources.items()
                    if Path(path).name.lower() in ('scene.xml', 'panda.xml')}
    require(set(scene_hashes) == {'scene.xml', 'panda.xml'}, 'Missing scene/model hashes')
    result = {'protocol_sha256': proto_sha, 'training_device': protocol['training_device'],
              'evaluation_device': protocol['evaluation_device'], 'versions': versions,
              'baseline': {}, 'runs': []}
    for split in ('validation', 'test'):
        result['baseline'][split] = eval_audit(root / 'baseline' / split, split, protocol, proto_sha, None)
    common_config = None
    for seed in seeds:
        for arm in arms:
            base = root / f'seed-{seed}' / arm
            cfg, summary = read(base / 'config.json'), read(base / 'train-summary.json')
            require(cfg['seed'] == seed and cfg['arm'] == arm and cfg['steps'] == protocol['steps']
                    and cfg['protocol_sha256'] == proto_sha, f'Training identity mismatch: {base}')
            require([cfg['her'], cfg['shape'], cfg['smooth']] == protocol['arms'][arm], f'Arm flags mismatch: {base}')
            require(cfg['config']['device'] == protocol['training_device']
                    and cfg['torch'] == versions['torch'] and cfg['numpy'] == versions['numpy'],
                    f'Training device/version mismatch: {base}')
            require(cfg['source_sha256'] == frozen_code and
                    cfg['xml_sha256'] == {'scene': scene_hashes['scene.xml'], 'panda': scene_hashes['panda.xml']},
                    f'Training source hash mismatch: {base}')
            if common_config is None:
                common_config = cfg['config']
            require(cfg['config'] == common_config, f'SAC config mismatch: {base}')
            require(summary['status'] == 'PASS' and summary['steps'] == protocol['steps']
                    and summary['updates'] == protocol['steps'] - protocol['warmup'], f'Training summary: {base}')
            episodes = list(lines(base / 'episodes.jsonl'))
            outcomes = Counter(e['outcome'] for e in episodes)
            require(set(outcomes) <= {'success', 'timeout', 'failure'}, f'Unexpected training outcome: {base}')
            require(dict(outcomes) == summary['outcomes'] and len(episodes) == summary['completed_episodes'],
                    f'Training outcome count mismatch: {base}')
            failures = outcomes['failure']
            rate = failures / len(episodes) if episodes else None
            require((rate is None and summary['failure_rate'] is None) or
                    (rate is not None and math.isclose(rate, summary['failure_rate'], abs_tol=1e-12)),
                    f'Failure rate mismatch: {base}')
            require(all(e.get('actual_limit_violation') and not e.get('physics_warning')
                        and e.get('final_replay_train_base_reward') == -250.0
                        for e in episodes if e['outcome'] == 'failure'), f'Invalid failure semantics: {base}')
            result_row = {'seed': seed, 'arm': arm, 'training': {'outcomes': dict(outcomes),
                          'completed_episodes': len(episodes), 'failure_rate': rate,
                          'unlogged_partial_steps': protocol['steps'] - sum(e['episode_steps'] for e in episodes)},
                          'validation': {}, 'test': None}
            result_row['training']['actor_smoothness_by_outcome'] = {}
            for kind in ('success', 'timeout', 'failure'):
                group = [e for e in episodes if e['outcome'] == kind]
                variations = [e['actor_action_tv'] / e['actor_action_pairs'] for e in group
                              if e['actor_action_pairs'] > 0]
                result_row['training']['actor_smoothness_by_outcome'][kind] = {
                    'episodes': len(group), 'episodes_with_pairs': len(variations),
                    'mean_action_change_per_step': mean(variations) if variations else None,
                    'note': 'actor-only; excludes warmup actions'}
            require(result_row['training']['unlogged_partial_steps'] >= 0, f'Too many episode steps: {base}')
            for step in milestones:
                ckpt = base / f'checkpoint-{step}.pt'
                checkpoint = torch.load(ckpt, map_location='cpu', weights_only=True)
                require(checkpoint['seed'] == seed and checkpoint['arm'] == arm and checkpoint['step'] == step
                        and checkpoint['config'] == cfg['config']
                        and checkpoint['run_config']['xml_sha256'] == cfg['xml_sha256'],
                        f'Checkpoint identity/config mismatch: {ckpt}')
                require(all(bool(torch.isfinite(t).all()) for state in checkpoint['networks'].values()
                            for t in state.values() if isinstance(t, torch.Tensor)), f'Nonfinite checkpoint: {ckpt}')
                result_row['validation'][str(step)] = eval_audit(base / 'validation' / str(step), 'validation',
                                                                  protocol, proto_sha, ckpt)
            result_row['test'] = eval_audit(base / 'test', 'test', protocol, proto_sha,
                                            base / 'checkpoint-50000.pt')
            result['runs'].append(result_row)
    warnings = []
    def inspect_issues(label, evaluation):
        for outcome, counts in evaluation['by_outcome'].items():
            for key in ('physics_warning_episodes', 'nonfinite_final_episodes', 'nonfinite_step_rows'):
                if counts[key]:
                    warnings.append({'stage': label, 'outcome': outcome, 'issue': key, 'count': counts[key]})
    for split, evaluation in result['baseline'].items():
        inspect_issues(f'baseline/{split}', evaluation)
    for run in result['runs']:
        for step, evaluation in run['validation'].items():
            inspect_issues(f"seed-{run['seed']}/{run['arm']}/validation/{step}", evaluation)
        inspect_issues(f"seed-{run['seed']}/{run['arm']}/test", run['test'])
    result['warnings'] = warnings
    result['status'] = 'WARN' if warnings else 'PASS'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='completed recovery result directory')
    args = parser.parse_args()
    print(json.dumps(audit(args.root), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
