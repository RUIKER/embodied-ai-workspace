"""Read-only factorial results aggregation. Never starts training or evaluation."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import hypot, sqrt
from pathlib import Path
from statistics import mean, stdev


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def statistics(values):
    average = mean(values)
    margin = 2.776 * stdev(values) / sqrt(5)
    return {'per_seed_percentage_points': values, 'mean_pp': average,
            't95_interval_pp': [average - margin, average + margin]}


def training_metrics(path):
    by_outcome = defaultdict(list)
    with path.open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            by_outcome[row['outcome']].append(row)
    result = {}
    for outcome, rows in by_outcome.items():
        actor = [r['actor_action_tv'] / r['actor_action_pairs'] for r in rows if r['actor_action_pairs']]
        result[outcome] = {'completed_episodes': len(rows),
                           'episodes_with_actor_pairs': len(actor),
                           'mean_actor_action_change_per_step': mean(actor) if actor else None,
                           'mean_joint_path_length_rad': mean(r['joint_path_rad'] for r in rows),
                           'mean_velocity_change_rad_per_s_per_step': mean(
                               r['velocity_delta_rad_s'] / r['steps'] for r in rows)}
    return result


def trajectory_metrics(path, outcomes):
    by_case = defaultdict(list)
    with path.open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            by_case[row['seed']].append(row)
    groups = defaultdict(lambda: {'actions': [], 'speed_changes': [], 'paths': []})
    for case_id, rows in by_case.items():
        group = groups[outcomes[case_id]]
        sequence = [r['action_normalized'] for r in rows]
        if len(sequence) >= 2:
            group['actions'].append(mean(hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(sequence, sequence[1:])))
        states = [rows[0]['observation']] + [r['next_observation'] for r in rows]
        group['paths'].append(sum(hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(states, states[1:])))
        group['speed_changes'].append(mean(hypot(b[2]-a[2], b[3]-a[3]) for a, b in zip(states, states[1:])))
    return {label: {'episode_count': len(group['paths']), 'episodes_with_two_actions': len(group['actions']),
                    'mean_action_change_per_step': mean(group['actions']) if group['actions'] else None,
                    'mean_joint_path_length_rad': mean(group['paths']) if group['paths'] else None,
                    'mean_joint_velocity_change_rad_per_s_per_step': mean(group['speed_changes'])}
            for label, group in groups.items()}


def evaluate(folder, expected_cases, protocol_sha, checkpoint=None):
    config = load(folder / 'config.json')
    result = load(folder / 'summary.json')
    checkpoint_sha = sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None
    if (config['cases'] != expected_cases or result['episodes'] != len(expected_cases)
            or config['protocol_sha256'] != protocol_sha or config['checkpoint_sha256'] != checkpoint_sha):
        raise ValueError(f'Case mismatch: {folder}')
    episodes = result['per_episode']
    if [[e['seed'], e['goal_rad']] for e in episodes] != expected_cases:
        raise ValueError(f'Per-episode cases mismatch: {folder}')
    successes = sum(e['outcome'] == 'success' for e in episodes)
    if result['successes'] != successes:
        raise ValueError(f'Success count mismatch: {folder}')
    counts = {kind: sum(e['outcome'] == kind for e in episodes) for kind in ('success', 'timeout', 'failure')}
    if sum(counts.values()) != len(expected_cases):
        raise ValueError(f'Unexpected outcome: {folder}')
    return {'success_percent': successes * 100 / len(expected_cases), 'outcomes': counts,
            'trajectory_metrics_by_outcome': trajectory_metrics(folder / 'steps.jsonl',
                                                {e['seed']: e['outcome'] for e in episodes})}


def summarize(root):
    protocol = load(root / 'protocol.json')
    statuses = [json.loads(line) for line in (root / 'status.jsonl').read_text(encoding='utf-8').splitlines()]
    if not statuses or statuses[-1].get('stage') != 'job' or statuses[-1].get('status') != 'complete':
        raise ValueError('Experiment not complete; do not infer full-study results')
    seeds, checkpoints = protocol['seeds'], protocol['milestones']
    if len(seeds) != 5 or checkpoints != [10000, 20000, 30000, 40000, 50000]:
        raise ValueError('Expected five seeds and preregistered checkpoints')
    arms = ('sac', 'her', 'her_shape', 'her_smooth', 'her_both')
    protocol_sha = sha256((root / 'protocol.json').read_bytes()).hexdigest()
    out = {'seeds': seeds, 'runs': [], 'baseline': {}}
    for split in ('validation', 'test'):
        out['baseline'][split] = evaluate(root / 'baseline' / split, protocol[split + '_cases'], protocol_sha)
    for seed in seeds:
        result = {'seed': seed, 'arms': {}}
        for arm in arms:
            base = root / f'seed-{seed}' / arm
            training = load(base / 'train-summary.json')
            conf = load(base / 'config.json')
            if (training['status'] != 'PASS' or training['steps'] != 50000 or conf['seed'] != seed
                    or conf['arm'] != arm or conf['protocol_sha256'] != protocol_sha):
                raise ValueError(f'Training incomplete: {base}')
            validation = [evaluate(base / 'validation' / str(step), protocol['validation_cases'], protocol_sha,
                                   base / f'checkpoint-{step}.pt') for step in checkpoints]
            curve = [x['success_percent'] for x in validation]
            # Equal-spaced trapezoidal AUC divided by 40k, no unobserved zero-step value.
            auc = sum((x + y)/2 for x, y in zip(curve, curve[1:])) / 4
            first_80 = next((step for step, score in zip(checkpoints, curve) if score >= 80), None)
            result['arms'][arm] = {'validation': validation, 'validation_auc_percent': auc,
                                   'first_observed_80_percent_step': first_80,
                                   'threshold_censored_at_50000': first_80 is None,
                                   'test': evaluate(base / 'test', protocol['test_cases'], protocol_sha,
                                                    base / 'checkpoint-50000.pt'),
                                   'training_by_outcome': training_metrics(base / 'episodes.jsonl'),
                                   'training_seconds': training['training_seconds']}
        out['runs'].append(result)
    def diff(expression, key):
        return statistics([sum(coef * row['arms'][arm][key] for arm, coef in expression)
                           for row in out['runs']])
    out['primary_D_minus_A'] = diff([('her_both', 1), ('her', -1)], 'validation_auc_percent')
    out['B_minus_A'] = diff([('her_shape', 1), ('her', -1)], 'validation_auc_percent')
    out['C_minus_A'] = diff([('her_smooth', 1), ('her', -1)], 'validation_auc_percent')
    out['D_minus_B'] = diff([('her_both', 1), ('her_shape', -1)], 'validation_auc_percent')
    out['D_minus_C'] = diff([('her_both', 1), ('her_smooth', -1)], 'validation_auc_percent')
    out['A_minus_S'] = diff([('her', 1), ('sac', -1)], 'validation_auc_percent')
    out['shaping_main_effect'] = diff(
        [('her_shape', .5), ('her', -.5), ('her_both', .5), ('her_smooth', -.5)], 'validation_auc_percent')
    out['smoothing_main_effect'] = diff(
        [('her_smooth', .5), ('her', -.5), ('her_both', .5), ('her_shape', -.5)], 'validation_auc_percent')
    out['interaction_D_minus_B_minus_C_plus_A'] = diff(
        [('her_both', 1), ('her_shape', -1), ('her_smooth', -1), ('her', 1)], 'validation_auc_percent')
    primary = out['primary_D_minus_A']
    out['primary_criteria'] = {'mean_at_least_10pp': primary['mean_pp'] >= 10,
                               'at_least_four_positive': sum(x > 0 for x in primary['per_seed_percentage_points']) >= 4,
                               'ci_lower_above_zero': primary['t95_interval_pp'][0] > 0}
    out['test_pairwise'] = {}
    for ref in ('her', 'her_shape', 'her_smooth'):
        values = [row['arms']['her_both']['test']['success_percent'] - row['arms'][ref]['test']['success_percent']
                  for row in out['runs']]
        out['test_pairwise']['D_minus_' + ref] = statistics(values)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='completed result directory')
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
