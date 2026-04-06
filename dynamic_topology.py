#!/usr/bin/env python3
"""
Dynamic topology experiment: evolving the evolution strategy.

Tests whether a topology schedule (sparse early → dense late) outperforms
any fixed topology on deceptive landscapes.

Motivation: on Goldberg traps, diversity ordering is preserved
(none > ring > star > random > FC) but fitness ordering INVERTS
(FC > random > star > ring > none). Sparse topologies discover building
blocks; dense topologies assemble them. No fixed topology can do both.

A dynamic schedule can: ring for the first N generations (discovery phase),
then switch to FC (assembly phase).

Implements several schedules and compares them against fixed topologies.

Usage:
    python dynamic_topology.py                # All schedules, k=3,5,7
    python dynamic_topology.py --k 5          # Just k=5
    python dynamic_topology.py --seeds 10     # Fewer seeds
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy import stats

# Import GA infrastructure
CAT_EVO_EXPERIMENTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'categorical-evolution', 'experiments'
)
sys.path.insert(0, os.path.abspath(CAT_EVO_EXPERIMENTS))

from onemax_stats import (
    GAConfig, tournament_select, one_point_crossover, point_mutate,
    split_population, merge_populations,
    ring_migrate, star_migrate, fully_connected_migrate, random_migrate, no_migrate,
    hamming_diversity, population_divergence,
)

from trap_domain import make_trap_evaluate, random_trap_population


# ---------------------------------------------------------------------------
# Migration dispatch (local, independent of GAConfig.topology)
# ---------------------------------------------------------------------------

MIGRATE_FNS = {
    'none': no_migrate,
    'ring': ring_migrate,
    'star': star_migrate,
    'random': random_migrate,
    'fully_connected': fully_connected_migrate,
}


# ---------------------------------------------------------------------------
# Topology schedules
# ---------------------------------------------------------------------------

def make_fixed_schedule(topology: str) -> Callable[[int, int], str]:
    """Return a schedule that always uses the same topology."""
    def schedule(gen: int, max_gen: int) -> str:
        return topology
    return schedule


def make_step_schedule(sparse: str, dense: str, switch_frac: float) -> Callable[[int, int], str]:
    """Switch from sparse to dense at switch_frac * max_gen."""
    def schedule(gen: int, max_gen: int) -> str:
        if gen < switch_frac * max_gen:
            return sparse
        return dense
    return schedule


def make_linear_schedule(stages: List[Tuple[float, str]]) -> Callable[[int, int], str]:
    """Multi-stage schedule. stages = [(frac, topology), ...] sorted by frac.

    E.g., [(0.0, 'none'), (0.33, 'ring'), (0.67, 'fully_connected')]
    means: none for first 33%, ring for middle 33%, FC for final 33%.
    """
    def schedule(gen: int, max_gen: int) -> str:
        frac = gen / max_gen
        current = stages[0][1]
        for threshold, topo in stages:
            if frac >= threshold:
                current = topo
        return current
    return schedule


# All schedules to test
SCHEDULES = {
    # Fixed baselines
    'fixed_none': make_fixed_schedule('none'),
    'fixed_ring': make_fixed_schedule('ring'),
    'fixed_star': make_fixed_schedule('star'),
    'fixed_random': make_fixed_schedule('random'),
    'fixed_fc': make_fixed_schedule('fully_connected'),

    # Step schedules: sparse → dense
    'none→fc@50%': make_step_schedule('none', 'fully_connected', 0.5),
    'ring→fc@50%': make_step_schedule('ring', 'fully_connected', 0.5),
    'ring→fc@25%': make_step_schedule('ring', 'fully_connected', 0.25),
    'ring→fc@75%': make_step_schedule('ring', 'fully_connected', 0.75),

    # Reverse: dense → sparse (control — should perform worse)
    'fc→ring@50%': make_step_schedule('fully_connected', 'ring', 0.5),

    # Multi-stage: gradual densification
    'none→ring→fc': make_linear_schedule([
        (0.0, 'none'), (0.33, 'ring'), (0.67, 'fully_connected')
    ]),
    'none→ring→star→fc': make_linear_schedule([
        (0.0, 'none'), (0.25, 'ring'), (0.50, 'star'), (0.75, 'fully_connected')
    ]),
}


# ---------------------------------------------------------------------------
# Core: run one experiment with a topology schedule
# ---------------------------------------------------------------------------

def run_scheduled_experiment(
    seed: int,
    config: GAConfig,
    schedule_fn: Callable[[int, int], str],
    evaluate_fn: Callable,
    init_fn: Callable = None,
    crossover_fn: Callable = None,
    mutate_fn: Callable = None,
    diversity_fn: Callable = None,
) -> List[dict]:
    """Run island-model GA with a dynamic topology schedule.

    Like run_experiment_e_single but the topology can change each
    migration event based on schedule_fn(gen, max_gen).
    """
    if init_fn is None:
        init_fn = random_trap_population
    if crossover_fn is None:
        crossover_fn = one_point_crossover
    if mutate_fn is None:
        mutate_fn = point_mutate
    if diversity_fn is None:
        diversity_fn = hamming_diversity

    rng = np.random.default_rng(seed)
    init_pop = init_fn(rng, config.population_size, config.genome_length)
    islands = split_population(init_pop, config.num_islands)

    rows = []
    for gen in range(config.max_generations):
        # Evolve each island
        new_islands = []
        for isl in islands:
            fitnesses = evaluate_fn(isl)
            selected = tournament_select(rng, isl, fitnesses, config.tournament_size)
            crossed = crossover_fn(rng, selected, config.crossover_rate)
            mutated = mutate_fn(rng, crossed, config.mutation_rate)
            new_islands.append(mutated)
        islands = new_islands

        # Dynamic migration: topology depends on generation
        if gen > 0 and gen % config.migration_freq == 0:
            topo = schedule_fn(gen, config.max_generations)
            migrate_fn = MIGRATE_FNS[topo]
            islands = migrate_fn(rng, islands, config.migration_rate)
        else:
            topo = schedule_fn(gen, config.max_generations)

        # Metrics
        all_pop = merge_populations(islands)
        all_fit = evaluate_fn(all_pop)
        ham_div = diversity_fn(all_pop)
        best_fit = float(np.max(all_fit))
        mean_fit = float(np.mean(all_fit))

        rows.append({
            'seed': seed,
            'generation': gen,
            'topology_at_gen': topo,
            'hamming_diversity': ham_div,
            'best_fitness': best_fit,
            'mean_fitness': mean_fit,
        })

    return rows


# ---------------------------------------------------------------------------
# Run all schedules for one trap size
# ---------------------------------------------------------------------------

def run_all_schedules(k: int, num_traps: int, num_seeds: int = 30):
    """Run all topology schedules on trap-k and return results."""
    genome_length = k * num_traps
    config = GAConfig(
        population_size=100,
        genome_length=genome_length,
        num_islands=5,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=1.0 / genome_length,
        max_generations=200,
        migration_freq=5,
        migration_rate=0.05,
    )

    evaluate_fn = make_trap_evaluate(k, num_traps)
    seeds = list(range(num_seeds))

    all_results = []
    total = len(SCHEDULES) * num_seeds
    done = 0

    for sched_name, sched_fn in SCHEDULES.items():
        for seed in seeds:
            done += 1
            rows = run_scheduled_experiment(
                seed=seed,
                config=config,
                schedule_fn=sched_fn,
                evaluate_fn=evaluate_fn,
            )
            for row in rows:
                row['schedule'] = sched_name
                row['trap_k'] = k
            all_results.extend(rows)

        print(f'  {done}/{total} runs complete ({sched_name})', flush=True)

    return all_results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(results: List[dict], k: int):
    """Print summary table comparing schedules."""
    import pandas as pd

    df = pd.DataFrame(results)
    final = df[df.generation == df.generation.max()]

    attractor = (k - 1) / k

    print(f'\n{"="*70}')
    print(f'  Trap-{k}: schedule comparison (attractor={attractor:.4f}, optimum=1.0)')
    print(f'{"="*70}')
    print(f'{"Schedule":25s}  {"Best Fit":>10s}  {"Mean Fit":>10s}  {"Diversity":>10s}  {"Solved":>8s}')
    print(f'{"-"*70}')

    schedule_stats = []
    for sched in SCHEDULES:
        sched_final = final[final.schedule == sched]
        best = sched_final.best_fitness.mean()
        mean = sched_final.mean_fitness.mean()
        div = sched_final.hamming_diversity.mean()
        solved = (sched_final.best_fitness >= 0.999).sum()
        total = len(sched_final)
        schedule_stats.append((sched, best, mean, div, solved, total))
        print(f'{sched:25s}  {best:10.4f}  {mean:10.4f}  {div:10.4f}  {solved:3d}/{total}')

    # Rank by best fitness
    ranked = sorted(schedule_stats, key=lambda x: -x[1])
    print(f'\nRanking by solution quality:')
    for i, (name, best, _, _, solved, total) in enumerate(ranked):
        marker = ' ★' if '→' in name else ''
        print(f'  {i+1}. {name:25s}  best={best:.4f}  solved={solved}/{total}{marker}')

    # Statistical test: best dynamic vs best fixed
    best_dynamic = max(
        [(s, b) for s, b, _, _, _, _ in schedule_stats if '→' in s],
        key=lambda x: x[1]
    )
    best_fixed = max(
        [(s, b) for s, b, _, _, _, _ in schedule_stats if '→' not in s],
        key=lambda x: x[1]
    )

    dyn_name = best_dynamic[0]
    fix_name = best_fixed[0]

    dyn_fits = final[final.schedule == dyn_name].best_fitness.values
    fix_fits = final[final.schedule == fix_name].best_fitness.values

    t_stat, p_val = stats.ttest_ind(dyn_fits, fix_fits)
    dyn_mean = dyn_fits.mean()
    fix_mean = fix_fits.mean()
    print(f'\nBest dynamic: {dyn_name} (mean={dyn_mean:.4f})')
    print(f'Best fixed:   {fix_name} (mean={fix_mean:.4f})')
    print(f'Welch t-test: t={t_stat:.3f}, p={p_val:.6f}')
    if dyn_mean > fix_mean and p_val < 0.05:
        print(f'→ Dynamic schedule SIGNIFICANTLY outperforms best fixed topology!')
    elif dyn_mean > fix_mean:
        print(f'→ Dynamic schedule outperforms but not significantly (p={p_val:.4f})')
    else:
        print(f'→ Fixed topology wins (dynamic scheduling did not help)')

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Dynamic topology experiments')
    parser.add_argument('--k', type=int, choices=[3, 5, 7],
                        help='Run only this trap size')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of seeds (default: 30)')
    args = parser.parse_args()

    trap_sizes = [args.k] if args.k else [3, 5, 7]

    print('Dynamic Topology Experiment')
    print(f'  Trap sizes: {trap_sizes}')
    print(f'  Seeds: {args.seeds}')
    print(f'  Schedules: {len(SCHEDULES)}')
    print(f'  Total runs: {len(SCHEDULES) * args.seeds * len(trap_sizes)}')

    for k in trap_sizes:
        num_traps = 10
        print(f'\n--- Running Trap-{k} ({k*num_traps}-bit genome, {num_traps} traps) ---')

        t0 = time.time()
        results = run_all_schedules(k, num_traps, num_seeds=args.seeds)
        elapsed = time.time() - t0
        print(f'  Completed in {elapsed:.1f}s')

        # Save CSV
        csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f'dynamic_trap{k}.csv'
        )
        keys = results[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f'  Saved: {csv_path}')

        analyze_results(results, k)


if __name__ == '__main__':
    main()
