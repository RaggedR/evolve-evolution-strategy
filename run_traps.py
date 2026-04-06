#!/usr/bin/env python3
"""
Topology sweep on Goldberg trap functions.

Runs experiment_e (the standard topology sweep from the ACT paper)
on concatenated k-trap landscapes for k=3, 5, 7. The question:
does the topology ordering (none > ring > star > random > FC) hold
on deceptive landscapes?

Uses the GA infrastructure from categorical-evolution/experiments/.
Produces CSV files compatible with multi_domain_analysis.py.

Usage:
    python run_traps.py                    # All three trap sizes
    python run_traps.py --k 5              # Just k=5
    python run_traps.py --seeds 10         # Fewer seeds (faster)
    python run_traps.py --resume           # Resume interrupted run
"""

import argparse
import os
import sys
import time

# Add categorical-evolution experiments to path
CAT_EVO_EXPERIMENTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'categorical-evolution', 'experiments'
)
sys.path.insert(0, os.path.abspath(CAT_EVO_EXPERIMENTS))

from onemax_stats import (
    GAConfig, run_experiment_e, random_population,
    one_point_crossover, point_mutate,
    hamming_diversity, population_divergence,
)

from trap_domain import make_trap_evaluate, random_trap_population


# ---------------------------------------------------------------------------
# Configuration for each trap size
# ---------------------------------------------------------------------------

TRAP_CONFIGS = {
    3: {
        'k': 3,
        'num_traps': 10,   # 30-bit genome
        'label': 'Trap-3',
        'description': 'k=3: mildly deceptive, 33% gap',
    },
    5: {
        'k': 5,
        'num_traps': 10,   # 50-bit genome
        'label': 'Trap-5',
        'description': 'k=5: moderately deceptive, 20% gap',
    },
    7: {
        'k': 7,
        'num_traps': 10,   # 70-bit genome
        'label': 'Trap-7',
        'description': 'k=7: strongly deceptive, 14.3% gap',
    },
}


def run_trap_experiment(k: int, num_seeds: int = 30, resume: bool = False):
    """Run the topology sweep for a single trap size."""
    tc = TRAP_CONFIGS[k]
    genome_length = tc['k'] * tc['num_traps']

    print(f"\n{'='*60}")
    print(f"  {tc['label']}: {tc['description']}")
    print(f"  genome_length={genome_length}, num_traps={tc['num_traps']}")
    print(f"  seeds={num_seeds}")
    print(f"{'='*60}\n")

    # GA config — match the ACT paper's standard parameters
    # but with longer runs since deceptive landscapes need more generations
    config = GAConfig(
        population_size=100,       # 5 islands × 20 each
        genome_length=genome_length,
        num_islands=5,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=1.0 / genome_length,  # standard 1/L
        max_generations=200,       # longer than OneMax (40) — deception needs time
        migration_freq=5,
        migration_rate=0.05,
    )

    seeds = list(range(num_seeds))

    # Output CSV
    output_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(output_dir, f'experiment_e_trap{k}.csv')

    evaluate_fn = make_trap_evaluate(tc['k'], tc['num_traps'])

    t0 = time.time()

    results = run_experiment_e(
        seeds=seeds,
        config=config,
        evaluate_fn=evaluate_fn,
        init_fn=random_trap_population,
        # Use default binary crossover and mutation (same genome representation)
        crossover_fn=one_point_crossover,
        mutate_fn=point_mutate,
        diversity_fn=hamming_diversity,
        divergence_fn=population_divergence,
        incremental_csv=csv_path,
        resume=resume,
    )

    elapsed = time.time() - t0
    print(f"\n  Completed {tc['label']}: {len(results)} rows in {elapsed:.1f}s")
    print(f"  Output: {csv_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Topology sweep on Goldberg trap functions')
    parser.add_argument('--k', type=int, choices=[3, 5, 7],
                        help='Run only this trap size (default: all three)')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of seeds per (topology, trap) combo (default: 30)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume interrupted run (skip completed pairs)')
    args = parser.parse_args()

    trap_sizes = [args.k] if args.k else [3, 5, 7]

    print("Goldberg Trap Topology Sweep")
    print(f"  Trap sizes: {trap_sizes}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Topologies: none, ring, star, random, fully_connected")

    t_total = time.time()

    for k in trap_sizes:
        run_trap_experiment(k, num_seeds=args.seeds, resume=args.resume)

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  All done in {elapsed_total:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
