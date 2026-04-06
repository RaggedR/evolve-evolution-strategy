#!/usr/bin/env python3
"""
Topology sweep and graph evolution on HIFF.

Tests whether the findings from Goldberg traps (fitness inversion,
evolved asymmetric graphs) extend to hierarchical deception.

Usage:
    python run_hiff.py                # Topology sweep + evolve graph
    python run_hiff.py --sweep-only   # Just the topology sweep
    python run_hiff.py --evolve-only  # Just graph evolution
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy import stats

CAT_EVO_EXPERIMENTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'categorical-evolution', 'experiments'
)
sys.path.insert(0, os.path.abspath(CAT_EVO_EXPERIMENTS))

from onemax_stats import (
    GAConfig, run_experiment_e,
    one_point_crossover, point_mutate,
    hamming_diversity, population_divergence,
)

from hiff_domain import make_hiff_evaluate, random_hiff_population

# Import graph evolution machinery
from evolve_graph import (
    set_num_islands, NUM_EDGES, evolve_graphs, compare_against_baselines,
    OuterConfig, compute_lambda2, graph_label,
)


def run_topology_sweep(genome_length: int, num_seeds: int = 30):
    """Run standard topology sweep on HIFF."""
    config = GAConfig(
        population_size=100,
        genome_length=genome_length,
        num_islands=5,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=1.0 / genome_length,
        max_generations=300,  # HIFF is harder, needs more gens
        migration_freq=5,
        migration_rate=0.05,
    )

    seeds = list(range(num_seeds))
    evaluate_fn = make_hiff_evaluate()

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'experiment_e_hiff.csv'
    )

    print(f'\n  HIFF Topology Sweep')
    print(f'  genome_length={genome_length} (L={int(np.log2(genome_length))})')
    print(f'  seeds={num_seeds}, 300 generations')

    t0 = time.time()
    results = run_experiment_e(
        seeds=seeds,
        config=config,
        evaluate_fn=evaluate_fn,
        init_fn=random_hiff_population,
        crossover_fn=one_point_crossover,
        mutate_fn=point_mutate,
        diversity_fn=hamming_diversity,
        divergence_fn=population_divergence,
        incremental_csv=csv_path,
        resume=False,
    )
    elapsed = time.time() - t0
    print(f'  Completed in {elapsed:.1f}s')

    # Analyze
    import pandas as pd
    df = pd.read_csv(csv_path)
    final = df[df.generation == df.generation.max()]

    topo_order = ['none', 'ring', 'star', 'random', 'fully_connected']
    print(f'\n  {"Topology":20s}  {"Diversity":>10s}  {"Best Fit":>10s}')
    print(f'  {"-"*45}')

    for topo in topo_order:
        rows = final[final.topology == topo]
        div = rows.hamming_diversity.mean()
        fit = rows.best_fitness.mean()
        solved = (rows.best_fitness >= 0.999).sum()
        print(f'  {topo:20s}  {div:10.4f}  {fit:10.4f}  ({solved}/{len(rows)} solved)')

    # Fitness ordering
    fit_ranking = sorted(
        [(t, final[final.topology == t].best_fitness.mean()) for t in topo_order],
        key=lambda x: -x[1]
    )
    print(f'\n  Fitness ordering: {" > ".join(f"{t}({f:.4f})" for t, f in fit_ranking)}')

    # Diversity ordering
    div_ranking = sorted(
        [(t, final[final.topology == t].hamming_diversity.mean()) for t in topo_order],
        key=lambda x: -x[1]
    )
    print(f'  Diversity ordering: {" > ".join(f"{t}({d:.4f})" for t, d in div_ranking)}')

    return results


def run_graph_evolution(genome_length: int, n_islands: int = 5):
    """Evolve migration graphs for HIFF."""
    set_num_islands(n_islands)
    from evolve_graph import NUM_EDGES

    pop_per_island = 20
    inner_cfg = GAConfig(
        population_size=pop_per_island * n_islands,
        genome_length=genome_length,
        num_islands=n_islands,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=1.0 / genome_length,
        max_generations=300,
        migration_freq=5,
        migration_rate=0.05,
    )

    outer_cfg = OuterConfig(
        pop_size=20,
        max_generations=30,
        mutation_rate=1.0 / NUM_EDGES,
        inner_seeds_per_eval=5,
    )

    # Monkey-patch the evaluate function used by inner_ga_fitness
    import hiff_domain
    import evolve_graph
    original_make = evolve_graph.__builtins__ if hasattr(evolve_graph, '__builtins__') else {}

    # We need to pass HIFF evaluate to the inner GA
    # The cleanest way: temporarily replace the trap imports in evolve_graph
    # Actually, let's just call evolve_graphs with the right inner config
    # and manually override the evaluate function

    # Patch inner_ga_fitness to use HIFF
    hiff_eval = make_hiff_evaluate()
    _original_inner_ga_fitness = evolve_graph.inner_ga_fitness

    def hiff_inner_ga_fitness(adj, k, num_traps, inner_seeds, inner_config):
        """Inner GA fitness using HIFF instead of traps."""
        from evolve_graph import (
            inner_tournament_select, inner_crossover, inner_mutate,
            split_population, merge_populations, adjacency_migrate,
        )
        fitnesses = []
        for seed in inner_seeds:
            rng = np.random.default_rng(seed)
            init_pop = random_hiff_population(rng, inner_config.population_size,
                                              inner_config.genome_length)
            islands = split_population(init_pop, inner_config.num_islands)

            for gen in range(inner_config.max_generations):
                new_islands = []
                for isl in islands:
                    fit = hiff_eval(isl)
                    selected = inner_tournament_select(rng, isl, fit,
                                                       inner_config.tournament_size)
                    crossed = inner_crossover(rng, selected, inner_config.crossover_rate)
                    mutated = inner_mutate(rng, crossed, inner_config.mutation_rate)
                    new_islands.append(mutated)
                islands = new_islands

                if gen > 0 and gen % inner_config.migration_freq == 0:
                    islands = adjacency_migrate(rng, islands, adj,
                                                inner_config.migration_rate)

            all_pop = merge_populations(islands)
            all_fit = hiff_eval(all_pop)
            fitnesses.append(float(np.max(all_fit)))

        return float(np.mean(fitnesses))

    # Patch
    evolve_graph.inner_ga_fitness = hiff_inner_ga_fitness

    print(f'\n  HIFF Graph Evolution')
    print(f'  genome_length={genome_length}, {n_islands} islands')

    t0 = time.time()
    best_graph, best_fitness, history = evolve_graphs(
        k=0, num_traps=0,  # unused due to patch
        outer_cfg=outer_cfg, inner_cfg=inner_cfg, seed=42)
    elapsed = time.time() - t0

    print(f'\n  Evolution completed in {elapsed:.1f}s')
    print(f'  Best: {graph_label(best_graph)} '
          f'({int(best_graph.sum())} edges, λ₂={compute_lambda2(best_graph):.3f})')

    # Fair comparison
    # Patch compare_against_baselines too
    compare_against_baselines(best_graph, 0, 0, inner_cfg, num_seeds=30)

    # Restore
    evolve_graph.inner_ga_fitness = _original_inner_ga_fitness


def main():
    parser = argparse.ArgumentParser(description='HIFF topology experiments')
    parser.add_argument('--L', type=int, default=6,
                        help='HIFF level (genome = 2^L bits, default: 6 → 64 bits)')
    parser.add_argument('--sweep-only', action='store_true')
    parser.add_argument('--evolve-only', action='store_true')
    parser.add_argument('--seeds', type=int, default=30)
    args = parser.parse_args()

    genome_length = 2 ** args.L

    if not args.evolve_only:
        run_topology_sweep(genome_length, num_seeds=args.seeds)

    if not args.sweep_only:
        run_graph_evolution(genome_length, n_islands=5)


if __name__ == '__main__':
    main()
