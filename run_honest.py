#!/usr/bin/env python3
"""
Evolve migration graphs on honest (non-deceptive) domains.

Tests whether the K₄-e + pendant topology also emerges on honest
landscapes, or whether deception is required to produce it.

Domains: knapsack (50-bit), maze (60-bit), graph coloring (40-bit),
sorting network (28-bit), No Thanks (13-bit).

Usage:
    python run_honest.py                       # All domains, n=5
    python run_honest.py --domain knapsack     # Just knapsack
    python run_honest.py --islands 8           # n=8 islands
"""

import argparse
import os
import sys
import time

import numpy as np

CAT_EVO_EXPERIMENTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'categorical-evolution', 'experiments'
)
sys.path.insert(0, os.path.abspath(CAT_EVO_EXPERIMENTS))

from onemax_stats import (
    GAConfig, one_point_crossover, point_mutate,
    hamming_diversity, population_divergence,
)

from knapsack_domain import (
    evaluate_knapsack, random_knapsack_population, KNAPSACK_GENOME_LENGTH,
)
from maze_domain import (
    evaluate_maze, random_maze_population, MAZE_GENOME_LENGTH,
)
from graph_coloring_domain import (
    evaluate_graph_coloring, random_graph_coloring_population,
    GRAPH_COLORING_GENOME_LENGTH,
)
from sorting_network_domain import (
    evaluate_sorting_network, random_sorting_network_population,
    SORTING_NETWORK_GENOME_LENGTH,
)

import evolve_graph
from evolve_graph import (
    set_num_islands, OuterConfig, evolve_graphs,
    compare_against_baselines, compute_lambda2, graph_label,
    inner_tournament_select, inner_crossover, inner_mutate,
    split_population, merge_populations, adjacency_migrate,
)


# ---------------------------------------------------------------------------
# Domain configs
# ---------------------------------------------------------------------------

HONEST_DOMAINS = {
    'knapsack': {
        'evaluate_fn': evaluate_knapsack,
        'init_fn': random_knapsack_population,
        'genome_length': KNAPSACK_GENOME_LENGTH,  # 50
        'label': 'Knapsack (50-bit, rugged)',
    },
    'maze': {
        'evaluate_fn': evaluate_maze,
        'init_fn': random_maze_population,
        'genome_length': MAZE_GENOME_LENGTH,  # 60
        'label': 'Maze (60-bit, rugged)',
    },
    'graph_coloring': {
        'evaluate_fn': evaluate_graph_coloring,
        'init_fn': random_graph_coloring_population,
        'genome_length': GRAPH_COLORING_GENOME_LENGTH,  # 40
        'label': 'Graph Coloring (40-bit, constraint)',
    },
    'sorting_network': {
        'evaluate_fn': evaluate_sorting_network,
        'init_fn': random_sorting_network_population,
        'genome_length': SORTING_NETWORK_GENOME_LENGTH,  # 28
        'label': 'Sorting Network (28-bit)',
    },
}


# ---------------------------------------------------------------------------
# Run graph evolution for one domain
# ---------------------------------------------------------------------------

def run_domain(name, config, n_islands=5, seed=42):
    """Evolve migration graph for an honest domain."""
    set_num_islands(n_islands)
    from evolve_graph import NUM_EDGES

    evaluate_fn = config['evaluate_fn']
    init_fn = config['init_fn']
    genome_length = config['genome_length']

    pop_per_island = 20
    inner_cfg = GAConfig(
        population_size=pop_per_island * n_islands,
        genome_length=genome_length,
        num_islands=n_islands,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=1.0 / genome_length,
        max_generations=200,
        migration_freq=5,
        migration_rate=0.05,
    )

    outer_pop = 20
    outer_gens = 30
    if NUM_EDGES > 15:
        outer_pop = 30
        outer_gens = 50

    outer_cfg = OuterConfig(
        pop_size=outer_pop,
        max_generations=outer_gens,
        mutation_rate=1.0 / NUM_EDGES,
        inner_seeds_per_eval=5,
    )

    # Patch inner_ga_fitness
    _original = evolve_graph.inner_ga_fitness

    def custom_inner_ga_fitness(adj, k, num_traps, inner_seeds, inner_config):
        fitnesses = []
        for s in inner_seeds:
            rng = np.random.default_rng(s)
            pop = init_fn(rng, inner_config.population_size,
                          inner_config.genome_length)
            islands = split_population(pop, inner_config.num_islands)

            for gen in range(inner_config.max_generations):
                new_islands = []
                for isl in islands:
                    fit = evaluate_fn(isl)
                    selected = inner_tournament_select(rng, isl, fit,
                                                       inner_config.tournament_size)
                    crossed = inner_crossover(rng, selected,
                                              inner_config.crossover_rate)
                    mutated = inner_mutate(rng, crossed,
                                           inner_config.mutation_rate)
                    new_islands.append(mutated)
                islands = new_islands

                if gen > 0 and gen % inner_config.migration_freq == 0:
                    islands = adjacency_migrate(rng, islands, adj,
                                                inner_config.migration_rate)

            all_pop = merge_populations(islands)
            all_fit = evaluate_fn(all_pop)
            fitnesses.append(float(np.max(all_fit)))
        return float(np.mean(fitnesses))

    evolve_graph.inner_ga_fitness = custom_inner_ga_fitness

    print(f'\n{"="*60}')
    print(f'  {config["label"]}')
    print(f'  {n_islands} islands, {NUM_EDGES}-bit graph genome')
    print(f'  Outer: pop={outer_pop}, gens={outer_gens}')
    print(f'{"="*60}')

    t0 = time.time()
    best_graph, best_fitness, history = evolve_graphs(
        k=0, num_traps=0, outer_cfg=outer_cfg, inner_cfg=inner_cfg, seed=seed)
    elapsed = time.time() - t0

    print(f'\n  Completed in {elapsed:.1f}s')
    print(f'  Best: {graph_label(best_graph)} '
          f'({int(best_graph.sum())} edges, λ₂={compute_lambda2(best_graph):.3f})')

    compare_against_baselines(best_graph, 0, 0, inner_cfg, num_seeds=30)

    evolve_graph.inner_ga_fitness = _original

    return best_graph


def main():
    parser = argparse.ArgumentParser(
        description='Evolve graphs on honest domains')
    parser.add_argument('--domain', choices=list(HONEST_DOMAINS.keys()),
                        help='Run only this domain')
    parser.add_argument('--islands', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    domains = [args.domain] if args.domain else list(HONEST_DOMAINS.keys())

    print(f'Evolving Graphs on Honest Domains')
    print(f'  Islands: {args.islands}')
    print(f'  Domains: {", ".join(domains)}')

    results = {}
    for name in domains:
        best = run_domain(name, HONEST_DOMAINS[name],
                          n_islands=args.islands, seed=args.seed)
        results[name] = best

    # Summary
    print(f'\n{"="*60}')
    print(f'  SUMMARY: Honest domains, n={args.islands}')
    print(f'{"="*60}')
    print(f'  {"Domain":20s}  {"Edges":>5s}  {"λ₂":>6s}  {"Degrees"}')
    print(f'  {"-"*55}')
    for name, adj in results.items():
        from evolve_graph import adj_vector_to_matrix
        mat = adj_vector_to_matrix(adj)
        degrees = sorted(mat.sum(axis=1).astype(int))
        l2 = compute_lambda2(adj)
        print(f'  {name:20s}  {int(adj.sum()):5d}  {l2:6.3f}  {degrees}')

    # Check isomorphism with K4-e+pendant
    if args.islands == 5:
        from itertools import permutations
        k4e = np.zeros(10, dtype=np.int8)
        for i, j in [(0,1),(0,2),(0,3),(1,3),(2,3),(2,4)]:
            idx = i * 5 - i * (i + 1) // 2 + (j - i - 1)
            k4e[idx] = 1
        k4e_mat = adj_vector_to_matrix(k4e)

        print(f'\n  Isomorphic to K₄-e + pendant (the Nick graph)?')
        for name, adj in results.items():
            mat = adj_vector_to_matrix(adj)
            is_iso = False
            for p in permutations(range(5)):
                P = np.zeros((5,5), dtype=int)
                for i in range(5): P[i, p[i]] = 1
                if np.array_equal(P @ k4e_mat @ P.T, mat):
                    is_iso = True
                    break
            print(f'  {name:20s}: {"YES" if is_iso else "NO"}')


if __name__ == '__main__':
    main()
