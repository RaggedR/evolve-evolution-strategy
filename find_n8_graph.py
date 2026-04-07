#!/usr/bin/env python3
"""
Find the n=8 Nick graph by running all 5 deceptive domains
with multiple outer seeds.

At n=5, 3/5 domains converged to K₄-e + pendant.
At n=8, do we get a consensus graph?
"""

import os
import sys
import time

import numpy as np
from itertools import permutations

CAT_EVO_EXPERIMENTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'categorical-evolution', 'experiments'
)
sys.path.insert(0, os.path.abspath(CAT_EVO_EXPERIMENTS))

from onemax_stats import (
    GAConfig, tournament_select as inner_tournament_select,
    one_point_crossover as inner_crossover,
    point_mutate as inner_mutate,
    split_population, merge_populations,
)

import evolve_graph
from evolve_graph import (
    set_num_islands, OuterConfig, evolve_graphs,
    compute_lambda2, graph_label, adj_vector_to_matrix,
    adjacency_migrate,
)

from trap_domain import make_trap_evaluate, random_trap_population
from hiff_domain import make_hiff_evaluate, random_hiff_population
from mmdp_domain import make_mmdp_evaluate, random_mmdp_population, MMDP_K
from overlap_trap_domain import (
    make_overlap_trap_evaluate, random_overlap_population,
    genome_length_for_overlap,
)


# ---------------------------------------------------------------------------
# Domain configs
# ---------------------------------------------------------------------------

DOMAINS = {
    'trap5': {
        'evaluate_fn': make_trap_evaluate(5, 10),
        'init_fn': random_trap_population,
        'genome_length': 50,
    },
    'trap7': {
        'evaluate_fn': make_trap_evaluate(7, 10),
        'init_fn': random_trap_population,
        'genome_length': 70,
    },
    'hiff': {
        'evaluate_fn': make_hiff_evaluate(),
        'init_fn': random_hiff_population,
        'genome_length': 64,
    },
    'mmdp': {
        'evaluate_fn': make_mmdp_evaluate(10),
        'init_fn': random_mmdp_population,
        'genome_length': MMDP_K * 10,
    },
    'overlap': {
        'evaluate_fn': make_overlap_trap_evaluate(5, 2),
        'init_fn': random_overlap_population,
        'genome_length': genome_length_for_overlap(5, 2, 10),
    },
}


def run_one(domain_name, outer_seed, n_islands=8):
    """Run one graph evolution and return the best graph."""
    cfg = DOMAINS[domain_name]
    set_num_islands(n_islands)
    from evolve_graph import NUM_EDGES

    evaluate_fn = cfg['evaluate_fn']
    init_fn = cfg['init_fn']

    inner_cfg = GAConfig(
        population_size=20 * n_islands,
        genome_length=cfg['genome_length'],
        num_islands=n_islands,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=1.0 / cfg['genome_length'],
        max_generations=200,
        migration_freq=5,
        migration_rate=0.05,
    )

    outer_cfg = OuterConfig(
        pop_size=30,
        max_generations=50,
        mutation_rate=1.0 / NUM_EDGES,
        inner_seeds_per_eval=5,
        elite_count=3,
    )

    # Patch inner_ga_fitness
    _original = evolve_graph.inner_ga_fitness

    def custom_fitness(adj, k, num_traps, inner_seeds, inner_config):
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
                    sel = inner_tournament_select(rng, isl, fit,
                                                  inner_config.tournament_size)
                    cx = inner_crossover(rng, sel, inner_config.crossover_rate)
                    mut = inner_mutate(rng, cx, inner_config.mutation_rate)
                    new_islands.append(mut)
                islands = new_islands
                if gen > 0 and gen % inner_config.migration_freq == 0:
                    islands = adjacency_migrate(rng, islands, adj,
                                                inner_config.migration_rate)
            all_pop = merge_populations(islands)
            all_fit = evaluate_fn(all_pop)
            fitnesses.append(float(np.max(all_fit)))
        return float(np.mean(fitnesses))

    evolve_graph.inner_ga_fitness = custom_fitness

    best_graph, best_fitness, _ = evolve_graphs(
        k=0, num_traps=0, outer_cfg=outer_cfg, inner_cfg=inner_cfg,
        seed=outer_seed)

    evolve_graph.inner_ga_fitness = _original
    return best_graph, best_fitness


def check_isomorphism(A, B, n=8):
    """Check if two adjacency matrices are isomorphic.

    For n=8, brute force over 8! = 40320 is feasible.
    """
    for p in permutations(range(n)):
        P = np.zeros((n, n), dtype=int)
        for i in range(n):
            P[i, p[i]] = 1
        if np.array_equal(P @ A @ P.T, B):
            return True
    return False


def main():
    n_islands = 8
    outer_seeds = [42, 137]  # two seeds per domain for robustness

    print(f'Finding the n={n_islands} Nick graph')
    print(f'  Domains: {list(DOMAINS.keys())}')
    print(f'  Outer seeds: {outer_seeds}')
    print(f'  Total runs: {len(DOMAINS) * len(outer_seeds)}')
    print()

    all_graphs = []  # (domain, seed, graph, fitness)

    for domain in DOMAINS:
        for seed in outer_seeds:
            print(f'\n--- {domain}, seed={seed} ---')
            t0 = time.time()
            graph, fitness = run_one(domain, seed, n_islands)
            elapsed = time.time() - t0

            mat = adj_vector_to_matrix(graph)
            degrees = sorted(mat.sum(axis=1).astype(int))
            l2 = compute_lambda2(graph)
            n_edges = int(graph.sum())

            print(f'  {elapsed:.0f}s | {n_edges}e | λ₂={l2:.3f} | '
                  f'deg={degrees} | fit={fitness:.4f}')

            all_graphs.append((domain, seed, graph, fitness))

    # --- Analysis ---
    print(f'\n{"="*70}')
    print(f'  ALL RESULTS (n={n_islands})')
    print(f'{"="*70}')
    print(f'  {"Domain":10s} {"Seed":>4s}  {"Edges":>5s}  {"λ₂":>6s}  '
          f'{"Fitness":>8s}  {"Degree sequence"}')
    print(f'  {"-"*65}')

    for domain, seed, graph, fitness in all_graphs:
        mat = adj_vector_to_matrix(graph)
        degrees = sorted(mat.sum(axis=1).astype(int))
        l2 = compute_lambda2(graph)
        n_edges = int(graph.sum())
        print(f'  {domain:10s} {seed:4d}  {n_edges:5d}  {l2:6.3f}  '
              f'{fitness:8.4f}  {degrees}')

    # --- Pairwise isomorphism ---
    print(f'\n  Pairwise isomorphism check:')
    n = len(all_graphs)
    matrices = [adj_vector_to_matrix(g) for _, _, g, _ in all_graphs]
    labels = [f'{d}({s})' for d, s, _, _ in all_graphs]

    iso_groups = []
    assigned = [False] * n
    for i in range(n):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if check_isomorphism(matrices[i], matrices[j], n_islands):
                group.append(j)
                assigned[j] = True
        iso_groups.append(group)

    for gi, group in enumerate(iso_groups):
        members = [labels[i] for i in group]
        mat = matrices[group[0]]
        degrees = sorted(mat.sum(axis=1).astype(int))
        l2 = compute_lambda2(all_graphs[group[0]][2])
        n_edges = int(all_graphs[group[0]][2].sum())
        print(f'  Group {gi+1}: {", ".join(members)}')
        print(f'           {n_edges}e, λ₂={l2:.3f}, deg={degrees}')

    print(f'\n  {len(iso_groups)} distinct graphs from {n} runs')
    if len(iso_groups) == 1:
        print(f'  ALL ISOMORPHIC — universal n={n_islands} Nick graph found!')
    else:
        # Find largest group
        largest = max(iso_groups, key=len)
        if len(largest) >= n // 2:
            members = [labels[i] for i in largest]
            print(f'  Largest group ({len(largest)}/{n} runs): {", ".join(members)}')
            print(f'  Candidate n={n_islands} Nick graph')


if __name__ == '__main__':
    main()
