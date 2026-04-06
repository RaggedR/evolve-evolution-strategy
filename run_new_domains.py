#!/usr/bin/env python3
"""
Topology sweep + graph evolution on MMDP and overlapping traps.

Usage:
    python run_new_domains.py                   # Both domains
    python run_new_domains.py --domain mmdp     # Just MMDP
    python run_new_domains.py --domain overlap   # Just overlapping traps
    python run_new_domains.py --sweep-only       # No graph evolution
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
    GAConfig, run_experiment_e,
    one_point_crossover, point_mutate,
    hamming_diversity, population_divergence,
)

from mmdp_domain import make_mmdp_evaluate, random_mmdp_population, MMDP_K
from overlap_trap_domain import (
    make_overlap_trap_evaluate, random_overlap_population,
    genome_length_for_overlap,
)

import evolve_graph
from evolve_graph import (
    set_num_islands, OuterConfig, evolve_graphs,
    compare_against_baselines, compute_lambda2, graph_label,
    inner_tournament_select, inner_crossover, inner_mutate,
    split_population, merge_populations, adjacency_migrate,
)


# ---------------------------------------------------------------------------
# Generic sweep + evolve pipeline
# ---------------------------------------------------------------------------

def run_sweep(name, genome_length, evaluate_fn, init_fn, num_seeds=30, max_gens=200):
    """Run topology sweep for a domain."""
    config = GAConfig(
        population_size=100,
        genome_length=genome_length,
        num_islands=5,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=1.0 / genome_length,
        max_generations=max_gens,
        migration_freq=5,
        migration_rate=0.05,
    )

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'experiment_e_{name}.csv'
    )

    print(f'\n  {name} Topology Sweep')
    print(f'  genome_length={genome_length}, seeds={num_seeds}, gens={max_gens}')

    t0 = time.time()
    results = run_experiment_e(
        seeds=list(range(num_seeds)),
        config=config,
        evaluate_fn=evaluate_fn,
        init_fn=init_fn,
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
    print(f'\n  {"Topology":20s}  {"Diversity":>10s}  {"Best Fit":>10s}  {"Solved":>8s}')
    print(f'  {"-"*55}')

    for topo in topo_order:
        rows = final[final.topology == topo]
        div = rows.hamming_diversity.mean()
        fit = rows.best_fitness.mean()
        solved = (rows.best_fitness >= 0.999).sum()
        print(f'  {topo:20s}  {div:10.4f}  {fit:10.4f}  {solved:3d}/{len(rows)}')

    fit_ranking = sorted(
        [(t, final[final.topology == t].best_fitness.mean()) for t in topo_order],
        key=lambda x: -x[1]
    )
    print(f'\n  Fitness ordering: {" > ".join(f"{t}({f:.4f})" for t, f in fit_ranking)}')

    div_ranking = sorted(
        [(t, final[final.topology == t].hamming_diversity.mean()) for t in topo_order],
        key=lambda x: -x[1]
    )
    print(f'  Diversity ordering: {" > ".join(f"{t}({d:.4f})" for t, d in div_ranking)}')

    return config


def run_evolve(name, genome_length, evaluate_fn, init_fn, max_gens=200):
    """Evolve migration graphs for a domain."""
    n_islands = 5
    set_num_islands(n_islands)
    from evolve_graph import NUM_EDGES

    inner_cfg = GAConfig(
        population_size=100,
        genome_length=genome_length,
        num_islands=n_islands,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=1.0 / genome_length,
        max_generations=max_gens,
        migration_freq=5,
        migration_rate=0.05,
    )

    outer_cfg = OuterConfig(
        pop_size=20,
        max_generations=30,
        mutation_rate=1.0 / NUM_EDGES,
        inner_seeds_per_eval=5,
    )

    # Patch inner_ga_fitness to use our evaluate function
    _original = evolve_graph.inner_ga_fitness

    def custom_inner_ga_fitness(adj, k, num_traps, inner_seeds, inner_config):
        fitnesses = []
        for seed in inner_seeds:
            rng = np.random.default_rng(seed)
            pop = init_fn(rng, inner_config.population_size, inner_config.genome_length)
            islands = split_population(pop, inner_config.num_islands)

            for gen in range(inner_config.max_generations):
                new_islands = []
                for isl in islands:
                    fit = evaluate_fn(isl)
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
            all_fit = evaluate_fn(all_pop)
            fitnesses.append(float(np.max(all_fit)))
        return float(np.mean(fitnesses))

    evolve_graph.inner_ga_fitness = custom_inner_ga_fitness

    print(f'\n  {name} Graph Evolution')
    print(f'  genome_length={genome_length}, {n_islands} islands')

    t0 = time.time()
    best_graph, best_fitness, history = evolve_graphs(
        k=0, num_traps=0, outer_cfg=outer_cfg, inner_cfg=inner_cfg, seed=42)
    elapsed = time.time() - t0

    print(f'\n  Evolution completed in {elapsed:.1f}s')
    print(f'  Best: {graph_label(best_graph)} '
          f'({int(best_graph.sum())} edges, λ₂={compute_lambda2(best_graph):.3f})')

    compare_against_baselines(best_graph, 0, 0, inner_cfg, num_seeds=30)

    evolve_graph.inner_ga_fitness = _original


# ---------------------------------------------------------------------------
# Domain configs
# ---------------------------------------------------------------------------

def run_mmdp(sweep_only=False, num_seeds=30):
    """Run MMDP experiments."""
    num_blocks = 10
    genome_length = MMDP_K * num_blocks  # 60 bits

    evaluate_fn = make_mmdp_evaluate(num_blocks)
    init_fn = random_mmdp_population

    print(f'\n{"="*60}')
    print(f'  MMDP: {num_blocks} blocks × {MMDP_K} bits = {genome_length}-bit genome')
    print(f'  Two optima per block, deceptive attractor at u=3')
    print(f'{"="*60}')

    run_sweep('mmdp', genome_length, evaluate_fn, init_fn,
              num_seeds=num_seeds, max_gens=200)

    if not sweep_only:
        run_evolve('mmdp', genome_length, evaluate_fn, init_fn, max_gens=200)


def run_overlap(sweep_only=False, num_seeds=30):
    """Run overlapping trap experiments."""
    k = 5
    overlap = 2
    num_blocks = 10
    genome_length = genome_length_for_overlap(k, overlap, num_blocks)  # 32 bits

    evaluate_fn = make_overlap_trap_evaluate(k, overlap)
    init_fn = random_overlap_population

    print(f'\n{"="*60}')
    print(f'  Overlapping Traps: k={k}, overlap={overlap}, {num_blocks} blocks')
    print(f'  genome_length={genome_length} (shared bits create inter-block epistasis)')
    print(f'{"="*60}')

    run_sweep('overlap', genome_length, evaluate_fn, init_fn,
              num_seeds=num_seeds, max_gens=200)

    if not sweep_only:
        run_evolve('overlap', genome_length, evaluate_fn, init_fn, max_gens=200)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MMDP and overlap experiments')
    parser.add_argument('--domain', choices=['mmdp', 'overlap'],
                        help='Run only this domain')
    parser.add_argument('--sweep-only', action='store_true')
    parser.add_argument('--seeds', type=int, default=30)
    args = parser.parse_args()

    if args.domain in (None, 'mmdp'):
        run_mmdp(sweep_only=args.sweep_only, num_seeds=args.seeds)

    if args.domain in (None, 'overlap'):
        run_overlap(sweep_only=args.sweep_only, num_seeds=args.seeds)


if __name__ == '__main__':
    main()
