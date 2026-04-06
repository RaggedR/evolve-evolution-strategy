#!/usr/bin/env python3
"""
Evolve the evolution strategy: outer GA evolves migration graphs,
inner GA solves Goldberg trap functions using those graphs.

The graph genome is an adjacency vector encoding the upper triangle
of an n×n symmetric matrix (n islands, C(n,2) possible edges).

Outer GA:
  - Population of graph genomes
  - Fitness = mean best_fitness of inner GA runs on trap-k
  - Tournament selection, uniform crossover, bit-flip mutation

Inner GA:
  - Standard island-model GA on concatenated k-traps
  - Migration topology determined by the graph genome

Usage:
    python evolve_graph.py                    # Default: k=5, 5 islands
    python evolve_graph.py --k 7              # Strongly deceptive
    python evolve_graph.py --islands 8        # 8 islands (28-bit graph genome)
    python evolve_graph.py --outer-gens 50 --outer-pop 30
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Import GA infrastructure
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
    hamming_diversity,
)

from trap_domain import make_trap_evaluate, random_trap_population


# ---------------------------------------------------------------------------
# Graph genome: adjacency vector for n islands
# ---------------------------------------------------------------------------

# Default (overridden by --islands flag)
NUM_ISLANDS = 5
NUM_EDGES = NUM_ISLANDS * (NUM_ISLANDS - 1) // 2  # 10


def set_num_islands(n: int):
    """Set the global island count (called from main)."""
    global NUM_ISLANDS, NUM_EDGES
    NUM_ISLANDS = n
    NUM_EDGES = n * (n - 1) // 2


def edge_index(i: int, j: int) -> int:
    """Map island pair (i,j) with i<j to index in the adjacency vector."""
    assert i < j
    return i * NUM_ISLANDS - i * (i + 1) // 2 + (j - i - 1)


def adj_vector_to_pairs(adj: np.ndarray) -> List[Tuple[int, int]]:
    """Convert adjacency vector to list of (i,j) edge pairs."""
    pairs = []
    idx = 0
    for i in range(NUM_ISLANDS):
        for j in range(i + 1, NUM_ISLANDS):
            if adj[idx]:
                pairs.append((i, j))
            idx += 1
    return pairs


def adj_vector_to_matrix(adj: np.ndarray) -> np.ndarray:
    """Convert adjacency vector to n×n symmetric matrix."""
    mat = np.zeros((NUM_ISLANDS, NUM_ISLANDS), dtype=int)
    idx = 0
    for i in range(NUM_ISLANDS):
        for j in range(i + 1, NUM_ISLANDS):
            mat[i, j] = adj[idx]
            mat[j, i] = adj[idx]
            idx += 1
    return mat


def graph_label(adj: np.ndarray) -> str:
    """Human-readable label for a graph genome."""
    edges = adj_vector_to_pairs(adj)
    n_edges = len(edges)
    if n_edges == 0:
        return "disconnected"
    if n_edges == NUM_EDGES:
        return "complete"
    mat = adj_vector_to_matrix(adj)
    degrees = mat.sum(axis=1)
    if n_edges == NUM_ISLANDS and all(d == 2 for d in degrees):
        return f"ring({n_edges}e)"
    if max(degrees) == NUM_ISLANDS - 1 and sum(d == 1 for d in degrees) == NUM_ISLANDS - 1:
        return f"star(hub={np.argmax(degrees)})"
    return f"graph({n_edges}e)"


def canonical_graphs():
    """Return dict of canonical topology names -> adjacency vectors."""
    graphs = {}

    # None (disconnected)
    graphs['none'] = np.zeros(NUM_EDGES, dtype=np.int8)

    # Ring: 0-1-2-..-(n-1)-0
    ring = np.zeros(NUM_EDGES, dtype=np.int8)
    ring_edges = [(i, (i + 1) % NUM_ISLANDS) for i in range(NUM_ISLANDS)]
    for i, j in ring_edges:
        a, b = min(i, j), max(i, j)
        ring[edge_index(a, b)] = 1
    graphs['ring'] = ring

    # Star: hub = 0
    star = np.zeros(NUM_EDGES, dtype=np.int8)
    for j in range(1, NUM_ISLANDS):
        star[edge_index(0, j)] = 1
    graphs['star'] = star

    # Fully connected
    graphs['fc'] = np.ones(NUM_EDGES, dtype=np.int8)

    return graphs


def compute_lambda2(adj: np.ndarray) -> float:
    """Compute algebraic connectivity λ₂ of the graph."""
    mat = adj_vector_to_matrix(adj)
    degrees = mat.sum(axis=1)
    laplacian = np.diag(degrees) - mat
    eigenvalues = np.sort(np.linalg.eigvalsh(laplacian.astype(float)))
    return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0


# ---------------------------------------------------------------------------
# Custom migration using adjacency vector
# ---------------------------------------------------------------------------

def adjacency_migrate(rng: np.random.Generator, islands: List[np.ndarray],
                      adj: np.ndarray, migration_rate: float) -> List[np.ndarray]:
    """Migrate individuals along edges defined by adjacency vector."""
    result = [isl.copy() for isl in islands]
    edges = adj_vector_to_pairs(adj)

    if not edges:
        return result

    for i, j in edges:
        pop_size_i = len(result[i])
        pop_size_j = len(result[j])
        num_migrants = max(1, round(migration_rate * min(pop_size_i, pop_size_j)))

        idx_i = rng.choice(pop_size_i, size=num_migrants, replace=False)
        idx_j = rng.choice(pop_size_j, size=num_migrants, replace=False)

        migrants_i = result[i][idx_i].copy()
        migrants_j = result[j][idx_j].copy()
        result[i][idx_i] = migrants_j
        result[j][idx_j] = migrants_i

    return result


# ---------------------------------------------------------------------------
# Inner GA: solve traps with a given graph topology
# ---------------------------------------------------------------------------

def inner_ga_fitness(adj: np.ndarray, k: int, num_traps: int,
                     inner_seeds: List[int], inner_config: GAConfig) -> float:
    """Run inner GA with the given migration graph, return mean best fitness.

    Averages over multiple seeds for robustness.
    """
    evaluate_fn = make_trap_evaluate(k, num_traps)
    fitnesses = []

    for seed in inner_seeds:
        rng = np.random.default_rng(seed)
        init_pop = random_trap_population(rng, inner_config.population_size,
                                          inner_config.genome_length)
        islands = split_population(init_pop, inner_config.num_islands)

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

        # Final fitness
        all_pop = merge_populations(islands)
        all_fit = evaluate_fn(all_pop)
        fitnesses.append(float(np.max(all_fit)))

    return float(np.mean(fitnesses))


# ---------------------------------------------------------------------------
# Outer GA: evolve graph genomes
# ---------------------------------------------------------------------------

@dataclass
class OuterConfig:
    pop_size: int = 20
    max_generations: int = 30
    tournament_size: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.15    # per-edge flip rate (higher than inner)
    elite_count: int = 2           # elitism: keep top N unchanged
    inner_seeds_per_eval: int = 5  # seeds for inner GA fitness averaging


def outer_tournament_select(rng, pop, fitnesses, tournament_size):
    """Tournament selection for graph genomes."""
    n = len(pop)
    selected = []
    for _ in range(n):
        contestants = rng.integers(0, n, size=tournament_size)
        best = contestants[np.argmax(fitnesses[contestants])]
        selected.append(pop[best].copy())
    return selected


def outer_crossover(rng, parent1, parent2, rate):
    """Uniform crossover on graph genomes."""
    if rng.random() < rate:
        mask = rng.integers(0, 2, size=len(parent1), dtype=np.int8)
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    return parent1.copy(), parent2.copy()


def outer_mutate(rng, genome, rate):
    """Bit-flip mutation on graph genome."""
    mutant = genome.copy()
    for i in range(len(mutant)):
        if rng.random() < rate:
            mutant[i] = 1 - mutant[i]
    return mutant


def evolve_graphs(k: int, num_traps: int, outer_cfg: OuterConfig,
                  inner_cfg: GAConfig, seed: int = 42):
    """Run the outer GA to evolve migration graphs."""
    rng = np.random.default_rng(seed)
    inner_seeds = list(range(outer_cfg.inner_seeds_per_eval))

    # Initialize outer population: random graphs
    pop = [rng.integers(0, 2, size=NUM_EDGES, dtype=np.int8)
           for _ in range(outer_cfg.pop_size)]

    # Inject canonical topologies into initial population
    canonical = canonical_graphs()
    for i, (name, adj) in enumerate(canonical.items()):
        if i < len(pop):
            pop[i] = adj.copy()

    # Evaluate initial population
    fitnesses = np.array([
        inner_ga_fitness(g, k, num_traps, inner_seeds, inner_cfg)
        for g in pop
    ])

    history = []  # track best per generation

    print(f'\n  Gen  Best Fit  Mean Fit  Best Graph             λ₂')
    print(f'  {"─"*60}')

    for gen in range(outer_cfg.max_generations):
        best_idx = np.argmax(fitnesses)
        best_fit = fitnesses[best_idx]
        mean_fit = np.mean(fitnesses)
        best_graph = pop[best_idx]
        l2 = compute_lambda2(best_graph)

        history.append({
            'generation': gen,
            'best_fitness': best_fit,
            'mean_fitness': mean_fit,
            'best_graph': best_graph.copy(),
            'best_label': graph_label(best_graph),
            'best_lambda2': l2,
            'best_edges': int(best_graph.sum()),
        })

        print(f'  {gen:3d}  {best_fit:.4f}    {mean_fit:.4f}    '
              f'{graph_label(best_graph):20s}  λ₂={l2:.3f}')

        # --- Selection ---
        selected = outer_tournament_select(rng, pop, fitnesses,
                                           outer_cfg.tournament_size)

        # --- Crossover ---
        children = []
        for i in range(0, len(selected) - 1, 2):
            c1, c2 = outer_crossover(rng, selected[i], selected[i + 1],
                                     outer_cfg.crossover_rate)
            children.extend([c1, c2])
        if len(selected) % 2 == 1:
            children.append(selected[-1].copy())

        # --- Mutation ---
        children = [outer_mutate(rng, c, outer_cfg.mutation_rate)
                    for c in children]

        # --- Elitism ---
        elite_indices = np.argsort(fitnesses)[-outer_cfg.elite_count:]
        for i, ei in enumerate(elite_indices):
            children[i] = pop[ei].copy()

        # --- Evaluate children ---
        child_fitnesses = np.array([
            inner_ga_fitness(g, k, num_traps, inner_seeds, inner_cfg)
            for g in children
        ])

        pop = children
        fitnesses = child_fitnesses

    # Final report
    best_idx = np.argmax(fitnesses)
    return pop[best_idx], fitnesses[best_idx], history


# ---------------------------------------------------------------------------
# Analysis: compare evolved graph against baselines
# ---------------------------------------------------------------------------

def compare_against_baselines(best_graph, k, num_traps, inner_cfg,
                              num_seeds=30):
    """Run the evolved graph and baselines with more seeds for fair comparison."""
    seeds = list(range(num_seeds))
    candidates = canonical_graphs()
    candidates['evolved'] = best_graph

    print(f'\n  {"="*65}')
    print(f'  Final comparison (k={k}, {num_seeds} seeds each)')
    print(f'  {"="*65}')
    print(f'  {"Topology":20s}  {"Edges":>5s}  {"λ₂":>6s}  {"Best Fit":>10s}  {"Std":>8s}')
    print(f'  {"─"*65}')

    for name, adj in candidates.items():
        fits = []
        for seed in seeds:
            f = inner_ga_fitness(adj, k, num_traps, [seed], inner_cfg)
            fits.append(f)
        fits = np.array(fits)
        n_edges = int(adj.sum())
        l2 = compute_lambda2(adj)
        print(f'  {name:20s}  {n_edges:5d}  {l2:6.3f}  {fits.mean():10.4f}  {fits.std():8.4f}'
              + ('  ★' if name == 'evolved' else ''))

    # Print the evolved graph
    print(f'\n  Evolved graph adjacency:')
    mat = adj_vector_to_matrix(best_graph)
    edges = adj_vector_to_pairs(best_graph)
    print(f'    Edges: {edges}')
    print(f'    Degrees: {list(mat.sum(axis=1).astype(int))}')
    print(f'    Matrix:')
    for row in mat:
        print(f'      [{", ".join(str(int(x)) for x in row)}]')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evolve migration graphs for deceptive landscapes')
    parser.add_argument('--k', type=int, default=5, choices=[3, 5, 7],
                        help='Trap size (default: 5)')
    parser.add_argument('--islands', type=int, default=5,
                        help='Number of islands (default: 5)')
    parser.add_argument('--outer-gens', type=int, default=30,
                        help='Outer GA generations (default: 30)')
    parser.add_argument('--outer-pop', type=int, default=20,
                        help='Outer GA population size (default: 20)')
    parser.add_argument('--inner-seeds', type=int, default=5,
                        help='Inner GA seeds per fitness eval (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Outer GA random seed (default: 42)')
    args = parser.parse_args()

    # Set island count globally
    set_num_islands(args.islands)

    # Scale outer GA for larger search spaces
    outer_pop = args.outer_pop
    outer_gens = args.outer_gens
    if NUM_EDGES > 15 and outer_pop < 30:
        outer_pop = 30  # larger pop for bigger search space
    if NUM_EDGES > 15 and outer_gens < 50:
        outer_gens = 50

    k = args.k
    num_traps = 10
    genome_length = k * num_traps

    # Total inner population scales with island count
    pop_per_island = 20
    total_inner_pop = pop_per_island * NUM_ISLANDS

    inner_cfg = GAConfig(
        population_size=total_inner_pop,
        genome_length=genome_length,
        num_islands=NUM_ISLANDS,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=1.0 / genome_length,
        max_generations=200,
        migration_freq=5,
        migration_rate=0.05,
    )

    # Scale mutation rate for graph genome: ~1 expected flip per mutation
    graph_mut_rate = 1.0 / NUM_EDGES

    outer_cfg = OuterConfig(
        pop_size=outer_pop,
        max_generations=outer_gens,
        mutation_rate=graph_mut_rate,
        inner_seeds_per_eval=args.inner_seeds,
    )

    total_inner_runs = outer_cfg.pop_size * outer_cfg.max_generations * outer_cfg.inner_seeds_per_eval
    print(f'Evolving Migration Graphs')
    print(f'  Islands: {NUM_ISLANDS} (graph genome: {NUM_EDGES} bits)')
    print(f'  Trap: k={k}, {num_traps} traps, {genome_length}-bit genome')
    print(f'  Inner GA: 200 gens, {NUM_ISLANDS} islands × {pop_per_island} = {total_inner_pop} individuals')
    print(f'  Outer GA: pop={outer_cfg.pop_size}, gens={outer_cfg.max_generations}, '
          f'mut_rate={graph_mut_rate:.3f}')
    print(f'  Inner seeds/eval: {outer_cfg.inner_seeds_per_eval}')
    print(f'  Total inner GA runs: ~{total_inner_runs}')
    print(f'  Graph search space: 2^{NUM_EDGES} = {2**NUM_EDGES:,} possible graphs')

    t0 = time.time()
    best_graph, best_fitness, history = evolve_graphs(
        k, num_traps, outer_cfg, inner_cfg, seed=args.seed)
    elapsed = time.time() - t0

    print(f'\n  Evolution completed in {elapsed:.1f}s')
    print(f'  Best graph: {graph_label(best_graph)} '
          f'({int(best_graph.sum())} edges, λ₂={compute_lambda2(best_graph):.3f})')
    print(f'  Best fitness: {best_fitness:.4f}')

    # Fair comparison with more seeds
    compare_against_baselines(best_graph, k, num_traps, inner_cfg, num_seeds=30)


if __name__ == '__main__':
    main()
