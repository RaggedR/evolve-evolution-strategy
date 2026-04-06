#!/usr/bin/env python3
"""
Goldberg trap function domain for topology sweep experiments.

Concatenated k-traps: the genome is divided into blocks of k bits.
Each block is scored by a trap function that is *deceptive* — the
fitness gradient within each block points toward all-zeros, but the
global optimum is all-ones.

For a single block of k bits with u = number of ones:
    f(u) = k        if u == k   (global optimum)
    f(u) = k-1-u    otherwise   (deceptive attractor at u=0)

The total fitness is the sum over all blocks, normalized to [0, 1].

Key property: the all-zeros genome scores (k-1)*num_traps, which is
*higher* than any genome with partially-correct blocks. A GA doing
hillclimbing will converge to all-zeros unless it maintains enough
diversity to discover all-ones blocks.

Reference: Goldberg, D.E. (1987). "Simple genetic algorithms and the
minimal deceptive problem." In Genetic Algorithms and Simulated
Annealing, pp. 74-88.

Compatible with the experiment_e infrastructure in categorical-evolution.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Trap function core
# ---------------------------------------------------------------------------

def trap_block_fitness(block: np.ndarray, k: int) -> float:
    """Score a single k-bit block using the trap function.

    Args:
        block: 1D array of k bits.
        k: trap size (== len(block)).

    Returns:
        Float fitness for this block.
    """
    u = int(np.sum(block))
    if u == k:
        return float(k)
    else:
        return float(k - 1 - u)


def trap_fitness(genome: np.ndarray, k: int, num_traps: int) -> float:
    """Fitness for a concatenated k-trap genome.

    The genome is split into num_traps consecutive blocks of k bits.
    Total fitness is the sum of per-block trap scores, normalized
    to [0, 1] where 1.0 = all blocks solved (all ones).

    Args:
        genome: 1D binary array of length k * num_traps.
        k: trap size.
        num_traps: number of concatenated traps.

    Returns:
        Float in [0, 1].
    """
    max_score = k * num_traps
    total = 0.0
    for t in range(num_traps):
        block = genome[t * k : (t + 1) * k]
        total += trap_block_fitness(block, k)
    return total / max_score


# ---------------------------------------------------------------------------
# Vectorized population evaluation
# ---------------------------------------------------------------------------

def evaluate_traps(pop: np.ndarray, k: int, num_traps: int) -> np.ndarray:
    """Evaluate trap fitness for entire population.

    Args:
        pop: (pop_size, genome_length) binary array.
        k: trap size.
        num_traps: number of traps.

    Returns:
        (pop_size,) float64 array of fitnesses in [0, 1].
    """
    max_score = k * num_traps
    pop_size = pop.shape[0]

    # Reshape to (pop_size, num_traps, k) and sum each block
    blocks = pop.reshape(pop_size, num_traps, k)
    u = blocks.sum(axis=2)  # (pop_size, num_traps) — ones count per block

    # Trap scoring: k if u==k, else k-1-u
    scores = np.where(u == k, k, k - 1 - u)  # (pop_size, num_traps)
    totals = scores.sum(axis=1).astype(np.float64)  # (pop_size,)

    return totals / max_score


# ---------------------------------------------------------------------------
# Factory: create evaluate_fn for a specific (k, num_traps)
# ---------------------------------------------------------------------------

def make_trap_evaluate(k: int, num_traps: int):
    """Return an evaluate function closed over (k, num_traps).

    The returned function has signature:
        evaluate_fn(pop: np.ndarray) -> np.ndarray

    Compatible with run_experiment_e's evaluate_fn parameter.
    """
    def evaluate_fn(pop: np.ndarray) -> np.ndarray:
        return evaluate_traps(pop, k, num_traps)
    return evaluate_fn


# ---------------------------------------------------------------------------
# Population initialization
# ---------------------------------------------------------------------------

def random_trap_population(rng, pop_size: int, genome_length: int) -> np.ndarray:
    """Generate random binary population for trap domain.

    Same as OneMax — uniform random bits.
    """
    return rng.integers(0, 2, size=(pop_size, genome_length), dtype=np.int8)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def test_trap_domain():
    """Self-tests for the trap domain."""
    print("=== Goldberg Trap Domain Self-Test ===\n")

    # --- Test 1: Single block scoring ---
    for k in [3, 5, 7]:
        # All ones = global optimum, score = k
        ones = np.ones(k, dtype=np.int8)
        assert trap_block_fitness(ones, k) == k, f"k={k}: all-ones should score {k}"

        # All zeros = deceptive attractor, score = k-1
        zeros = np.zeros(k, dtype=np.int8)
        assert trap_block_fitness(zeros, k) == k - 1, f"k={k}: all-zeros should score {k-1}"

        # One bit set = worst score, k-1-1 = k-2
        one_bit = np.zeros(k, dtype=np.int8)
        one_bit[0] = 1
        assert trap_block_fitness(one_bit, k) == k - 2, f"k={k}: one bit should score {k-2}"

        print(f"  [PASS] k={k}: block scoring correct "
              f"(optimum={k}, attractor={k-1}, worst={k-2})")

    # --- Test 2: Deception property ---
    # For any k, all-zeros scores higher than any partially-ones block
    for k in [3, 5, 7]:
        zero_score = trap_block_fitness(np.zeros(k, dtype=np.int8), k)
        for u in range(1, k):
            block = np.zeros(k, dtype=np.int8)
            block[:u] = 1
            partial_score = trap_block_fitness(block, k)
            assert zero_score > partial_score, (
                f"k={k}: all-zeros ({zero_score}) should beat {u}-ones ({partial_score})")
        print(f"  [PASS] k={k}: deception verified — all-zeros beats all partial solutions")

    # --- Test 3: Full genome scoring ---
    k, num_traps = 5, 10
    genome_length = k * num_traps

    # All ones = 1.0
    all_ones = np.ones(genome_length, dtype=np.int8)
    f = trap_fitness(all_ones, k, num_traps)
    assert abs(f - 1.0) < 1e-10, f"All-ones should have fitness 1.0, got {f}"
    print(f"  [PASS] k={k}, {num_traps} traps: all-ones fitness = {f:.4f}")

    # All zeros = (k-1)/k = 0.8 for k=5
    all_zeros = np.zeros(genome_length, dtype=np.int8)
    f_zero = trap_fitness(all_zeros, k, num_traps)
    expected = (k - 1) / k
    assert abs(f_zero - expected) < 1e-10, (
        f"All-zeros should have fitness {expected}, got {f_zero}")
    print(f"  [PASS] k={k}, {num_traps} traps: all-zeros fitness = {f_zero:.4f} "
          f"(deceptive attractor)")

    # --- Test 4: Vectorized matches scalar ---
    rng = np.random.default_rng(42)
    pop = random_trap_population(rng, 200, genome_length)
    vec_fit = evaluate_traps(pop, k, num_traps)
    scalar_fit = np.array([trap_fitness(pop[i], k, num_traps) for i in range(200)])
    assert np.allclose(vec_fit, scalar_fit), (
        f"Vectorized/scalar mismatch: max diff = {np.max(np.abs(vec_fit - scalar_fit))}")
    print(f"  [PASS] Vectorized matches scalar for 200 individuals")

    # --- Test 5: Fitness in [0, 1] ---
    assert np.all(vec_fit >= 0.0), f"Found negative fitness: {vec_fit.min()}"
    assert np.all(vec_fit <= 1.0 + 1e-10), f"Found fitness > 1: {vec_fit.max()}"
    print(f"  [PASS] All fitnesses in [0, 1]")

    # --- Test 6: Deceptive attractor is higher than random population mean ---
    mean_fit = vec_fit.mean()
    print(f"  [INFO] Random pop mean fitness = {mean_fit:.4f}, "
          f"deceptive attractor = {f_zero:.4f}, "
          f"global optimum = 1.0000")
    assert f_zero > mean_fit, (
        f"Deceptive attractor ({f_zero}) should exceed random mean ({mean_fit})")
    print(f"  [PASS] Deceptive attractor ({f_zero:.4f}) > random mean ({mean_fit:.4f})")

    # --- Test 7: Landscape ruggedness from deceptive attractor ---
    # Flipping any bit in all-zeros DECREASES fitness (except flipping
    # all bits in a block, which is extremely unlikely). This confirms
    # the attractor is a local optimum.
    all_zeros_genome = np.zeros(genome_length, dtype=np.int8)
    f_attractor = trap_fitness(all_zeros_genome, k, num_traps)
    worse_count = 0
    for i in range(genome_length):
        flipped = all_zeros_genome.copy()
        flipped[i] = 1
        f_flipped = trap_fitness(flipped, k, num_traps)
        if f_flipped < f_attractor:
            worse_count += 1
    assert worse_count == genome_length, (
        f"Every single bit flip from all-zeros should decrease fitness, "
        f"but {genome_length - worse_count} didn't")
    print(f"  [PASS] All {genome_length} single-bit flips from attractor decrease fitness "
          f"— confirmed local optimum")

    # --- Test 8: factory function ---
    eval_fn = make_trap_evaluate(k, num_traps)
    factory_fit = eval_fn(pop)
    assert np.allclose(factory_fit, vec_fit), "Factory function should match direct call"
    print(f"  [PASS] make_trap_evaluate factory matches direct call")

    # --- Summary ---
    print(f"\n  Deception gap by k:")
    for k_val in [3, 5, 7]:
        attractor = (k_val - 1) / k_val
        gap = 1.0 - attractor
        print(f"    k={k_val}: attractor={attractor:.4f}, "
              f"optimum=1.0000, gap={gap:.4f} ({gap*100:.1f}%)")

    print("\nAll trap domain tests passed!")
    return True


if __name__ == '__main__':
    test_trap_domain()
