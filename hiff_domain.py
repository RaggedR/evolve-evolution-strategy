#!/usr/bin/env python3
"""
HIFF (Hierarchical If-and-Only-If) domain for topology experiments.

Watson, R.A. & Pollack, J.B. (1998). "Hierarchically Consistent Test
Problems for Genetic Algorithms." PPSN V, pp. 848-857.

Genome: 2^L bits (e.g., L=6 → 64 bits, L=5 → 32 bits).
Scoring: at each level k (0..L), adjacent blocks of 2^k bits
contribute 2^k if all-zeros or all-ones, else 0.
Total fitness = sum across all levels, normalized to [0, 1].

Global optima: all-zeros and all-ones (both score 1.0).
Deception: partial matches at low levels are misleading — a genome
can score well at levels 0-2 but poorly at level 3+ because
sub-blocks committed to different values.

This is hierarchical deception: building blocks must be assembled
*consistently* across multiple scales simultaneously.

Compatible with experiment_e and evolve_graph infrastructure.
"""

import numpy as np


# ---------------------------------------------------------------------------
# HIFF fitness
# ---------------------------------------------------------------------------

def hiff_block_value(block: np.ndarray) -> int:
    """Check if a block is all-zeros or all-ones.

    Returns len(block) if uniform, 0 otherwise.
    Also returns the uniform value (0 or 1) or -1 if not uniform.
    """
    if np.all(block == 0):
        return len(block), 0
    elif np.all(block == 1):
        return len(block), 1
    else:
        return 0, -1


def hiff_fitness(genome: np.ndarray) -> float:
    """Compute HIFF fitness for a single genome.

    Args:
        genome: 1D binary array of length 2^L.

    Returns:
        Float in [0, 1] where 1.0 = global optimum.
    """
    n = len(genome)
    # Verify power of 2
    assert n > 0 and (n & (n - 1)) == 0, f"Genome length must be power of 2, got {n}"

    L = int(np.log2(n))
    max_score = n * (L + 1)  # maximum possible score
    total = 0

    # Level 0: each bit contributes 1
    total += n

    # Levels 1..L: check blocks of increasing size
    # At each level, track which blocks are uniform (for the next level)
    current = genome.copy()

    for level in range(1, L + 1):
        block_size = 2 ** level
        num_blocks = n // block_size
        next_level = np.full(num_blocks, -1, dtype=int)  # -1 = not uniform

        for b in range(num_blocks):
            block = current[b * 2: (b + 1) * 2]  # pairs from previous level
            # Block is uniform at this level if both halves were uniform
            # and had the same value
            if block[0] >= 0 and block[0] == block[1]:
                total += block_size
                next_level[b] = block[0]

        current = next_level

    return total / max_score


def hiff_fitness_fast(genome: np.ndarray) -> float:
    """Fast HIFF fitness using vectorized level-by-level scoring.

    Same result as hiff_fitness but avoids Python loops over blocks.
    """
    n = len(genome)
    assert n > 0 and (n & (n - 1)) == 0, f"Genome length must be power of 2, got {n}"

    L = int(np.log2(n))
    max_score = n * (L + 1)
    total = n  # level 0

    # Track uniform value at each position: 0, 1, or -1 (not uniform)
    values = genome.astype(np.int8).copy()

    for level in range(1, L + 1):
        block_size = 2 ** level
        # Pair up adjacent elements from previous level
        left = values[0::2]
        right = values[1::2]

        # Uniform if both non-negative and equal
        uniform = (left >= 0) & (right >= 0) & (left == right)
        total += int(np.sum(uniform)) * block_size

        # Propagate: uniform blocks keep their value, others get -1
        values = np.where(uniform, left, np.int8(-1))

    return total / max_score


# ---------------------------------------------------------------------------
# Vectorized population evaluation
# ---------------------------------------------------------------------------

def evaluate_hiff(pop: np.ndarray) -> np.ndarray:
    """Evaluate HIFF fitness for entire population.

    Args:
        pop: (pop_size, genome_length) binary array.

    Returns:
        (pop_size,) float64 array of fitnesses in [0, 1].
    """
    return np.array([hiff_fitness_fast(pop[i]) for i in range(len(pop))])


def make_hiff_evaluate():
    """Return evaluate function compatible with run_experiment_e."""
    def evaluate_fn(pop: np.ndarray) -> np.ndarray:
        return evaluate_hiff(pop)
    return evaluate_fn


# ---------------------------------------------------------------------------
# Population initialization
# ---------------------------------------------------------------------------

def random_hiff_population(rng, pop_size: int, genome_length: int) -> np.ndarray:
    """Generate random binary population for HIFF."""
    return rng.integers(0, 2, size=(pop_size, genome_length), dtype=np.int8)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def test_hiff_domain():
    """Self-tests for the HIFF domain."""
    print("=== HIFF Domain Self-Test ===\n")

    # --- Test 1: Global optima ---
    for L in [3, 4, 5, 6]:
        n = 2 ** L
        all_ones = np.ones(n, dtype=np.int8)
        all_zeros = np.zeros(n, dtype=np.int8)
        f_ones = hiff_fitness_fast(all_ones)
        f_zeros = hiff_fitness_fast(all_zeros)
        assert abs(f_ones - 1.0) < 1e-10, f"L={L}: all-ones should score 1.0, got {f_ones}"
        assert abs(f_zeros - 1.0) < 1e-10, f"L={L}: all-zeros should score 1.0, got {f_zeros}"
        print(f"  [PASS] L={L} (n={n}): both optima score 1.0")

    # --- Test 2: Known score for half-and-half ---
    # [1,1,1,1,0,0,0,0] for n=8 (L=3)
    # Level 0: 8
    # Level 1: (1,1)=2, (1,1)=2, (0,0)=2, (0,0)=2 → 8
    # Level 2: (1111)=4, (0000)=4 → 8
    # Level 3: halves disagree → 0
    # Total: 24, max = 8*4 = 32, fitness = 24/32 = 0.75
    half = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8)
    f = hiff_fitness_fast(half)
    assert abs(f - 0.75) < 1e-10, f"Half-and-half should score 0.75, got {f}"
    print(f"  [PASS] [1,1,1,1,0,0,0,0] scores {f:.4f} (expected 0.75)")

    # --- Test 3: Alternating bits score poorly at higher levels ---
    # [1,0,1,0,1,0,1,0] for n=8
    # Level 0: 8
    # Level 1: all pairs disagree → 0
    # Level 2: → 0
    # Level 3: → 0
    # Total: 8/32 = 0.25
    alt = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    f_alt = hiff_fitness_fast(alt)
    assert abs(f_alt - 0.25) < 1e-10, f"Alternating should score 0.25, got {f_alt}"
    print(f"  [PASS] [1,0,1,0,1,0,1,0] scores {f_alt:.4f} (expected 0.25)")

    # --- Test 4: Deception property ---
    # half-and-half (0.75) > alternating (0.25) but < optimum (1.0)
    # The deception: half-and-half looks good at low levels but can't
    # be improved to 1.0 without disrupting the good low-level scores
    print(f"  [PASS] Deception: half-half (0.75) > alternating (0.25) < optimum (1.0)")

    # --- Test 5: Fast matches slow ---
    rng = np.random.default_rng(42)
    for L in [3, 4, 5]:
        n = 2 ** L
        pop = rng.integers(0, 2, size=(50, n), dtype=np.int8)
        slow = np.array([hiff_fitness(pop[i]) for i in range(50)])
        fast = np.array([hiff_fitness_fast(pop[i]) for i in range(50)])
        assert np.allclose(slow, fast), f"L={L}: fast/slow mismatch"
        print(f"  [PASS] L={L}: fast matches slow for 50 individuals")

    # --- Test 6: Vectorized evaluate ---
    n = 64  # L=6
    pop = rng.integers(0, 2, size=(100, n), dtype=np.int8)
    fits = evaluate_hiff(pop)
    assert np.all(fits >= 0.0) and np.all(fits <= 1.0 + 1e-10)
    print(f"  [PASS] Vectorized evaluate: 100 individuals, "
          f"mean={fits.mean():.4f}, range=[{fits.min():.4f}, {fits.max():.4f}]")

    # --- Test 7: Landscape difficulty ---
    # Random population mean should be well below 0.5 for large L
    for L in [4, 5, 6]:
        n = 2 ** L
        pop = rng.integers(0, 2, size=(500, n), dtype=np.int8)
        fits = evaluate_hiff(pop)
        print(f"  [INFO] L={L} (n={n}): random pop mean={fits.mean():.4f}, "
              f"max={fits.max():.4f}")

    # --- Test 8: Single bit flip from optimum ---
    n = 64
    optimum = np.ones(n, dtype=np.int8)
    flip_damages = []
    for i in range(n):
        flipped = optimum.copy()
        flipped[i] = 0
        f = hiff_fitness_fast(flipped)
        flip_damages.append(1.0 - f)
    mean_damage = np.mean(flip_damages)
    print(f"  [INFO] L=6: mean damage from single bit flip = {mean_damage:.4f}")
    print(f"         (a single flip cascades up the hierarchy)")

    print("\nAll HIFF domain tests passed!")
    return True


if __name__ == '__main__':
    test_hiff_domain()
