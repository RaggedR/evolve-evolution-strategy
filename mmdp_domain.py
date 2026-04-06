#!/usr/bin/env python3
"""
MMDP (Massively Multimodal Deceptive Problem) domain.

Goldberg, D.E., Deb, K. & Horn, J. (1992). "Massive Multimodality,
Deception, and Genetic Algorithms." PPSN II, pp. 37-46.

Each 6-bit block is scored by unitation (number of ones):
    u=0: 1.000  (global optimum — all zeros)
    u=1: 0.000
    u=2: 0.360384
    u=3: 0.640576
    u=4: 0.360384
    u=5: 0.000
    u=6: 1.000  (global optimum — all ones)

Key properties:
  - TWO global optima per block (all-0 and all-1), both scoring 1.0
  - The "deceptive attractor" at u=3 (0.640576) is a local optimum
  - Four deceptive basins (u=1,2,4,5) pull search toward u=3
  - Much harder than standard traps because there are multiple wrong answers

Total fitness = sum of per-block scores, normalized to [0, 1].

Compatible with experiment_e and evolve_graph infrastructure.
"""

import numpy as np


# ---------------------------------------------------------------------------
# MMDP block fitness (unitation-based)
# ---------------------------------------------------------------------------

MMDP_K = 6  # bits per block

# Fitness at each unitation level (Goldberg et al. 1992)
MMDP_FITNESS = np.array([
    1.000000,   # u=0: global optimum (all zeros)
    0.000000,   # u=1
    0.360384,   # u=2
    0.640576,   # u=3: deceptive attractor
    0.360384,   # u=4
    0.000000,   # u=5
    1.000000,   # u=6: global optimum (all ones)
])


def mmdp_block_fitness(block: np.ndarray) -> float:
    """Score a single 6-bit MMDP block by unitation."""
    u = int(np.sum(block))
    return float(MMDP_FITNESS[u])


def mmdp_fitness(genome: np.ndarray, num_blocks: int) -> float:
    """Fitness for a concatenated MMDP genome.

    Args:
        genome: 1D binary array of length 6 * num_blocks.
        num_blocks: number of MMDP blocks.

    Returns:
        Float in [0, 1].
    """
    total = 0.0
    for b in range(num_blocks):
        block = genome[b * MMDP_K : (b + 1) * MMDP_K]
        total += mmdp_block_fitness(block)
    return total / num_blocks


# ---------------------------------------------------------------------------
# Vectorized evaluation
# ---------------------------------------------------------------------------

def evaluate_mmdp(pop: np.ndarray, num_blocks: int) -> np.ndarray:
    """Evaluate MMDP fitness for entire population.

    Args:
        pop: (pop_size, genome_length) binary array.
        num_blocks: number of 6-bit blocks.

    Returns:
        (pop_size,) float64 array of fitnesses in [0, 1].
    """
    pop_size = pop.shape[0]
    blocks = pop.reshape(pop_size, num_blocks, MMDP_K)
    u = blocks.sum(axis=2).astype(int)  # (pop_size, num_blocks)
    scores = MMDP_FITNESS[u]  # lookup table
    return scores.mean(axis=1)


def make_mmdp_evaluate(num_blocks: int):
    """Return evaluate function closed over num_blocks."""
    def evaluate_fn(pop: np.ndarray) -> np.ndarray:
        return evaluate_mmdp(pop, num_blocks)
    return evaluate_fn


# ---------------------------------------------------------------------------
# Population initialization
# ---------------------------------------------------------------------------

def random_mmdp_population(rng, pop_size: int, genome_length: int) -> np.ndarray:
    """Generate random binary population for MMDP."""
    return rng.integers(0, 2, size=(pop_size, genome_length), dtype=np.int8)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def test_mmdp_domain():
    """Self-tests for the MMDP domain."""
    print("=== MMDP Domain Self-Test ===\n")

    # --- Test 1: Block scoring at each unitation level ---
    for u in range(MMDP_K + 1):
        block = np.zeros(MMDP_K, dtype=np.int8)
        block[:u] = 1
        f = mmdp_block_fitness(block)
        assert abs(f - MMDP_FITNESS[u]) < 1e-10
    print(f"  [PASS] Block scoring correct at all unitation levels")

    # --- Test 2: Both optima score 1.0 ---
    num_blocks = 10
    genome_length = MMDP_K * num_blocks
    all_zeros = np.zeros(genome_length, dtype=np.int8)
    all_ones = np.ones(genome_length, dtype=np.int8)
    assert abs(mmdp_fitness(all_zeros, num_blocks) - 1.0) < 1e-10
    assert abs(mmdp_fitness(all_ones, num_blocks) - 1.0) < 1e-10
    print(f"  [PASS] Both all-zeros and all-ones score 1.0")

    # --- Test 3: Deceptive attractor score ---
    # All blocks at u=3 (half ones, half zeros)
    attractor = np.zeros(genome_length, dtype=np.int8)
    for b in range(num_blocks):
        attractor[b * MMDP_K : b * MMDP_K + 3] = 1
    f_att = mmdp_fitness(attractor, num_blocks)
    assert abs(f_att - 0.640576) < 1e-6
    print(f"  [PASS] All-attractor genome scores {f_att:.6f}")

    # --- Test 4: Deception property ---
    # u=3 (0.640576) > u=2 (0.360384) > u=1 (0.0)
    # So the gradient from u=1,2 points toward u=3, not toward u=0 or u=6
    assert MMDP_FITNESS[3] > MMDP_FITNESS[2] > MMDP_FITNESS[1]
    assert MMDP_FITNESS[3] > MMDP_FITNESS[4] > MMDP_FITNESS[5]
    print(f"  [PASS] Deception: gradient points toward u=3 from both sides")

    # --- Test 5: Multimodality ---
    # u=0 and u=6 are both global optima
    assert MMDP_FITNESS[0] == MMDP_FITNESS[6] == 1.0
    assert MMDP_FITNESS[3] < 1.0
    print(f"  [PASS] Multimodal: two optima (u=0, u=6), attractor (u=3) is suboptimal")

    # --- Test 6: Vectorized matches scalar ---
    rng = np.random.default_rng(42)
    pop = random_mmdp_population(rng, 200, genome_length)
    vec_fit = evaluate_mmdp(pop, num_blocks)
    scalar_fit = np.array([mmdp_fitness(pop[i], num_blocks) for i in range(200)])
    assert np.allclose(vec_fit, scalar_fit)
    print(f"  [PASS] Vectorized matches scalar for 200 individuals")

    # --- Test 7: Random population statistics ---
    fits = vec_fit
    print(f"  [INFO] Random pop (200): mean={fits.mean():.4f}, "
          f"range=[{fits.min():.4f}, {fits.max():.4f}]")
    print(f"         Deceptive attractor={0.640576:.4f}, optima=1.0")

    # --- Test 8: Landscape ruggedness from attractor ---
    improvements = 0
    degradations = 0
    f_base = mmdp_fitness(attractor, num_blocks)
    for i in range(genome_length):
        flipped = attractor.copy()
        flipped[i] = 1 - flipped[i]
        f_flip = mmdp_fitness(flipped, num_blocks)
        if f_flip > f_base + 1e-10:
            improvements += 1
        elif f_flip < f_base - 1e-10:
            degradations += 1
    print(f"  [INFO] From attractor: {improvements} improving flips, "
          f"{degradations} degrading flips (out of {genome_length})")

    print("\nAll MMDP domain tests passed!")
    return True


if __name__ == '__main__':
    test_mmdp_domain()
