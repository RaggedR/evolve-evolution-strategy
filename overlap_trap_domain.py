#!/usr/bin/env python3
"""
Overlapping trap function domain.

Standard Goldberg k-traps but adjacent blocks share `overlap` bits.
This creates inter-block epistasis: solving one block's shared region
can disrupt the neighbor.

With k=5 and overlap=2:
  Block 0: bits [0, 1, 2, 3, 4]
  Block 1: bits [3, 4, 5, 6, 7]    ← shares bits 3,4 with block 0
  Block 2: bits [6, 7, 8, 9, 10]   ← shares bits 6,7 with block 1
  ...

The global optimum is still all-ones (every block solved), but
getting there requires coordinating shared bits across blocks —
a bit that's good for block A might be bad for block B.

Compatible with experiment_e and evolve_graph infrastructure.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Overlapping trap fitness
# ---------------------------------------------------------------------------

def overlap_trap_block_fitness(block: np.ndarray, k: int) -> float:
    """Score a single k-bit block using the standard trap function."""
    u = int(np.sum(block))
    if u == k:
        return float(k)
    else:
        return float(k - 1 - u)


def overlap_trap_fitness(genome: np.ndarray, k: int, overlap: int) -> float:
    """Fitness for overlapping k-traps.

    Blocks are spaced (k - overlap) apart, each spanning k bits.
    Total fitness = sum of per-block scores, normalized to [0, 1].

    Args:
        genome: 1D binary array.
        k: trap size.
        overlap: number of shared bits between adjacent blocks.

    Returns:
        Float in [0, 1].
    """
    stride = k - overlap
    num_blocks = (len(genome) - overlap) // stride
    max_score = k * num_blocks
    total = 0.0
    for b in range(num_blocks):
        start = b * stride
        block = genome[start : start + k]
        total += overlap_trap_block_fitness(block, k)
    return total / max_score


def genome_length_for_overlap(k: int, overlap: int, num_blocks: int) -> int:
    """Compute genome length for given parameters."""
    stride = k - overlap
    return stride * (num_blocks - 1) + k


# ---------------------------------------------------------------------------
# Vectorized evaluation
# ---------------------------------------------------------------------------

def evaluate_overlap_traps(pop: np.ndarray, k: int, overlap: int) -> np.ndarray:
    """Evaluate overlapping trap fitness for entire population."""
    pop_size = pop.shape[0]
    genome_len = pop.shape[1]
    stride = k - overlap
    num_blocks = (genome_len - overlap) // stride

    max_score = k * num_blocks
    totals = np.zeros(pop_size, dtype=np.float64)

    for b in range(num_blocks):
        start = b * stride
        blocks = pop[:, start : start + k]
        u = blocks.sum(axis=1)
        scores = np.where(u == k, k, k - 1 - u)
        totals += scores.astype(np.float64)

    return totals / max_score


def make_overlap_trap_evaluate(k: int, overlap: int):
    """Return evaluate function closed over (k, overlap)."""
    def evaluate_fn(pop: np.ndarray) -> np.ndarray:
        return evaluate_overlap_traps(pop, k, overlap)
    return evaluate_fn


# ---------------------------------------------------------------------------
# Population initialization
# ---------------------------------------------------------------------------

def random_overlap_population(rng, pop_size: int, genome_length: int) -> np.ndarray:
    """Generate random binary population."""
    return rng.integers(0, 2, size=(pop_size, genome_length), dtype=np.int8)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def test_overlap_trap_domain():
    """Self-tests for the overlapping trap domain."""
    print("=== Overlapping Trap Domain Self-Test ===\n")

    k = 5
    overlap = 2
    num_blocks = 10
    genome_length = genome_length_for_overlap(k, overlap, num_blocks)
    stride = k - overlap

    print(f"  Config: k={k}, overlap={overlap}, num_blocks={num_blocks}")
    print(f"  genome_length={genome_length}, stride={stride}")
    print(f"  Block positions: ", end="")
    for b in range(min(4, num_blocks)):
        s = b * stride
        print(f"[{s}:{s+k}] ", end="")
    print("...")

    # --- Test 1: All ones = global optimum ---
    all_ones = np.ones(genome_length, dtype=np.int8)
    f = overlap_trap_fitness(all_ones, k, overlap)
    assert abs(f - 1.0) < 1e-10, f"All-ones should score 1.0, got {f}"
    print(f"  [PASS] All-ones fitness = {f:.4f}")

    # --- Test 2: All zeros = deceptive attractor ---
    all_zeros = np.zeros(genome_length, dtype=np.int8)
    f_zero = overlap_trap_fitness(all_zeros, k, overlap)
    expected = (k - 1) / k
    assert abs(f_zero - expected) < 1e-10
    print(f"  [PASS] All-zeros fitness = {f_zero:.4f} (attractor)")

    # --- Test 3: Overlap creates conflict ---
    # Set block 0 to all-ones, block 1 to all-zeros
    # The shared bits [3,4] can't be both 1 (for block 0) and 0 (for block 1)
    genome = np.zeros(genome_length, dtype=np.int8)
    genome[0:k] = 1  # block 0 all-ones
    # Block 1 starts at stride=3, bits [3,4,5,6,7]
    # Bits 3,4 are already 1 from block 0 → block 1 has u=2, NOT u=0
    block1 = genome[stride : stride + k]
    u1 = int(np.sum(block1))
    assert u1 == overlap, f"Block 1 should have {overlap} ones from overlap, got {u1}"
    print(f"  [PASS] Overlap conflict: block 0 all-ones forces block 1 to u={u1}")

    # --- Test 4: Vectorized matches scalar ---
    rng = np.random.default_rng(42)
    pop = random_overlap_population(rng, 200, genome_length)
    vec_fit = evaluate_overlap_traps(pop, k, overlap)
    scalar_fit = np.array([
        overlap_trap_fitness(pop[i], k, overlap) for i in range(200)
    ])
    assert np.allclose(vec_fit, scalar_fit)
    print(f"  [PASS] Vectorized matches scalar for 200 individuals")

    # --- Test 5: Fitness in [0, 1] ---
    assert np.all(vec_fit >= 0.0) and np.all(vec_fit <= 1.0 + 1e-10)
    print(f"  [PASS] All fitnesses in [0, 1]")

    # --- Test 6: Random population statistics ---
    print(f"  [INFO] Random pop (200): mean={vec_fit.mean():.4f}, "
          f"range=[{vec_fit.min():.4f}, {vec_fit.max():.4f}]")

    # --- Test 7: Harder than non-overlapping ---
    # Compare: flip a shared bit from the attractor
    # In non-overlapping traps, a flip affects 1 block
    # In overlapping traps, a flip in the overlap region affects 2 blocks
    f_base = overlap_trap_fitness(all_zeros, k, overlap)
    shared_bit = stride  # first bit of the overlap region between block 0 and 1
    flipped = all_zeros.copy()
    flipped[shared_bit] = 1
    f_flipped = overlap_trap_fitness(flipped, k, overlap)
    damage = f_base - f_flipped
    print(f"  [INFO] Flipping shared bit {shared_bit}: damage={damage:.4f} "
          f"(affects 2 blocks simultaneously)")

    non_shared_bit = 0  # first bit of block 0, not shared
    flipped2 = all_zeros.copy()
    flipped2[non_shared_bit] = 1
    f_flipped2 = overlap_trap_fitness(flipped2, k, overlap)
    damage2 = f_base - f_flipped2
    print(f"  [INFO] Flipping non-shared bit {non_shared_bit}: damage={damage2:.4f} "
          f"(affects 1 block only)")

    print("\nAll overlapping trap domain tests passed!")
    return True


if __name__ == '__main__':
    test_overlap_trap_domain()
