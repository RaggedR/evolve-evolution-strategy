# Fitness Inversion on Deceptive Landscapes

## The Result

On Goldberg trap functions (k=3, 5, 7 with 10 concatenated traps each),
the diversity ordering is preserved but the fitness ordering **inverts**
compared to honest landscapes.

**Diversity ordering (preserved):**
none > ring > star > random > FC — same as OneMax, maze, knapsack, etc.
Kendall's W consistent with the ACT 2026 result.

**Fitness ordering (inverted):**
FC > random ≈ star > ring > none — the **exact reverse** of the diversity
ordering.

On honest landscapes (OneMax, graph coloring, knapsack), more diversity
correlates with better solutions. On deceptive landscapes, diversity and
fitness **decouple**: the most diverse populations find the worst solutions.

### Data (30 seeds, 200 generations, 5 islands)

| Trap | Topology | Diversity | Best Fitness | Solved |
|------|----------|-----------|-------------|--------|
| k=3 | none | 0.284 | 0.976 | 11/30 |
| k=3 | ring | 0.174 | 0.994 | 25/30 |
| k=3 | star | 0.165 | 0.999 | 29/30 |
| k=3 | random | 0.166 | 0.998 | 28/30 |
| k=3 | FC | 0.145 | **1.000** | **30/30** |
| k=5 | none | 0.330 | 0.853 | 0/30 |
| k=5 | ring | 0.178 | 0.902 | 0/30 |
| k=5 | star | 0.127 | 0.907 | 0/30 |
| k=5 | random | 0.107 | **0.914** | 0/30 |
| k=5 | FC | 0.098 | 0.910 | 0/30 |
| k=7 | none | 0.155 | 0.875 | 0/30 |
| k=7 | ring | 0.083 | 0.889 | 0/30 |
| k=7 | star | 0.077 | 0.889 | 0/30 |
| k=7 | random | 0.068 | 0.890 | 0/30 |
| k=7 | FC | 0.065 | **0.891** | 0/30 |

## Why: Building Block Assembly

The mechanism is building block theory (Goldberg 1989).

On a trap function, each k-bit block must be solved independently (all
ones), and these solutions must be **assembled** across the population.
Two phases are needed:

1. **Discovery**: an island must find all-ones for a trap block despite
   the deceptive gradient pointing toward all-zeros. This benefits from
   diversity (sparse topology).

2. **Assembly**: once a block is solved on one island, that building block
   must spread to other islands so it can combine with other solved blocks.
   This benefits from connectivity (dense topology).

On honest landscapes, there's nothing to spread — every island converges
to the same optimum independently. Diversity directly predicts fitness.

On deceptive landscapes, the assembly phase dominates. FC's fast mixing
(highest λ₂) spreads building blocks fastest. Isolated islands (none)
trap partial solutions locally where they're disrupted before combining.

## Implications

### For the categorical framework

The laxator φ_G measures how much migration disrupts composition. On
honest landscapes, this disruption is harmful (homogenizes diverse
islands). On deceptive landscapes, this same disruption is **beneficial**
(spreads building blocks).

The laxator's *sign* depends on context:
- Honest landscape: large ||φ_G|| = bad (diversity erosion)
- Deceptive landscape: large ||φ_G|| = good (building block assembly)

This connects to the coupling sign-flip result (Lan vs Ran):
the same categorical structure governs both, with the landscape
determining which adjunction is beneficial.

### For evolving the strategy

If the optimal topology depends on the landscape, no fixed topology is
universally best for fitness. This motivates evolving the migration
graph — either:
- **Meta-evolution**: outer GA evolves topology per problem class
- **Dynamic scheduling**: sparse early (discovery) → dense late (assembly)
- **Self-adaptive**: individuals encode their own connectivity

Preliminary dynamic scheduling experiments (none→ring→star→FC over 200
generations) show the right direction but marginal improvement over
fixed FC on traps. The scheduling may need to be evolved rather than
hand-designed.

## Reproducing

```bash
cd ~/git/nick/evolve-evolution-strategy

# Run trap experiments (fixed topologies)
python run_traps.py --seeds 30

# Run dynamic topology experiments
python dynamic_topology.py --seeds 30
```

## Connection to Existing Work

- **ACT 2026 paper**: extends the W=1.0 result to deceptive landscapes
  (diversity ordering holds universally; fitness ordering does not)
- **Topology-experiments (Sudoku)**: Sudoku shows strong λ₂-diversity
  correlation (ρ=0.83) but flat fitness ordering — hard but not deceptive
- **Orchestration (sign-flip)**: the coupling sign determines whether
  mixing helps or hurts — same principle as fitness inversion
- **Lyra's NK landscapes**: η² scaling with K approaches this regime
  from the epistasis angle

## Open Questions

1. **Where is the crossover?** At what level of deception does the fitness
   ordering flip? NK landscapes with varying K could trace this transition.
2. **Does the inversion hold for non-trap deception?** HIFF, Royal Road
   with deceptive byways, real-world deceptive problems.
3. **Can an evolved topology schedule significantly beat fixed FC?** Our
   hand-designed schedules show the right direction but p > 0.6.
4. **What does the laxator look like for deceptive fitness functions?**
   The explicit construction of φ_G (open problem from the ACT paper)
   would formalize the sign-dependent effect.
