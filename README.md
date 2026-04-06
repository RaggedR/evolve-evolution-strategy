# Evolve the Evolution Strategy

Meta-evolution of migration graphs for genetic algorithms on deceptive landscapes.

## Key Findings

1. **Fitness inversion**: On Goldberg trap functions, the diversity ordering
   (none > ring > star > FC) is preserved, but the fitness ordering *inverts*
   (FC > star > ring > none). More connectivity helps on deceptive landscapes
   because building block assembly dominates over diversity preservation.
   See [FITNESS_INVERSION.md](FITNESS_INVERSION.md).

2. **Evolved graphs beat canonical topologies**: An outer GA evolving the
   migration graph discovers moderately-connected asymmetric graphs (~50%
   edge density, pendant vertices) that outperform all standard topologies
   (none, ring, star, FC). The gap grows with island count.
   See [EVOLVED_GRAPHS.md](EVOLVED_GRAPHS.md).

## Structure

| File | Description |
|------|-------------|
| `trap_domain.py` | Goldberg concatenated k-trap fitness function (k=3,5,7) |
| `hiff_domain.py` | Watson & Pollack HIFF (hierarchical deception) |
| `run_traps.py` | Topology sweep on trap functions (experiment E compatible) |
| `run_hiff.py` | Topology sweep + graph evolution on HIFF |
| `dynamic_topology.py` | Hand-designed topology schedules (sparse→dense) |
| `evolve_graph.py` | **Outer GA evolving migration graphs** |
| `FITNESS_INVERSION.md` | Write-up: diversity vs fitness ordering on deceptive landscapes |
| `EVOLVED_GRAPHS.md` | Write-up: evolved graph structures and cross-domain patterns |

## Usage

Requires the GA infrastructure from
[categorical-evolution](https://github.com/GayleJewson/categorical-evolution):

```bash
# Run trap topology sweep
python run_traps.py --seeds 30

# Evolve migration graphs (n=5 islands, k=5 traps)
python evolve_graph.py --k 5 --islands 5

# Evolve for larger island counts
python evolve_graph.py --k 5 --islands 8
```

## Context

Extends the ACT 2026 paper "From Games to Graphs" which showed migration
topology determines diversity dynamics (Kendall's W = 1.0) across six
domains. This project tests that result on deceptive landscapes and
explores whether the topology itself can be evolved.

Collaboration between Robin Langer and Nick Meinhold.
