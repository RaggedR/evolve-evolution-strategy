#!/usr/bin/env python3
"""Draw the evolved graphs from all five deceptive domains."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Graph data from experiments ---

GRAPHS = {
    'Trap-5\n(flat deception)': {
        'edges': [(0,2), (0,3), (0,4), (1,2), (1,4), (2,3)],
        'degrees': [3, 2, 3, 2, 2],
        'lambda2': 1.38,
        'fitness': 0.915,
        'n': 5,
    },
    'Trap-7\n(strong deception)': {
        'edges': [(0,1), (0,2), (0,3), (1,3), (2,3), (2,4)],
        'degrees': [3, 2, 3, 3, 1],
        'lambda2': 0.83,
        'fitness': 0.893,
        'n': 5,
    },
    'HIFF\n(hierarchical)': {
        'edges': [(0,1), (0,2), (0,3), (1,2), (3,4)],
        'degrees': [3, 2, 2, 2, 1],
        'lambda2': 0.52,
        'fitness': 0.763,
        'n': 5,
    },
    'MMDP\n(multimodal)': {
        'edges': [(0,1), (0,2), (0,4), (1,2), (2,4), (3,4)],
        'degrees': [3, 2, 3, 1, 3],
        'lambda2': 0.83,
        'fitness': 0.939,
        'n': 5,
    },
    'Overlap\n(epistatic)': {
        'edges': [(0,2), (0,3), (0,4), (1,2), (1,3), (2,3)],
        'degrees': [3, 2, 3, 3, 1],
        'lambda2': 0.83,
        'fitness': 0.837,
        'n': 5,
    },
}

# Canonical topologies for comparison
CANONICAL = {
    'Ring': [(0,1), (1,2), (2,3), (3,4), (0,4)],
    'Star': [(0,1), (0,2), (0,3), (0,4)],
    'FC': [(i,j) for i in range(5) for j in range(i+1, 5)],
}


def layout_pentagon(n=5, radius=0.35):
    """Place n nodes in a regular pentagon."""
    angles = [np.pi/2 + 2*np.pi*i/n for i in range(n)]
    return [(radius * np.cos(a), radius * np.sin(a)) for a in angles]


def draw_graph(ax, edges, degrees, title, positions, lambda2=None, fitness=None):
    """Draw a graph on the given axes."""
    n = len(degrees)

    # Draw edges
    for i, j in edges:
        x = [positions[i][0], positions[j][0]]
        y = [positions[i][1], positions[j][1]]
        ax.plot(x, y, '-', color='#555555', linewidth=1.5, zorder=1)

    # Draw nodes
    max_deg = max(degrees)
    min_deg = min(degrees)
    for i in range(n):
        d = degrees[i]
        # Size by degree
        size = 200 + 150 * (d - min_deg) / max(1, max_deg - min_deg)
        # Color: pendant=blue, hub=red, middle=grey
        if d == min_deg and d <= 1:
            color = '#4E79A7'  # blue — pendant
            ec = '#2a5080'
        elif d == max_deg:
            color = '#E15759'  # red — hub
            ec = '#a03030'
        else:
            color = '#B0B0B0'  # grey — regular
            ec = '#707070'

        ax.scatter(positions[i][0], positions[i][1], s=size,
                   c=color, edgecolors=ec, linewidths=1.5, zorder=2)
        ax.annotate(str(i), positions[i], ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white', zorder=3)

    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

    # Subtitle with stats
    n_edges = len(edges)
    subtitle = f'{n_edges}e  λ₂={lambda2:.2f}' if lambda2 else f'{n_edges} edges'
    ax.text(0.5, -0.08, subtitle, transform=ax.transAxes,
            ha='center', va='top', fontsize=8, color='#666666')

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    pos = layout_pentagon()

    # --- Main figure: evolved graphs ---
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    fig.suptitle('Evolved Migration Graphs Across Five Deceptive Domains',
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, (name, g) in zip(axes, GRAPHS.items()):
        draw_graph(ax, g['edges'], g['degrees'], name, pos,
                   lambda2=g['lambda2'], fitness=g['fitness'])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#E15759', edgecolor='#a03030', label='Hub (max degree)'),
        mpatches.Patch(facecolor='#4E79A7', edgecolor='#2a5080', label='Pendant (degree 1)'),
        mpatches.Patch(facecolor='#B0B0B0', edgecolor='#707070', label='Regular'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    fig.savefig('evolved_graphs.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.15)
    fig.savefig('evolved_graphs.pdf', bbox_inches='tight',
                facecolor='white', pad_inches=0.15)
    print('Saved evolved_graphs.png and evolved_graphs.pdf')

    # --- Comparison: canonical vs best evolved ---
    fig2, axes2 = plt.subplots(1, 4, figsize=(13, 3.5))
    fig2.suptitle('Canonical Topologies vs Evolved (Trap-7)',
                  fontsize=13, fontweight='bold', y=1.02)

    # Ring
    ring_deg = [2, 2, 2, 2, 2]
    draw_graph(axes2[0], CANONICAL['Ring'], ring_deg, 'Ring\n(canonical)', pos,
               lambda2=1.38)

    # Star
    star_deg = [4, 1, 1, 1, 1]
    draw_graph(axes2[1], CANONICAL['Star'], star_deg, 'Star\n(canonical)', pos,
               lambda2=1.00)

    # FC
    fc_deg = [4, 4, 4, 4, 4]
    draw_graph(axes2[2], CANONICAL['FC'], fc_deg, 'Fully Connected\n(canonical)', pos,
               lambda2=5.00)

    # Evolved (trap-7)
    g = GRAPHS['Trap-7\n(strong deception)']
    draw_graph(axes2[3], g['edges'], g['degrees'], 'Evolved\n(Trap-7)', pos,
               lambda2=g['lambda2'])

    plt.tight_layout()
    fig2.savefig('canonical_vs_evolved.png', dpi=200, bbox_inches='tight',
                 facecolor='white', pad_inches=0.15)
    fig2.savefig('canonical_vs_evolved.pdf', bbox_inches='tight',
                 facecolor='white', pad_inches=0.15)
    print('Saved canonical_vs_evolved.png and canonical_vs_evolved.pdf')


if __name__ == '__main__':
    main()
