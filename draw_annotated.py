#!/usr/bin/env python3
"""Draw the annotated K₄-e + pendant graph for a paper figure."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def main():
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))

    # Layout: hand-tuned for clarity
    # Diamond core (0,1,2,3) roughly in a diamond shape, pendant (4) off to the right
    pos = {
        0: (-0.8, 0.5),    # hub (top-left)
        3: (-0.8, -0.5),   # hub (bottom-left)
        2: (0.2, -0.5),    # hub (bottom-right)
        1: (0.2, 0.5),     # bridge (top-right)
        4: (1.3, -0.5),    # pendant (far right)
    }

    edges = [(0,1), (0,2), (0,3), (1,3), (2,3), (2,4)]

    # --- Shade the triangle region ---
    triangle_nodes = [0, 2, 3]
    tri_x = [pos[n][0] for n in triangle_nodes] + [pos[triangle_nodes[0]][0]]
    tri_y = [pos[n][1] for n in triangle_nodes] + [pos[triangle_nodes[0]][1]]
    ax.fill(tri_x, tri_y, alpha=0.08, color='#E15759', zorder=0)

    # --- Draw edges ---
    edge_styles = {
        (0,1): {'color': '#888888', 'lw': 2.0, 'ls': '-'},
        (0,2): {'color': '#E15759', 'lw': 2.5, 'ls': '-'},   # triangle
        (0,3): {'color': '#E15759', 'lw': 2.5, 'ls': '-'},   # triangle
        (1,3): {'color': '#888888', 'lw': 2.0, 'ls': '-'},
        (2,3): {'color': '#E15759', 'lw': 2.5, 'ls': '-'},   # triangle
        (2,4): {'color': '#4E79A7', 'lw': 2.5, 'ls': '--'},  # pendant link
    }

    for (i, j), style in edge_styles.items():
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        ax.plot(x, y, color=style['color'], linewidth=style['lw'],
                linestyle=style['ls'], zorder=1, solid_capstyle='round')

    # --- Draw missing edge (1,2) as dotted ---
    ax.plot([pos[1][0], pos[2][0]], [pos[1][1], pos[2][1]],
            color='#cccccc', linewidth=1.5, linestyle=':', zorder=1)
    mid_missing = ((pos[1][0]+pos[2][0])/2 + 0.12,
                   (pos[1][1]+pos[2][1])/2)
    ax.annotate('missing\nedge', mid_missing, ha='left', va='center',
                fontsize=7.5, color='#999999', style='italic')

    # --- Draw nodes ---
    node_config = {
        0: {'color': '#E15759', 'ec': '#a03030', 'size': 600, 'label': '0'},
        1: {'color': '#B0B0B0', 'ec': '#707070', 'size': 450, 'label': '1'},
        2: {'color': '#E15759', 'ec': '#a03030', 'size': 600, 'label': '2'},
        3: {'color': '#E15759', 'ec': '#a03030', 'size': 600, 'label': '3'},
        4: {'color': '#4E79A7', 'ec': '#2a5080', 'size': 450, 'label': '4'},
    }

    for node, cfg in node_config.items():
        ax.scatter(pos[node][0], pos[node][1], s=cfg['size'],
                   c=cfg['color'], edgecolors=cfg['ec'], linewidths=2, zorder=3)
        ax.annotate(cfg['label'], pos[node], ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white', zorder=4)

    # --- Degree labels ---
    for node in range(5):
        deg = sum(1 for (i,j) in edges if i == node or j == node)
        offset = {'x': 0, 'y': 0.18}
        if node == 3:
            offset = {'x': 0, 'y': -0.18}
        if node == 2:
            offset = {'x': 0, 'y': -0.18}
        if node == 4:
            offset = {'x': 0, 'y': -0.18}
        ax.annotate(f'deg {deg}', (pos[node][0] + offset['x'],
                    pos[node][1] + offset['y']),
                    ha='center', va='center', fontsize=8, color='#666666')

    # --- Annotations with arrows ---
    ann_style = dict(fontsize=9, bbox=dict(boxstyle='round,pad=0.4',
                     facecolor='white', edgecolor='#cccccc', alpha=0.95))
    arrow_style = dict(arrowstyle='->', color='#999999', lw=1.2,
                       connectionstyle='arc3,rad=0.2')

    # Triangle annotation
    ax.annotate('Dense core (triangle)\nRapid local exchange\nfor building block\nconsolidation',
                xy=(-0.15, -0.15), xytext=(-2.0, -0.8),
                ha='center', va='center', **ann_style,
                arrowprops=arrow_style)

    # Pendant annotation
    ax.annotate('Pendant vertex\nDiversity reservoir\nSemi-isolated search\nSingle pathway in/out',
                xy=(1.3, -0.3), xytext=(1.3, 0.65),
                ha='center', va='center', **ann_style,
                arrowprops={**arrow_style, 'connectionstyle': 'arc3,rad=-0.15'})

    # Bridge annotation
    ax.annotate('Bridge node (deg 2)\nAsymmetry within core\nTwo roles, not one',
                xy=(0.2, 0.35), xytext=(1.3, 1.4),
                ha='center', va='center', **ann_style,
                arrowprops=arrow_style)

    # Pendant link annotation
    mid_pendant = ((pos[2][0]+pos[4][0])/2, (pos[2][1]+pos[4][1])/2 - 0.18)
    ax.annotate('Bottleneck edge\n(single connection\nto reservoir)',
                xy=mid_pendant, xytext=(2.1, -0.8),
                ha='center', va='center',
                fontsize=8, color='#4E79A7', style='italic',
                arrowprops=dict(arrowstyle='->', color='#4E79A7', lw=1.0,
                               connectionstyle='arc3,rad=-0.2'))

    # --- Title and subtitle ---
    ax.set_title('K₄−e + Pendant: The Universal Evolved Topology',
                 fontsize=14, fontweight='bold', pad=20)
    ax.text(0.5, 1.02,
            'Independently discovered on Trap-7, MMDP, and Overlapping Traps\n'
            '6 edges  ·  λ₂ = 0.83  ·  degree sequence {1, 2, 3, 3, 3}',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9, color='#666666')

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor='#E15759', edgecolor='#a03030',
                       label='Hub (degree 3) — core triangle'),
        mpatches.Patch(facecolor='#B0B0B0', edgecolor='#707070',
                       label='Bridge (degree 2) — asymmetry node'),
        mpatches.Patch(facecolor='#4E79A7', edgecolor='#2a5080',
                       label='Pendant (degree 1) — diversity reservoir'),
    ]
    ax.legend(handles=legend_elements, loc='lower left',
              fontsize=8.5, frameon=True, fancybox=True,
              bbox_to_anchor=(-0.55, -0.15))

    ax.set_xlim(-2.4, 2.6)
    ax.set_ylim(-1.2, 1.7)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    fig.savefig('annotated_graph.png', dpi=250, bbox_inches='tight',
                facecolor='white', pad_inches=0.2)
    fig.savefig('annotated_graph.pdf', bbox_inches='tight',
                facecolor='white', pad_inches=0.2)
    print('Saved annotated_graph.png and annotated_graph.pdf')


if __name__ == '__main__':
    main()
