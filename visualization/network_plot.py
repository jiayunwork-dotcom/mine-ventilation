from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize, LinearSegmentedColormap

from core.network import VentilationNetwork


def plot_network(
    network: VentilationNetwork,
    show_airflow: bool = True,
    show_pressure: bool = True,
    show_fan_icon: bool = True,
    figsize: Tuple[int, int] = (12, 9),
    node_size: int = 800,
    edge_width_range: Tuple[float, float] = (1.0, 8.0),
    cmap_name: str = 'viridis',
    seed: Optional[int] = 42
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    G = nx.DiGraph()

    for node_id, node in network.nodes.items():
        G.add_node(node_id, elevation=node.elevation, pressure=node.pressure)

    edge_labels = {}
    edge_colors = []
    edge_widths = []
    fan_edges = []

    airflows = [abs(b.airflow) for b in network.branches.values()]
    max_airflow = max(airflows) if airflows else 1.0

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=max_airflow)

    for branch_id, branch in network.branches.items():
        q = branch.airflow
        abs_q = abs(q)

        if max_airflow > 0:
            width = edge_width_range[0] + (edge_width_range[1] - edge_width_range[0]) * (abs_q / max_airflow)
            color = cmap(norm(abs_q))
        else:
            width = edge_width_range[0]
            color = 'gray'

        if q >= 0:
            u, v = branch.from_node, branch.to_node
            label_dir = 1
        else:
            u, v = branch.to_node, branch.from_node
            label_dir = -1

        G.add_edge(u, v, branch_id=branch_id, airflow=q)

        edge_colors.append(color)
        edge_widths.append(width)

        label_parts = []
        if show_airflow:
            label_parts.append(f'Q={abs_q:.2f}')
        label_parts.append(f'R={branch.resistance:.4f}')
        edge_labels[(u, v)] = '\n'.join(label_parts)

        if branch.has_fan:
            fan_edges.append((u, v))

    if seed is not None:
        pos = nx.spring_layout(G, seed=seed, k=0.5, iterations=50)
    else:
        pos = nx.spring_layout(G, k=0.5, iterations=50)

    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=edge_widths,
        arrowstyle='->',
        arrowsize=20,
        ax=ax,
        alpha=0.8
    )

    for u, v in fan_edges:
        if show_fan_icon:
            edge_mid = ((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2)
            fan_circle = mpatches.Circle(edge_mid, 0.03, facecolor='red', edgecolor='black', linewidth=2, zorder=10)
            ax.add_patch(fan_circle)
            ax.text(edge_mid[0], edge_mid[1], 'F', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white', zorder=11)

    node_colors = []
    pressures = [n.pressure for n in network.nodes.values()]
    if pressures and max(pressures) != min(pressures):
        p_norm = Normalize(vmin=min(pressures), vmax=max(pressures))
        p_cmap = LinearSegmentedColormap.from_list('pressure', ['blue', 'white', 'red'])
        for node_id in G.nodes():
            node = network.get_node(node_id)
            node_colors.append(p_cmap(p_norm(node.pressure)))
    else:
        node_colors = ['lightblue' for _ in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_colors,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )

    node_labels = {}
    for node_id in G.nodes():
        node = network.get_node(node_id)
        label = f'N{node_id}\n'
        if show_pressure:
            label += f'{node.pressure:.1f} Pa'
        node_labels[node_id] = label

    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=10,
        font_weight='bold',
        ax=ax
    )

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=9,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
        ax=ax
    )

    if max_airflow > 0:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('风量 (m³/s)', fontsize=12)

    fan_patch = mpatches.Patch(color='red', label='扇风机')
    ax.legend(handles=[fan_patch], loc='upper right')

    ax.set_title('矿井通风网络拓扑图', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X坐标', fontsize=12)
    ax.set_ylabel('Y坐标', fontsize=12)
    ax.axis('off')
    ax.margins(0.1)

    plt.tight_layout()

    return fig


def plot_sensitivity_results(
    sensitivity_data: Dict,
    key_branches: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    resistance_factors = np.array(sensitivity_data['resistance_factors'])
    x_label = '阻力变化系数'

    if key_branches is None:
        key_branches = sensitivity_data['key_branches']

    colors = plt.cm.tab10(np.linspace(0, 1, len(key_branches)))

    for i, bid in enumerate(key_branches):
        if bid in sensitivity_data['airflow_data']:
            airflows = np.array(sensitivity_data['airflow_data'][bid])
            q0 = sensitivity_data['sensitivity_indices'][bid]['original_airflow']

            if abs(q0) > 1e-10:
                relative_change = (airflows - q0) / abs(q0) * 100
            else:
                relative_change = np.zeros_like(airflows)

            ax.plot(resistance_factors, relative_change, 
                   label=f'分支 {bid}', color=colors[i], linewidth=2, marker='o')

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='原始阻力')

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('风量相对变化 (%)', fontsize=12)
    ax.set_title(f'分支 {sensitivity_data["target_branch_id"]} 阻力变化对风量的影响', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()

    return fig


def plot_convergence_history(
    solver_info: Dict,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    residuals = solver_info.get('residuals_history', [])
    iterations = range(1, len(residuals) + 1)

    ax.semilogy(iterations, residuals, 'b-o', linewidth=2, markersize=6)

    if 'final_residual' in solver_info and 'tolerance' in solver_info:
        ax.axhline(y=solver_info['tolerance'], color='r', linestyle='--', 
                  label=f'收敛阈值 ({solver_info["tolerance"]})')

    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('残差 (对数坐标)', fontsize=12)
    ax.set_title('迭代收敛历史', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)

    converged = solver_info.get('converged', False)
    status = '收敛' if converged else '未收敛'
    final_res = solver_info.get('final_residual', 0)
    total_iter = solver_info.get('iterations', 0)

    text = f'状态: {status}\n迭代次数: {total_iter}\n最终残差: {final_res:.6e}'
    ax.text(0.02, 0.98, text, transform=ax.transAxes, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           verticalalignment='top', fontsize=10)

    plt.tight_layout()

    return fig


def plot_airflow_distribution(
    network: VentilationNetwork,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    branch_ids = sorted(network.branches.keys())
    airflows = [network.get_branch(bid).airflow for bid in branch_ids]
    abs_airflows = [abs(q) for q in airflows]

    colors = ['red' if q < 0 else 'blue' for q in airflows]

    ax1.bar(range(len(branch_ids)), airflows, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xlabel('分支编号', fontsize=12)
    ax1.set_ylabel('风量 (m³/s)', fontsize=12)
    ax1.set_title('各分支风量分布', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(branch_ids)))
    ax1.set_xticklabels([f'{bid}' for bid in branch_ids], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.hist(abs_airflows, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('风量绝对值 (m³/s)', fontsize=12)
    ax2.set_ylabel('频率', fontsize=12)
    ax2.set_title('风量分布直方图', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    mean_q = np.mean(abs_airflows) if abs_airflows else 0
    max_q = np.max(abs_airflows) if abs_airflows else 0
    min_q = np.min(abs_airflows) if abs_airflows else 0

    text = f'均值: {mean_q:.2f} m³/s\n最大: {max_q:.2f} m³/s\n最小: {min_q:.2f} m³/s'
    ax2.text(0.02, 0.98, text, transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top', fontsize=10)

    plt.tight_layout()

    return fig
