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


def plot_time_series_airflows(
    timestamps: List[float],
    airflow_data: Dict[int, List[float]],
    selected_branches: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (14, 8),
    time_markers: Optional[List[float]] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    if selected_branches is None:
        selected_branches = sorted(airflow_data.keys())

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected_branches), 10)))

    for i, bid in enumerate(selected_branches):
        if bid in airflow_data:
            airflows = airflow_data[bid]
            if len(airflows) == len(timestamps):
                ax.plot(
                    timestamps,
                    airflows,
                    label=f'分支 {bid}',
                    color=colors[i % len(colors)],
                    linewidth=2,
                    marker=None if len(timestamps) > 50 else 'o',
                    markersize=4,
                    alpha=0.85
                )

    if time_markers:
        for tm in time_markers:
            ax.axvline(x=tm, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_ylabel('风量 (m³/s)', fontsize=12)
    ax.set_title('各分支风量时间序列变化', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=min(3, max(1, len(selected_branches) // 5 + 1)))
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim(min(timestamps), max(timestamps))

    plt.tight_layout()
    return fig


def plot_time_series_pressures(
    timestamps: List[float],
    pressure_data: Dict[int, List[float]],
    selected_nodes: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    if selected_nodes is None:
        selected_nodes = sorted(pressure_data.keys())

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(selected_nodes), 9)))

    for i, nid in enumerate(selected_nodes):
        if nid in pressure_data:
            pressures = pressure_data[nid]
            if len(pressures) == len(timestamps):
                ax.plot(
                    timestamps,
                    pressures,
                    label=f'节点 {nid}',
                    color=colors[i % len(colors)],
                    linewidth=2,
                    marker=None if len(timestamps) > 50 else 's',
                    markersize=4,
                    alpha=0.85
                )

    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_ylabel('风压 (Pa)', fontsize=12)
    ax.set_title('各节点风压时间序列变化', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=min(3, max(1, len(selected_nodes) // 5 + 1)))
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim(min(timestamps), max(timestamps))

    plt.tight_layout()
    return fig


def plot_rule_preview(
    rule,
    total_hours: float = 24.0,
    n_points: int = 200,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    from core.time_series import compute_parameter_factor, ChangeMode

    fig, ax = plt.subplots(figsize=figsize)

    times = np.linspace(0, total_hours, n_points)
    factors = [compute_parameter_factor(rule, t) for t in times]

    param_name = "阻力系数倍率" if rule.parameter_type.value == "resistance" else "风机转速倍率"
    mode_names = {
        ChangeMode.STEP: "阶跃变化",
        ChangeMode.LINEAR: "线性变化",
        ChangeMode.SINE: "正弦波动",
    }
    mode_name = mode_names.get(rule.mode, str(rule.mode.value))

    ax.plot(times, factors, 'b-', linewidth=2.5, label=param_name)
    ax.fill_between(times, factors, alpha=0.2, color='skyblue')

    if rule.mode == ChangeMode.STEP:
        ax.axvline(x=rule.start_time, color='red', linestyle='--', alpha=0.7, label=f'变化时刻: {rule.start_time}h', linewidth=1.5)
    elif rule.mode == ChangeMode.LINEAR:
        ax.axvspan(rule.start_time, rule.end_time, alpha=0.15, color='orange', label=f'变化区间')
    elif rule.mode == ChangeMode.SINE:
        for k in range(int(total_hours // rule.period) + 1):
            phase_t = rule.phase + k * rule.period
            if 0 <= phase_t <= total_hours:
                ax.axvline(x=phase_t, color='green', linestyle=':', alpha=0.5, linewidth=1)

    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_ylabel(param_name, fontsize=12)
    ax.set_title(
        f'分支 {rule.branch_id} - {mode_name}曲线预览\n'
        f'基准={rule.base_value:.3f}, 目标/幅值={rule.target_value if rule.mode != ChangeMode.SINE else rule.amplitude:.3f}',
        fontsize=13,
        fontweight='bold'
    )
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, total_hours)
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    return fig


def plot_reliability_heatmap(
    heatmap_data: Dict,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    branch_ids = heatmap_data['branch_ids']
    failure_probs = heatmap_data['failure_probs']
    reliability_drops = np.array(heatmap_data['reliability_drops'])
    base_reliability = heatmap_data['base_reliability']

    max_drop = np.max(reliability_drops)

    im = ax.imshow(
        reliability_drops,
        cmap='Reds',
        aspect='auto',
        origin='lower',
        vmin=0,
        vmax=max_drop if max_drop > 0 else 0.01,
        extent=[
            min(failure_probs) - 0.01,
            max(failure_probs) + 0.01,
            -0.5,
            len(branch_ids) - 0.5
        ]
    )

    ax.set_yticks(range(len(branch_ids)))
    ax.set_yticklabels([f'分支 {bid}' for bid in branch_ids])

    xtick_labels = [f'{fp*100:.0f}%' for fp in failure_probs]
    ax.set_xticks(failure_probs)
    ax.set_xticklabels(xtick_labels)

    for i in range(len(branch_ids)):
        for j in range(len(failure_probs)):
            drop = reliability_drops[i, j]
            if drop >= max_drop * 0.3:
                text_color = 'white'
            else:
                text_color = 'black'
            ax.text(
                failure_probs[j],
                i,
                f'{drop*100:.1f}%',
                ha='center',
                va='center',
                color=text_color,
                fontsize=8
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('可靠度下降幅度', fontsize=12)

    ax.set_xlabel('故障概率', fontsize=12)
    ax.set_ylabel('分支编号', fontsize=12)
    ax.set_title(
        f'可靠度热力图 (基准可靠度: {base_reliability*100:.1f}%)\n'
        f'各分支单独故障时的系统可靠度下降幅度',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    return fig


def plot_airflow_distribution_histogram(
    failure_min_airflows: List[float],
    min_airflow_threshold: float = 4.0,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    arr = np.array(failure_min_airflows)

    n, bins, patches = ax.hist(
        arr,
        bins=20,
        edgecolor='black',
        alpha=0.7,
        color='skyblue'
    )

    for i, patch in enumerate(patches):
        if bins[i] < min_airflow_threshold:
            patch.set_facecolor('#ff6b6b')

    ax.axvline(
        x=min_airflow_threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'最低通风量阈值 ({min_airflow_threshold} m³/s)'
    )

    ax.set_xlabel('最小风量 (m³/s)', fontsize=12)
    ax.set_ylabel('频率', fontsize=12)
    ax.set_title(
        '系统失效时工作面最小风量统计分布\n'
        f'(失效场景数: {len(arr)})',
        fontsize=14,
        fontweight='bold'
    )

    stats_text = (
        f'均值: {np.mean(arr):.2f} m³/s\n'
        f'标准差: {np.std(arr):.2f} m³/s\n'
        f'5%分位数: {np.percentile(arr, 5):.2f} m³/s'
    )
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        verticalalignment='top',
        fontsize=10
    )

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    return fig


def plot_weak_branch_distribution(
    weak_branch_distribution: Dict[int, float],
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    branch_ids = sorted(weak_branch_distribution.keys())
    frequencies = [weak_branch_distribution[bid] for bid in branch_ids]

    colors = plt.cm.viridis(np.linspace(0, 1, len(branch_ids)))

    bars = ax.bar(
        range(len(branch_ids)),
        [f * 100 for f in frequencies],
        color=colors,
        edgecolor='black',
        alpha=0.8
    )

    ax.set_xlabel('分支编号', fontsize=12)
    ax.set_ylabel('出现频率 (%)', fontsize=12)
    ax.set_title(
        '最薄弱分支频率分布',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xticks(range(len(branch_ids)))
    ax.set_xticklabels([f'分支 {bid}' for bid in branch_ids], rotation=45, ha='right')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    return fig


def plot_critical_branches(
    critical_branches: List[Dict],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    branch_ids = [cb['branch_id'] for cb in critical_branches]
    reliability_drops = [cb['reliability_drop'] * 100 for cb in critical_branches]

    colors = ['#ff6b6b', '#ffa500', '#ffd93d'][:len(critical_branches)]

    bars = ax.barh(
        range(len(branch_ids)),
        reliability_drops,
        color=colors,
        edgecolor='black',
        alpha=0.8
    )

    ax.set_xlabel('可靠度下降幅度 (%)', fontsize=12)
    ax.set_ylabel('关键分支', fontsize=12)
    ax.set_title(
        '关键路径识别 - 前3条关键分支',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_yticks(range(len(branch_ids)))
    ax.set_yticklabels([f'分支 {bid}' for bid in branch_ids])
    ax.invert_yaxis()

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{width:.1f}%',
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    return fig


def plot_redundancy_greedy_curve(
    greedy_steps: List[Dict],
    target_reliability: float,
    figsize: Tuple[int, int] = (12, 7)
) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=figsize)

    steps = [s['step'] if isinstance(s, dict) else s.step for s in greedy_steps]
    costs = [s['cumulative_cost'] if isinstance(s, dict) else s.cumulative_cost for s in greedy_steps]
    reliabilities = [s['cumulative_reliability'] if isinstance(s, dict) else s.cumulative_reliability for s in greedy_steps]
    increments = [s['reliability_increment'] if isinstance(s, dict) else s.reliability_increment for s in greedy_steps]

    reliabilities_pct = [r * 100 for r in reliabilities]
    target_pct = target_reliability * 100

    color1 = '#1f77b4'
    color2 = '#ff7f0e'

    ax1.plot(
        costs,
        reliabilities_pct,
        'o-',
        color=color1,
        linewidth=2.5,
        markersize=10,
        label='系统可靠度',
        zorder=5
    )

    ax1.axhline(
        y=target_pct,
        color='#d62728',
        linestyle='--',
        linewidth=2,
        label=f'目标可靠度 ({target_pct:.1f}%)',
        zorder=3
    )

    for i in range(len(costs)):
        ax1.annotate(
            f'{reliabilities_pct[i]:.1f}%',
            (costs[i], reliabilities_pct[i]),
            textcoords="offset points",
            xytext=(0, 14),
            ha='center',
            fontsize=9,
            fontweight='bold',
            color=color1
        )
        ax1.annotate(
            f'步骤{steps[i]}',
            (costs[i], reliabilities_pct[i]),
            textcoords="offset points",
            xytext=(0, -22),
            ha='center',
            fontsize=8,
            color='gray'
        )

    ax1.fill_between(
        costs,
        reliabilities_pct,
        alpha=0.15,
        color=color1,
        zorder=2
    )

    ax1.set_xlabel('累计成本 (长度×断面积)', fontsize=12)
    ax1.set_ylabel('系统可靠度 (%)', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    bar_width = max(costs) * 0.03 if max(costs) > 0 else 1
    increments_pct = [inc * 100 for inc in increments]

    non_zero_mask = [i for i, inc in enumerate(increments_pct) if inc > 0]
    if non_zero_mask:
        filtered_costs = [costs[i] for i in non_zero_mask]
        filtered_incs = [increments_pct[i] for i in non_zero_mask]
        ax2.bar(
            filtered_costs,
            filtered_incs,
            width=bar_width,
            alpha=0.4,
            color=color2,
            label='该步可靠度增量',
            zorder=1
        )
    ax2.set_ylabel('本步可靠度增量 (%)', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)

    ax1.set_title(
        '贪心算法冗余路径组合优化 - 成本/可靠度收益曲线',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    ax1.grid(True, alpha=0.3, axis='both')
    ax1.set_xlim(left=min(costs) - max(costs) * 0.05 if max(costs) > 0 else -1)

    plt.tight_layout()

    return fig


def plot_network_with_redundancy(
    network: VentilationNetwork,
    recommended_branches: List[Dict],
    bottleneck_branch_ids: Optional[List[int]] = None,
    show_airflow: bool = True,
    show_pressure: bool = True,
    show_fan_icon: bool = True,
    figsize: Tuple[int, int] = (12, 9),
    node_size: int = 800,
    edge_width_range: Tuple[float, float] = (1.0, 6.0),
    cmap_name: str = 'viridis',
    seed: Optional[int] = 42
) -> plt.Figure:
    if bottleneck_branch_ids is None:
        bottleneck_branch_ids = []

    fig, ax = plt.subplots(figsize=figsize)

    G = nx.DiGraph()

    for node_id, node in network.nodes.items():
        G.add_node(node_id, elevation=node.elevation, pressure=node.pressure)

    edge_labels = {}
    edge_colors = []
    edge_widths = []
    edge_styles = []
    fan_edges = []
    bottleneck_edges = []
    redundancy_edges = []

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
        else:
            u, v = branch.to_node, branch.from_node

        G.add_edge(u, v, branch_id=branch_id, airflow=q)

        if branch_id in bottleneck_branch_ids:
            color = '#e74c3c'
            bottleneck_edges.append((u, v))

        edge_colors.append(color)
        edge_widths.append(width)
        edge_styles.append('solid')

        label_parts = [f'B{branch_id}']
        if show_airflow:
            label_parts.append(f'Q={abs_q:.2f}')
        edge_labels[(u, v)] = '\n'.join(label_parts)

        if branch.has_fan:
            fan_edges.append((u, v))

    rec_branch_infos = {}
    for idx, rec_branch in enumerate(recommended_branches, 1):
        u = rec_branch['from_node']
        v = rec_branch['to_node']
        rec_id = rec_branch.get('candidate_id', f'REC_{idx}')

        G.add_edge(u, v, branch_id=f'rec_{idx}', airflow=0, is_redundant=True)
        edge_colors.append('#2ecc71')
        edge_widths.append(3.0)
        edge_styles.append('dashed')
        redundancy_edges.append((u, v))

        label_parts = [f'冗余{idx}\n({rec_branch["from_node"]}→{rec_branch["to_node"]})']
        label_parts.append(f'L={rec_branch["length"]:.0f}m')
        label_parts.append(f'A={rec_branch["area"]:.2f}m²')
        edge_labels[(u, v)] = '\n'.join(label_parts)

    if seed is not None:
        pos = nx.spring_layout(G, seed=seed, k=0.5, iterations=50)
    else:
        pos = nx.spring_layout(G, k=0.5, iterations=50)

    solid_edges = [(u, v) for (u, v), style in zip(G.edges(), edge_styles) if style == 'solid']
    solid_colors = [c for c, s in zip(edge_colors, edge_styles) if s == 'solid']
    solid_widths = [w for w, s in zip(edge_widths, edge_styles) if s == 'solid']

    dashed_edges = [(u, v) for (u, v), style in zip(G.edges(), edge_styles) if style == 'dashed']
    dashed_colors = [c for c, s in zip(edge_colors, edge_styles) if s == 'dashed']
    dashed_widths = [w for w, s in zip(edge_widths, edge_styles) if s == 'dashed']

    nx.draw_networkx_edges(
        G, pos,
        edgelist=solid_edges,
        edge_color=solid_colors,
        width=solid_widths,
        arrowstyle='->',
        arrowsize=18,
        ax=ax,
        alpha=0.85
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=dashed_edges,
        edge_color=dashed_colors,
        width=dashed_widths,
        style='dashed',
        arrowstyle='->',
        arrowsize=20,
        ax=ax,
        alpha=0.95
    )

    for u, v in fan_edges:
        if show_fan_icon and (u, v) in G.edges():
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
            if node:
                node_colors.append(p_cmap(p_norm(node.pressure)))
            else:
                node_colors.append('lightblue')
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
        if node and show_pressure:
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
        font_size=8,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.9),
        ax=ax
    )

    if max_airflow > 0:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('风量 (m³/s)', fontsize=12)

    legend_patches = [
        mpatches.Patch(color='#e74c3c', label='瓶颈分支(标红)'),
        mpatches.Patch(color='#2ecc71', label='推荐冗余分支(虚线)'),
        mpatches.Patch(color='red', label='扇风机')
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    ax.set_title(
        '推荐冗余路径网络拓扑图\n(瓶颈分支标红 / 冗余分支绿色虚线)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('X坐标', fontsize=12)
    ax.set_ylabel('Y坐标', fontsize=12)
    ax.axis('off')
    ax.margins(0.1)

    plt.tight_layout()

    return fig

