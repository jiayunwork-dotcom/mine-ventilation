from __future__ import annotations
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from core.network import VentilationNetwork
from core.fan_operation import (
    calculate_fan_operating_point,
    evaluate_fan_pressure,
    find_operating_point
)
from core.resistance import calculate_branch_resistance


def plot_fan_operating_point(
    network: VentilationNetwork,
    branch_id: int,
    q_range: Optional[Tuple[float, float]] = None,
    q_points: int = 200,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    branch = network.get_branch(branch_id)
    if branch is None:
        raise ValueError(f'分支 {branch_id} 不存在')

    if not branch.has_fan or branch.fan_params is None:
        raise ValueError(f'分支 {branch_id} 没有安装扇风机')

    r = branch.resistance
    if r == 0:
        r = calculate_branch_resistance(branch)

    fan_params = branch.fan_params
    design_q = fan_params.get('design_q', None)
    efficiency_range = fan_params.get('efficiency_range', (0.7, 1.1))

    operating_q, operating_h, found = find_operating_point(fan_params, r)

    if q_range is None:
        max_q = operating_q * 1.5 if operating_q and operating_q > 0 else 50.0
        q_range = (0, max_q)

    q_values = np.linspace(q_range[0], q_range[1], q_points)

    h_fan = np.array([evaluate_fan_pressure(fan_params, q) for q in q_values])
    h_system = r * q_values * q_values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(q_values, h_fan, 'b-', linewidth=3, label='扇风机特性曲线 H-Q')
    ax1.plot(q_values, h_system, 'r--', linewidth=2, label=f'系统阻力曲线 (R={r:.4f})')

    if found and operating_q is not None and operating_h is not None:
        ax1.plot(operating_q, operating_h, 'go', markersize=12, markeredgecolor='black', 
                markeredgewidth=2, label=f'工作点 (Q={operating_q:.2f}, H={operating_h:.2f})')

        ax1.vlines(x=operating_q, ymin=0, ymax=operating_h, colors='green', linestyles=':', alpha=0.7)
        ax1.hlines(y=operating_h, xmin=0, xmax=operating_q, colors='green', linestyles=':', alpha=0.7)

    if design_q:
        q_low = design_q * efficiency_range[0]
        q_high = design_q * efficiency_range[1]
        ax1.axvspan(q_low, q_high, alpha=0.2, color='green', label='高效工作区')

    current_q = abs(branch.airflow)
    current_h = evaluate_fan_pressure(fan_params, current_q)
    ax1.plot(current_q, current_h, 'ro', markersize=8, markeredgecolor='black',
            label=f'当前工况 (Q={current_q:.2f}, H={current_h:.2f})')

    ax1.set_xlabel('风量 Q (m³/s)', fontsize=12)
    ax1.set_ylabel('风压 H (Pa)', fontsize=12)
    ax1.set_title(f'分支 {branch_id} 扇风机工作点图', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(q_range)
    ax1.set_ylim(bottom=0)

    if found and operating_q is not None:
        shaft_power = 0.0
        static_efficiency = 0.0
        air_power = operating_h * operating_q
        efficiency = fan_params.get('efficiency', 0.75)
        if efficiency > 0:
            shaft_power = air_power / efficiency
            static_efficiency = (r * operating_q ** 3) / shaft_power if shaft_power > 0 else 0.0

        power_values = []
        for q in q_values:
            h = evaluate_fan_pressure(fan_params, q)
            ap = h * q
            sp = ap / efficiency if efficiency > 0 else 0
            power_values.append(sp)

        ax2.plot(q_values, power_values, 'purple', linewidth=2, label='轴功率')
        if found:
            ax2.plot(operating_q, shaft_power, 'go', markersize=10, markeredgecolor='black')
            ax2.vlines(x=operating_q, ymin=0, ymax=shaft_power, colors='green', linestyles=':', alpha=0.7)

        ax2.set_xlabel('风量 Q (m³/s)', fontsize=12)
        ax2.set_ylabel('功率 (W)', fontsize=12)
        ax2.set_title('扇风机功率曲线', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(q_range)
        ax2.set_ylim(bottom=0)

    info_text = []
    info_text.append(f'系统阻力系数 R = {r:.6f} Ns²/m⁸')
    if found and operating_q is not None and operating_h is not None:
        info_text.append(f'工作点: Q={operating_q:.2f} m³/s, H={operating_h:.2f} Pa')
        if design_q:
            ratio = operating_q / design_q
            info_text.append(f'设计风量: {design_q:.2f} m³/s (工作点为设计的 {ratio*100:.1f}%)')
        if shaft_power > 0:
            info_text.append(f'轴功率: {shaft_power:.2f} W')
            info_text.append(f'有效功率: {air_power:.2f} W')
            info_text.append(f'静压效率: {static_efficiency*100:.1f}%')
    else:
        info_text.append('警告: 未找到有效工作点，扇风机能力不足')

    fig.text(0.02, 0.02, '\n'.join(info_text), 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            fontsize=10, verticalalignment='bottom')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    return fig


def plot_multiple_fan_curves(
    fan_params_list: Dict[int, Dict],
    labels: Optional[Dict[int, str]] = None,
    q_range: Tuple[float, float] = (0, 100),
    q_points: int = 200,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    q_values = np.linspace(q_range[0], q_range[1], q_points)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fan_params_list)))

    for i, (fan_id, params) in enumerate(fan_params_list.items()):
        h_values = np.array([evaluate_fan_pressure(params, q) for q in q_values])
        label = labels.get(fan_id, f'扇风机 {fan_id}') if labels else f'扇风机 {fan_id}'
        ax.plot(q_values, h_values, color=colors[i], linewidth=2, label=label)

    ax.set_xlabel('风量 Q (m³/s)', fontsize=12)
    ax.set_ylabel('风压 H (Pa)', fontsize=12)
    ax.set_title('多台扇风机特性曲线对比', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(q_range)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    return fig


def plot_system_curve_comparison(
    network: VentilationNetwork,
    branch_ids: List[int],
    q_range: Tuple[float, float] = (0, 100),
    q_points: int = 200,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    q_values = np.linspace(q_range[0], q_range[1], q_points)
    colors = plt.cm.tab10(np.linspace(0, 1, len(branch_ids)))

    for i, bid in enumerate(branch_ids):
        branch = network.get_branch(bid)
        if branch is None:
            continue

        r = branch.resistance
        if r == 0:
            r = calculate_branch_resistance(branch)

        h_system = r * q_values * q_values
        ax.plot(q_values, h_system, color=colors[i], linewidth=2, 
               label=f'分支 {bid} (R={r:.6f})')

    ax.set_xlabel('风量 Q (m³/s)', fontsize=12)
    ax.set_ylabel('系统阻力 H (Pa)', fontsize=12)
    ax.set_title('多分支系统阻力曲线对比', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(q_range)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    return fig
