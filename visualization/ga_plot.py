from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from core.genetic_optimization import GAOptimizationResult, GenerationHistory, GAParameters


def plot_ga_convergence_curve(
    history: List[GenerationHistory],
    figsize: Tuple[int, int] = (12, 7)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    generations = [h.generation for h in history]
    best_fitness = [h.best_fitness for h in history]
    avg_fitness = [h.avg_fitness for h in history]
    worst_fitness = [h.worst_fitness for h in history]

    ax.plot(generations, best_fitness, 'b-o', linewidth=2.5, markersize=6, label='最优适应度', zorder=5)
    ax.plot(generations, avg_fitness, 'g-^', linewidth=2, markersize=5, label='平均适应度', alpha=0.8)
    ax.plot(generations, worst_fitness, 'r-s', linewidth=1.5, markersize=4, label='最差适应度', alpha=0.6)

    ax.fill_between(generations, best_fitness, worst_fitness, alpha=0.1, color='gray', label='适应度范围')

    ax.set_xlabel('代数 (Generation)', fontsize=13)
    ax.set_ylabel('适应度值 (总功率+惩罚项，W)', fontsize=13)
    ax.set_title('遗传算法收敛曲线', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=11)

    if history:
        final_best = best_fitness[-1]
        initial_best = best_fitness[0] if len(best_fitness) > 0 else 0
        improvement = 0.0
        if initial_best > 0:
            improvement = (initial_best - final_best) / initial_best * 100.0

        text_str = (
            f'代数: {len(history)}\n'
            f'最终最优: {final_best:.1f} W\n'
            f'改善幅度: {improvement:.1f}%'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, zorder=10)

    plt.tight_layout()
    return fig


def plot_ga_radar_chart(
    result: GAOptimizationResult,
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    params = result.parameters
    fan_ids = sorted(result.fan_speeds.keys())
    damper_ids = sorted(result.damper_openings.keys())

    labels = []
    values = []
    normalized_values = []

    for fid in fan_ids:
        speed = result.fan_speeds[fid]
        labels.append(f'风机{int(fid)}\n转速')
        values.append(speed)
        norm = (speed - params.fan_speed_min) / (params.fan_speed_max - params.fan_speed_min)
        normalized_values.append(norm)

    for did in damper_ids:
        opening = result.damper_openings[did]
        labels.append(f'风门{int(did)}\n开度')
        values.append(opening)
        norm = (opening - params.damper_open_min) / (params.damper_open_max - params.damper_open_min)
        normalized_values.append(norm)

    n_vars = len(labels)
    if n_vars == 0:
        ax.text(0.5, 0.5, '无决策变量', ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        return fig

    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    normalized_values += normalized_values[:1]
    angles += angles[:1]

    ax.plot(angles, normalized_values, 'b-o', linewidth=2.5, markersize=8,
            label='最优方案', zorder=5)
    ax.fill(angles, normalized_values, alpha=0.25, color='skyblue', zorder=3)

    reference_angles = angles.copy()
    reference_values = [1.0] * (n_vars + 1)
    ax.plot(reference_angles, reference_values, 'r--', linewidth=1.5, alpha=0.6,
            label='基准方案(全开/额定)', zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)
    ax.set_ylabel('归一化值', fontsize=12, labelpad=20)

    ax.set_title('决策变量雷达图\n(归一化显示各风机转速和风门开度)',
                 fontsize=15, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10)

    ax.grid(True, alpha=0.4, linestyle='-')

    plt.tight_layout()
    return fig


def plot_ga_airflow_comparison(
    result: GAOptimizationResult,
    network,
    workface_ids: List[int],
    figsize: Tuple[int, int] = (14, 7)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    params = result.parameters
    n_workfaces = len(workface_ids)

    if n_workfaces == 0:
        ax.text(0.5, 0.5, '无工作面分支', ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        return fig

    x = np.arange(n_workfaces)
    width = 0.35

    initial_airflows = []
    optimized_airflows = []
    satisfied_list = []
    labels = []

    for wf_id in sorted(workface_ids):
        optimized_q = result.workface_airflows.get(wf_id, 0.0)
        branch = network.get_branch(wf_id)
        initial_q = abs(branch.airflow) if branch else 0.0

        initial_airflows.append(initial_q)
        optimized_airflows.append(optimized_q)
        satisfied = result.constraint_satisfied.get(wf_id, False)
        satisfied_list.append(satisfied)
        labels.append(f'分支{wf_id}')

    bars1 = ax.bar(x - width/2, initial_airflows, width, label='优化前(初始方案)',
                   color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.8, zorder=3)

    colors_opt = ['#2ecc71' if s else '#e74c3c' for s in satisfied_list]
    bars2 = ax.bar(x + width/2, optimized_airflows, width, label='优化后(遗传算法)',
                   color=colors_opt, alpha=0.85, edgecolor='black', linewidth=0.8, zorder=3)

    ax.axhline(y=params.min_airflow_threshold, color='#d35400', linestyle='--',
               linewidth=2.5, label=f'最低通风量阈值 ({params.min_airflow_threshold} m³/s)',
               zorder=4)

    for i, (bar, q, s) in enumerate(zip(bars1, initial_airflows, satisfied_list)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{q:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for i, (bar, q, s) in enumerate(zip(bars2, optimized_airflows, satisfied_list)):
        height = bar.get_height()
        status = '✓' if s else '✗'
        color = 'green' if s else 'red'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{q:.2f} {status}', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=color)

    ax.set_xlabel('工作面分支', fontsize=13)
    ax.set_ylabel('风量 (m³/s)', fontsize=13)
    ax.set_title('优化前后各工作面风量对比\n(绿色=满足约束，红色=不满足约束)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(axis='both', labelsize=11)

    all_satisfied = all(satisfied_list)
    n_satisfied = sum(satisfied_list)
    status_text = (
        f'工作面数: {n_workfaces}\n'
        f'满足约束: {n_satisfied}/{n_workfaces}\n'
        f'状态: {"全部满足 ✓" if all_satisfied else "部分不满足 ✗"}'
    )
    props = dict(boxstyle='round', facecolor='lightgreen' if all_satisfied else '#ffcccc', alpha=0.85)
    ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, zorder=5)

    plt.tight_layout()
    return fig


def plot_ga_decision_variables_bar(
    result: GAOptimizationResult,
    figsize: Tuple[int, int] = (14, 7)
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    fan_ids = sorted(result.fan_speeds.keys())
    damper_ids = sorted(result.damper_openings.keys())
    params = result.parameters

    if fan_ids:
        fan_speeds = [result.fan_speeds[fid] for fid in fan_ids]
        fan_labels = [f'风机{fid}' for fid in fan_ids]

        colors_fan = []
        for s in fan_speeds:
            if abs(s - 1.0) < 0.01:
                colors_fan.append('#3498db')
            elif s > 1.0:
                colors_fan.append('#e67e22')
            else:
                colors_fan.append('#27ae60')

        bars_f = ax1.bar(range(len(fan_ids)), fan_speeds,
                         color=colors_fan, alpha=0.85, edgecolor='black', linewidth=0.8)
        ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='额定转速(1.0)')
        ax1.axhline(y=params.fan_speed_min, color='red', linestyle=':', linewidth=1.2, alpha=0.6, label=f'下限({params.fan_speed_min})')
        ax1.axhline(y=params.fan_speed_max, color='red', linestyle=':', linewidth=1.2, alpha=0.6, label=f'上限({params.fan_speed_max})')

        for i, (bar, s) in enumerate(zip(bars_f, fan_speeds)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{s:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax1.set_xlabel('扇风机', fontsize=12)
        ax1.set_ylabel('转速系数 (倍数)', fontsize=12)
        ax1.set_title('各扇风机最优转速系数', fontsize=13, fontweight='bold', pad=10)
        ax1.set_xticks(range(len(fan_ids)))
        ax1.set_xticklabels(fan_labels, fontsize=10)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_ylim(0, params.fan_speed_max * 1.15)
    else:
        ax1.text(0.5, 0.5, '网络中无扇风机', ha='center', va='center', fontsize=12,
                transform=ax1.transAxes)
        ax1.set_title('扇风机转速', fontsize=13, fontweight='bold')

    if damper_ids:
        damper_openings = [result.damper_openings[did] for did in damper_ids]
        damper_labels = [f'风门{did}' for did in damper_ids]

        colors_damper = []
        for o in damper_openings:
            if o >= 0.9:
                colors_damper.append('#27ae60')
            elif o >= 0.5:
                colors_damper.append('#f1c40f')
            else:
                colors_damper.append('#e74c3c')

        bars_d = ax2.bar(range(len(damper_ids)), damper_openings,
                         color=colors_damper, alpha=0.85, edgecolor='black', linewidth=0.8)
        ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='全开(1.0)')
        ax2.axhline(y=0.0, color='red', linestyle=':', linewidth=1.2, alpha=0.6, label='全关(0.0)')

        for i, (bar, o) in enumerate(zip(bars_d, damper_openings)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{o:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax2.set_xlabel('调节风门', fontsize=12)
        ax2.set_ylabel('开度系数 (0=全关, 1=全开)', fontsize=12)
        ax2.set_title('各调节风门最优开度', fontsize=13, fontweight='bold', pad=10)
        ax2.set_xticks(range(len(damper_ids)))
        ax2.set_xticklabels(damper_labels, fontsize=10)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_ylim(-0.05, 1.15)
    else:
        ax2.text(0.5, 0.5, '网络中无调节风门', ha='center', va='center', fontsize=12,
                transform=ax2.transAxes)
        ax2.set_title('调节风门开度', fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_ga_power_comparison(
    result: GAOptimizationResult,
    figsize: Tuple[int, int] = (10, 7)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    scenarios = ['初始方案\n(额定转速/全开)', '优化方案\n(遗传算法)']
    powers = [result.initial_power, result.best_power]

    colors = ['#3498db', '#2ecc71']
    bars = ax.bar(scenarios, powers, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=1.2, width=0.5)

    for bar, p in zip(bars, powers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(powers)*0.01,
                f'{p:.2f} W', ha='center', va='bottom', fontsize=13, fontweight='bold')

    if result.initial_power > 0 and result.energy_saving_percent > 0:
        saving_w = result.initial_power - result.best_power
        ax.annotate(
            f'节省 {saving_w:.2f} W\n({result.energy_saving_percent:.1f}%)',
            xy=(1, result.best_power), xytext=(1.35, (result.initial_power + result.best_power)/2),
            fontsize=12, fontweight='bold', color='#27ae60',
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2),
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#d5f5e3', alpha=0.9)
        )

    ax.set_ylabel('系统总轴功率 (W)', fontsize=13)
    ax.set_title('优化前后系统能耗对比', fontsize=16, fontweight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    ylim_top = max(powers) * 1.25 if powers else 1
    ax.set_ylim(0, ylim_top)

    summary_text = (
        f'初始功率: {result.initial_power:.2f} W\n'
        f'优化功率: {result.best_power:.2f} W\n'
        f'节能率: {result.energy_saving_percent:.1f}%'
    )
    props = dict(boxstyle='round', facecolor='#d5f5e3' if result.energy_saving_percent > 0 else '#ffebee', alpha=0.9)
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    return fig
