from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from .network import VentilationNetwork
from .fan_operation import (
    calculate_all_fan_operating_points,
    calculate_total_power_consumption,
    check_fan_adequacy
)
from .resistance import calculate_branch_resistance


def detect_airflow_imbalance(
    network: VentilationNetwork,
    threshold_ratio: float = 3.0
) -> List[Dict]:
    issues = []
    branches = list(network.branches.values())

    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            b1, b2 = branches[i], branches[j]
            q1, q2 = abs(b1.airflow), abs(b2.airflow)

            if q1 > 1e-10 and q2 > 1e-10:
                ratio = max(q1, q2) / min(q1, q2)
                if ratio > threshold_ratio:
                    issues.append({
                        'type': 'airflow_imbalance',
                        'branch1_id': b1.id,
                        'branch2_id': b2.id,
                        'airflow1': b1.airflow,
                        'airflow2': b2.airflow,
                        'ratio': ratio,
                        'severity': 'high' if ratio > 5 else 'medium',
                        'suggestion': f'分支 {b1.id} ({q1:.2f} m³/s) 和分支 {b2.id} ({q2:.2f} m³/s) 风量比为 {ratio:.2f}:1，建议通过调节风门平衡风量'
                    })

    return issues


def detect_fan_efficiency_issues(network: VentilationNetwork) -> List[Dict]:
    issues = []
    fan_points = calculate_all_fan_operating_points(network)

    for branch_id, point in fan_points.items():
        if 'error' in point:
            continue

        if not point['in_efficiency_range'] and point['design_airflow']:
            operating_q = point['operating_airflow']
            design_q = point['design_airflow']
            ratio = operating_q / design_q if design_q > 0 else 0

            if ratio < 0.7:
                severity = 'high' if ratio < 0.5 else 'medium'
                issues.append({
                    'type': 'fan_underloaded',
                    'branch_id': branch_id,
                    'operating_airflow': operating_q,
                    'design_airflow': design_q,
                    'ratio': ratio,
                    'severity': severity,
                    'suggestion': f'扇风机工作在低风量区（{operating_q:.2f} m³/s，为设计风量的 {ratio*100:.1f}%），建议更换小型号风机或调整转速'
                })
            elif ratio > 1.1:
                severity = 'high' if ratio > 1.3 else 'medium'
                issues.append({
                    'type': 'fan_overloaded',
                    'branch_id': branch_id,
                    'operating_airflow': operating_q,
                    'design_airflow': design_q,
                    'ratio': ratio,
                    'severity': severity,
                    'suggestion': f'扇风机工作在大风量区（{operating_q:.2f} m³/s，为设计风量的 {ratio*100:.1f}%），建议更换大型号风机或增加并联风机'
                })

    return issues


def detect_high_resistance_branches(
    network: VentilationNetwork,
    threshold_pressure: float = 100.0
) -> List[Dict]:
    issues = []

    for branch_id, branch in network.branches.items():
        pressure_drop = abs(branch.pressure_drop)
        if pressure_drop > threshold_pressure and not branch.has_fan:
            r = branch.resistance
            if r == 0:
                r = calculate_branch_resistance(branch)

            velocity = network.get_air_velocity(branch_id)

            issues.append({
                'type': 'high_resistance',
                'branch_id': branch_id,
                'pressure_drop': pressure_drop,
                'resistance': r,
                'velocity': velocity,
                'severity': 'high' if pressure_drop > 300 else 'medium',
                'suggestion': f'分支 {branch_id} 阻力损失较大（{pressure_drop:.1f} Pa），建议优化巷道断面（当前 {branch.area:.2f} m²）或减少局部阻力'
            })

    return issues


def detect_low_velocity_branches(
    network: VentilationNetwork,
    threshold_velocity: float = 0.5
) -> List[Dict]:
    issues = []

    for branch_id, branch in network.branches.items():
        velocity = network.get_air_velocity(branch_id)
        if 0 < velocity < threshold_velocity and abs(branch.airflow) > 1e-10:
            issues.append({
                'type': 'low_velocity',
                'branch_id': branch_id,
                'velocity': velocity,
                'airflow': branch.airflow,
                'area': branch.area,
                'severity': 'low',
                'suggestion': f'分支 {branch_id} 风速偏低（{velocity:.2f} m/s），可能造成瓦斯积聚，建议缩小巷道断面或增加风量'
            })

    return issues


def detect_high_velocity_branches(
    network: VentilationNetwork,
    threshold_velocity: float = 6.0
) -> List[Dict]:
    issues = []

    for branch_id, branch in network.branches.items():
        velocity = network.get_air_velocity(branch_id)
        if velocity > threshold_velocity:
            issues.append({
                'type': 'high_velocity',
                'branch_id': branch_id,
                'velocity': velocity,
                'airflow': branch.airflow,
                'area': branch.area,
                'severity': 'high' if velocity > 8 else 'medium',
                'suggestion': f'分支 {branch_id} 风速偏高（{velocity:.2f} m/s），阻力和噪音较大，建议扩大巷道断面'
            })

    return issues


def suggest_damper_adjustments(network: VentilationNetwork) -> List[Dict]:
    suggestions = []
    damper_branches = network.get_damper_branches()

    airflows = [abs(b.airflow) for b in network.branches.values() if abs(b.airflow) > 1e-10]
    if not airflows:
        return suggestions

    mean_airflow = np.mean(airflows)

    for branch in damper_branches:
        q = abs(branch.airflow)
        if q > 1e-10:
            ratio = q / mean_airflow

            if ratio > 1.5:
                needed_increase = (ratio - 1) * 100
                suggestions.append({
                    'type': 'increase_damper',
                    'branch_id': branch.id,
                    'current_airflow': branch.airflow,
                    'mean_airflow': mean_airflow,
                    'ratio': ratio,
                    'suggested_resistance_increase': f'{needed_increase:.1f}%',
                    'suggestion': f'分支 {branch.id} 风量偏大（{q:.2f} m³/s，为均值的 {ratio*100:.1f}%），建议增加风门阻力约 {needed_increase:.1f}%'
                })
            elif ratio < 0.7:
                needed_decrease = (1 - ratio) * 100
                suggestions.append({
                    'type': 'decrease_damper',
                    'branch_id': branch.id,
                    'current_airflow': branch.airflow,
                    'mean_airflow': mean_airflow,
                    'ratio': ratio,
                    'suggested_resistance_decrease': f'{needed_decrease:.1f}%',
                    'suggestion': f'分支 {branch.id} 风量偏小（{q:.2f} m³/s，为均值的 {ratio*100:.1f}%），建议减小风门阻力约 {needed_decrease:.1f}%'
                })

    return suggestions


def generate_optimization_suggestions(
    network: VentilationNetwork,
    imbalance_threshold: float = 3.0,
    pressure_threshold: float = 100.0,
    velocity_low_threshold: float = 0.5,
    velocity_high_threshold: float = 6.0
) -> Dict:
    all_issues = []

    imbalance_issues = detect_airflow_imbalance(network, imbalance_threshold)
    all_issues.extend(imbalance_issues)

    fan_issues = detect_fan_efficiency_issues(network)
    all_issues.extend(fan_issues)

    resistance_issues = detect_high_resistance_branches(network, pressure_threshold)
    all_issues.extend(resistance_issues)

    low_vel_issues = detect_low_velocity_branches(network, velocity_low_threshold)
    all_issues.extend(low_vel_issues)

    high_vel_issues = detect_high_velocity_branches(network, velocity_high_threshold)
    all_issues.extend(high_vel_issues)

    damper_suggestions = suggest_damper_adjustments(network)
    all_issues.extend(damper_suggestions)

    fan_warnings = check_fan_adequacy(network)
    all_issues.extend(fan_warnings)

    power_info = calculate_total_power_consumption(network)

    severity_counts = {'high': 0, 'medium': 0, 'low': 0}
    for issue in all_issues:
        sev = issue.get('severity', 'low')
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    issues_by_type = {}
    for issue in all_issues:
        issue_type = issue.get('type', 'unknown')
        if issue_type not in issues_by_type:
            issues_by_type[issue_type] = []
        issues_by_type[issue_type].append(issue)

    suggestions_by_priority = sorted(
        all_issues,
        key=lambda x: {'high': 0, 'medium': 1, 'low': 2}.get(x.get('severity', 'low'), 3)
    )

    return {
        'total_issues': len(all_issues),
        'severity_counts': severity_counts,
        'issues_by_type': issues_by_type,
        'sorted_suggestions': suggestions_by_priority,
        'power_consumption': power_info,
        'summary': {
            'airflow_imbalance_count': len(imbalance_issues),
            'fan_efficiency_issues_count': len(fan_issues),
            'high_resistance_branches_count': len(resistance_issues),
            'velocity_issues_count': len(low_vel_issues) + len(high_vel_issues),
            'damper_adjustments_count': len(damper_suggestions),
            'fan_capacity_warnings_count': len(fan_warnings)
        }
    }


def generate_summary_report(suggestions: Dict) -> str:
    report = []
    report.append("=" * 60)
    report.append("矿井通风系统优化建议报告")
    report.append("=" * 60)
    report.append("")

    power = suggestions['power_consumption']
    report.append("一、系统能耗概况")
    report.append("-" * 60)
    report.append(f"  总轴功率: {power['total_shaft_power']:.2f} W")
    report.append(f"  总有效功率: {power['total_air_power']:.2f} W")
    report.append(f"  系统总效率: {power['total_efficiency']*100:.1f}%")
    report.append(f"  总风量: {power['total_airflow']:.2f} m³/s")
    report.append(f"  单位风量能耗: {power['specific_power']:.3f} W/(m³/s)")
    report.append("")

    report.append("二、问题统计")
    report.append("-" * 60)
    report.append(f"  严重问题: {suggestions['severity_counts'].get('high', 0)} 个")
    report.append(f"  中等问题: {suggestions['severity_counts'].get('medium', 0)} 个")
    report.append(f"  轻微问题: {suggestions['severity_counts'].get('low', 0)} 个")
    report.append("")

    report.append("三、详细建议（按优先级排序）")
    report.append("-" * 60)
    for i, issue in enumerate(suggestions['sorted_suggestions'], 1):
        severity = issue.get('severity', 'low').upper()
        suggestion = issue.get('suggestion', '无具体建议')
        report.append(f"  {i}. [{severity}] {suggestion}")
    report.append("")

    report.append("四、问题分类统计")
    report.append("-" * 60)
    for issue_type, issues in suggestions['issues_by_type'].items():
        report.append(f"  {issue_type}: {len(issues)} 个")
    report.append("")

    report.append("=" * 60)

    return "\n".join(report)
