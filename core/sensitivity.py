from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import copy

from .network import VentilationNetwork
from .hardy_cross import hardy_cross_solve
from .newton_raphson import newton_raphson_solve
from .resistance import calculate_branch_resistance


def sensitivity_analysis(
    network: VentilationNetwork,
    target_branch_id: int,
    resistance_range: Tuple[float, float] = (0.5, 1.5),
    step_size: float = 0.1,
    method: str = 'hardy_cross',
    tolerance: float = 0.001,
    key_branches: Optional[List[int]] = None
) -> Dict:
    if target_branch_id not in network.branches:
        return {'error': f'目标分支 {target_branch_id} 不存在'}

    target_branch = network.get_branch(target_branch_id)
    original_resistance = calculate_branch_resistance(target_branch)

    if key_branches is None:
        key_branches = list(network.branches.keys())

    resistance_factors = np.arange(resistance_range[0], resistance_range[1] + step_size / 2, step_size)

    results = {
        'target_branch_id': target_branch_id,
        'original_resistance': original_resistance,
        'resistance_factors': resistance_factors.tolist(),
        'resistance_values': (original_resistance * resistance_factors).tolist(),
        'key_branches': key_branches,
        'airflow_data': {},
        'sensitivity_indices': {}
    }

    original_airflows = network.get_branch_airflows()

    for bid in key_branches:
        results['airflow_data'][bid] = []

    for factor in resistance_factors:
        test_network = copy.deepcopy(network)
        test_branch = test_network.get_branch(target_branch_id)

        base_r = (test_branch.friction_coeff * test_branch.length * test_branch.perimeter / 
                 (test_branch.area ** 3) + test_branch.local_coeff / (2 * test_branch.area ** 2))
        
        new_damper = base_r * (factor - 1) + test_branch.damper_resistance
        test_branch.damper_resistance = max(0, new_damper)

        if method == 'hardy_cross':
            airflows, pressures, info = hardy_cross_solve(
                test_network, tolerance=tolerance
            )
        else:
            airflows, pressures, info = newton_raphson_solve(
                test_network, tolerance=tolerance
            )

        for bid in key_branches:
            results['airflow_data'][bid].append(airflows.get(bid, 0.0))

    for bid in key_branches:
        airflows = np.array(results['airflow_data'][bid])
        q0 = original_airflows.get(bid, 0.0)

        if abs(q0) > 1e-10:
            relative_changes = (airflows - q0) / abs(q0)
        else:
            relative_changes = np.zeros_like(airflows)

        resistance_changes = resistance_factors - 1.0

        valid_indices = np.abs(resistance_changes) > 1e-10
        if np.any(valid_indices):
            sensitivity = relative_changes[valid_indices] / resistance_changes[valid_indices]
            avg_sensitivity = np.mean(np.abs(sensitivity))
        else:
            avg_sensitivity = 0.0

        results['sensitivity_indices'][bid] = {
            'average_absolute_sensitivity': avg_sensitivity,
            'max_relative_change': np.max(np.abs(relative_changes)),
            'min_relative_change': np.min(relative_changes),
            'original_airflow': q0,
            'airflow_range': [float(np.min(airflows)), float(np.max(airflows))]
        }

    return results


def short_circuit_analysis(
    network: VentilationNetwork,
    target_branch_id: int,
    method: str = 'hardy_cross',
    tolerance: float = 0.001
) -> Dict:
    if target_branch_id not in network.branches:
        return {'error': f'目标分支 {target_branch_id} 不存在'}

    original_airflows = network.get_branch_airflows()
    original_pressures = network.get_node_pressures()

    test_network = copy.deepcopy(network)
    target_branch = test_network.get_branch(target_branch_id)

    target_branch.friction_coeff = 1e-10
    target_branch.local_coeff = 0.0
    target_branch.damper_resistance = 0.0

    if method == 'hardy_cross':
        airflows, pressures, info = hardy_cross_solve(
            test_network, tolerance=tolerance
        )
    else:
        airflows, pressures, info = newton_raphson_solve(
            test_network, tolerance=tolerance
        )

    results = {
        'target_branch_id': target_branch_id,
        'original_airflows': original_airflows,
        'short_circuit_airflows': airflows,
        'original_pressures': original_pressures,
        'short_circuit_pressures': pressures,
        'airflow_changes': {},
        'pressure_changes': {},
        'affected_branches': [],
        'total_airflow_change': 0.0
    }

    total_orig = 0.0
    total_new = 0.0

    for bid in network.branches:
        q_orig = original_airflows.get(bid, 0.0)
        q_new = airflows.get(bid, 0.0)

        total_orig += abs(q_orig)
        total_new += abs(q_new)

        if abs(q_orig) > 1e-10:
            change_percent = ((q_new - q_orig) / abs(q_orig)) * 100
        else:
            change_percent = 100.0 if abs(q_new) > 1e-10 else 0.0

        results['airflow_changes'][bid] = {
            'original': q_orig,
            'short_circuit': q_new,
            'absolute_change': q_new - q_orig,
            'relative_change': (q_new - q_orig) / abs(q_orig) if abs(q_orig) > 1e-10 else (1.0 if q_new != 0 else 0.0),
            'percent_change': change_percent
        }

        if abs(change_percent) > 5:
            results['affected_branches'].append(bid)

    for nid in network.nodes:
        p_orig = original_pressures.get(nid, 0.0)
        p_new = pressures.get(nid, 0.0)
        results['pressure_changes'][nid] = {
            'original': p_orig,
            'short_circuit': p_new,
            'change': p_new - p_orig
        }

    results['total_airflow_change'] = total_new - total_orig
    results['total_airflow_change_percent'] = ((total_new - total_orig) / total_orig * 100) if total_orig > 0 else 0

    return results


def multiple_parameter_sensitivity(
    network: VentilationNetwork,
    parameters: List[Dict],
    method: str = 'hardy_cross',
    tolerance: float = 0.001
) -> Dict:
    results = {
        'parameters': parameters,
        'baseline': {
            'airflows': network.get_branch_airflows(),
            'pressures': network.get_node_pressures()
        },
        'scenarios': []
    }

    for param in parameters:
        target_branch_id = param.get('branch_id')
        param_type = param.get('type', 'resistance')
        factor = param.get('factor', 1.0)

        if target_branch_id not in network.branches:
            continue

        test_network = copy.deepcopy(network)
        branch = test_network.get_branch(target_branch_id)

        if param_type == 'resistance':
            base_r = (branch.friction_coeff * branch.length * branch.perimeter / 
                     (branch.area ** 3) + branch.local_coeff / (2 * branch.area ** 2))
            new_damper = base_r * (factor - 1) + branch.damper_resistance
            branch.damper_resistance = max(0, new_damper)
        elif param_type == 'area':
            branch.area *= factor
        elif param_type == 'fan_speed':
            if branch.has_fan and branch.fan_params:
                branch.fan_params['a'] *= factor ** 2
                branch.fan_params['b'] *= factor
        elif param_type == 'temperature':
            node_id = param.get('node_id')
            if node_id in test_network.nodes:
                test_network.nodes[node_id].temperature *= factor

        if method == 'hardy_cross':
            airflows, pressures, info = hardy_cross_solve(
                test_network, tolerance=tolerance
            )
        else:
            airflows, pressures, info = newton_raphson_solve(
                test_network, tolerance=tolerance
            )

        scenario_result = {
            'parameter': param,
            'airflows': airflows,
            'pressures': pressures,
            'solver_info': info
        }
        results['scenarios'].append(scenario_result)

    return results
