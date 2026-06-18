from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque

from .network import VentilationNetwork
from .resistance import (
    calculate_all_branch_resistances,
    calculate_network_natural_pressures,
    calculate_branch_pressure_drop,
    calculate_pressure_drop,
    calculate_fan_pressure
)


def initialize_airflows(
    network: VentilationNetwork,
    initial_guess: Optional[Dict[int, float]] = None
) -> Dict[int, float]:
    if initial_guess is not None:
        return initial_guess.copy()

    airflows: Dict[int, float] = {bid: 0.0 for bid in network.branches.keys()}

    tree_branches, chord_branches = network.get_spanning_tree()
    
    fan_branches = [bid for bid, b in network.branches.items() if b.has_fan]
    for chord_id in chord_branches:
        if chord_id in fan_branches:
            airflows[chord_id] = 30.0
        else:
            airflows[chord_id] = 10.0
    
    nodes = list(network.nodes.keys())
    if not nodes:
        return airflows
    
    non_root_nodes = nodes[1:]
    
    n_nodes = len(non_root_nodes)
    n_tree = len(tree_branches)
    
    if n_nodes > 0 and n_tree > 0:
        A = np.zeros((n_nodes, n_tree))
        b = np.zeros(n_nodes)
        
        for row_idx, node_id in enumerate(non_root_nodes):
            for col_idx, tree_bid in enumerate(tree_branches):
                branch = network.get_branch(tree_bid)
                if branch.from_node == node_id:
                    A[row_idx, col_idx] = 1
                elif branch.to_node == node_id:
                    A[row_idx, col_idx] = -1
        
        for row_idx, node_id in enumerate(non_root_nodes):
            rhs = 0.0
            for chord_bid in chord_branches:
                branch = network.get_branch(chord_bid)
                q = airflows[chord_bid]
                if branch.from_node == node_id:
                    rhs -= q
                elif branch.to_node == node_id:
                    rhs += q
            b[row_idx] = rhs
        
        try:
            if n_nodes == n_tree:
                tree_flows = np.linalg.solve(A, b)
            else:
                tree_flows, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            for col_idx, tree_bid in enumerate(tree_branches):
                if col_idx < len(tree_flows):
                    airflows[tree_bid] = float(tree_flows[col_idx])
        except Exception as e:
            try:
                tree_flows, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                for col_idx, tree_bid in enumerate(tree_branches):
                    if col_idx < len(tree_flows):
                        airflows[tree_bid] = float(tree_flows[col_idx])
            except Exception as e2:
                print(f"线性方程组求解失败: {e}, {e2}")
                for tree_bid in tree_branches:
                    airflows[tree_bid] = 10.0
    
    undirected_adj: Dict[int, List[Tuple[int, int, int]]] = {}
    for nid in network.nodes:
        undirected_adj[nid] = []
    
    for bid, branch in network.branches.items():
        undirected_adj[branch.from_node].append((bid, branch.to_node, 1))
        undirected_adj[branch.to_node].append((bid, branch.from_node, -1))

    max_imbalance = 0.0
    for node_id in network.nodes.keys():
        net = 0.0
        for bid, neighbor, dir in undirected_adj[node_id]:
            net += airflows[bid] * dir
        max_imbalance = max(max_imbalance, abs(net))

    return airflows


def calculate_loop_correction(
    loop: List[int],
    airflows: Dict[int, float],
    resistances: Dict[int, float],
    network: VentilationNetwork,
    natural_pressures: Dict[int, float]
) -> Tuple[float, float, float]:
    numerator = 0.0
    denominator = 0.0
    loop_pressure_sum = 0.0

    for branch_ref in loop:
        branch_id = abs(branch_ref)
        direction = 1 if branch_ref > 0 else -1

        branch = network.get_branch(branch_id)
        if branch is None:
            continue

        q = airflows[branch_id] * direction
        r = resistances[branch_id]
        h_n = natural_pressures.get(branch_id, 0.0) * direction

        h_r = calculate_pressure_drop(r, q)

        h_fan = 0.0
        dh_fan_dq = 0.0
        if branch.has_fan and branch.fan_params:
            q_abs = abs(q)
            h_fan = calculate_fan_pressure(branch.fan_params, q_abs)
            if q < 0:
                h_fan = -h_fan
            if q_abs > 1e-10:
                dh_fan_dq = (branch.fan_params.get('b', 0.0) + 
                           2 * branch.fan_params.get('c', 0.0) * q_abs)
                if q < 0:
                    dh_fan_dq = -dh_fan_dq

        h_total = h_r - h_fan + h_n
        loop_pressure_sum += h_total

        dh_dq = 2 * r * abs(q) - dh_fan_dq

        numerator += h_total
        denominator += abs(dh_dq)

    if abs(denominator) < 1e-10:
        delta_q = 0.0
    else:
        delta_q = -numerator / denominator

    return delta_q, loop_pressure_sum, abs(numerator / denominator)


def hardy_cross_solve(
    network: VentilationNetwork,
    tolerance: float = 0.001,
    max_iterations: int = 1000,
    initial_guess: Optional[Dict[int, float]] = None,
    use_damping: bool = True,
    damping_factor: float = 0.8
) -> Tuple[Dict[int, float], Dict[int, float], Dict]:
    is_valid, errors = network.validate()
    if not is_valid:
        raise ValueError(f"网络验证失败: {errors}")

    resistances = calculate_all_branch_resistances(network)
    natural_pressures = calculate_network_natural_pressures(network)

    airflows = initialize_airflows(network, initial_guess)

    loops = network.find_independent_loops()

    if not loops:
        node_pressures = calculate_node_pressures(network, airflows, resistances, natural_pressures)
        return airflows, node_pressures, {
            'iterations': 0,
            'converged': True,
            'final_residual': 0.0,
            'loop_count': 0
        }

    iteration = 0
    converged = False
    max_correction = float('inf')
    residuals_history = []

    while iteration < max_iterations and not converged:
        iteration += 1
        max_delta = 0.0

        for loop in loops:
            delta_q, _, correction_mag = calculate_loop_correction(
                loop, airflows, resistances, network, natural_pressures
            )

            if use_damping:
                delta_q *= damping_factor

            max_delta = max(max_delta, abs(delta_q))

            for branch_ref in loop:
                branch_id = abs(branch_ref)
                direction = 1 if branch_ref > 0 else -1
                airflows[branch_id] += delta_q * direction

        max_correction = max_delta
        residuals_history.append(max_delta)

        if max_delta < tolerance:
            converged = True

    node_pressures = calculate_node_pressures(network, airflows, resistances, natural_pressures)

    for bid in airflows:
        if abs(airflows[bid]) < 1e-10:
            airflows[bid] = 0.0

    return airflows, node_pressures, {
        'iterations': iteration,
        'converged': converged,
        'final_residual': max_correction,
        'loop_count': len(loops),
        'residuals_history': residuals_history
    }


def calculate_node_pressures(
    network: VentilationNetwork,
    airflows: Dict[int, float],
    resistances: Dict[int, float],
    natural_pressures: Dict[int, float],
    reference_node: Optional[int] = None
) -> Dict[int, float]:
    atmospheric_nodes = set()
    for bid, branch in network.branches.items():
        if branch.is_atmospheric:
            atmospheric_nodes.add(branch.from_node)
            atmospheric_nodes.add(branch.to_node)
    
    if not atmospheric_nodes:
        if reference_node is None:
            node_ids = list(network.nodes.keys())
            if node_ids:
                reference_node = node_ids[0]
        if reference_node is not None:
            atmospheric_nodes.add(reference_node)

    ref_node = min(atmospheric_nodes) if atmospheric_nodes else reference_node
    if ref_node is None:
        node_ids = list(network.nodes.keys())
        ref_node = node_ids[0] if node_ids else None
        if ref_node is None:
            return {}

    node_ids = sorted(network.nodes.keys())
    pressures: Dict[int, float] = {nid: 0.0 for nid in node_ids}
    pressures[ref_node] = 0.0

    visited = {ref_node}
    queue = [ref_node]

    undirected_adj: Dict[int, List[Tuple[int, int, int]]] = {}
    for nid in network.nodes:
        undirected_adj[nid] = []
    for bid, branch in network.branches.items():
        undirected_adj[branch.from_node].append((bid, branch.to_node, 1))
        undirected_adj[branch.to_node].append((bid, branch.from_node, -1))

    while queue:
        current = queue.pop(0)
        for bid, neighbor, direction in undirected_adj[current]:
            if neighbor in visited:
                continue
            branch = network.get_branch(bid)
            if branch is None:
                continue

            q = airflows[bid]
            r = resistances[bid]
            h_n = natural_pressures.get(bid, 0.0)
            h_r = calculate_pressure_drop(r, q)
            h_fan = 0.0
            if branch.has_fan and branch.fan_params:
                q_abs = abs(q)
                h_fan = calculate_fan_pressure(branch.fan_params, q_abs)
                if q < 0:
                    h_fan = -h_fan

            C = h_r - h_fan + h_n

            if direction > 0:
                pressures[neighbor] = pressures[current] - C
            else:
                pressures[neighbor] = pressures[current] + C

            visited.add(neighbor)
            queue.append(neighbor)

    return pressures
