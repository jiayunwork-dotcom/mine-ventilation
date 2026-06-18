from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from .network import VentilationNetwork
from .resistance import (
    calculate_all_branch_resistances,
    calculate_network_natural_pressures,
    calculate_pressure_drop,
    calculate_fan_pressure
)


def calculate_branch_flow_and_derivatives(
    delta_p: float,
    r: float,
    h_n: float,
    branch=None
) -> Tuple[float, float]:
    """
    计算分支风量和风量对压差的导数
    
    压力平衡方程: p_from - p_to = h_r - h_fan + h_n
    即: h_r - h_fan = delta_p - h_n
    即: r * q * |q| - h_fan(q) = delta_p - h_n
    
    对于无扇风机分支: r * q * |q| = delta_p - h_n
    
    返回: (q, dq/ddelta_p)
    """
    if branch and branch.is_atmospheric:
        q = 10.0
        dq_ddp = 1e6
        return q, dq_ddp
    
    if branch and branch.has_fan and branch.fan_params:
        return _calculate_flow_with_fan(delta_p, r, h_n, branch)

    pressure_diff = delta_p - h_n

    if abs(pressure_diff) < 1e-12:
        return 0.0, 0.0

    if r <= 1e-12:
        sign = 1.0 if pressure_diff > 0 else -1.0
        q = sign * 100.0
        dq_ddp = 1e6
        return q, dq_ddp

    sign = 1.0 if pressure_diff > 0 else -1.0
    sqrt_term = np.sqrt(abs(pressure_diff) / r)
    q = sign * sqrt_term

    dq_ddp = sign / (2.0 * r * max(sqrt_term, 1e-12))

    return q, dq_ddp


def _calculate_flow_with_fan(
    delta_p: float,
    r: float,
    h_n: float,
    branch
) -> Tuple[float, float]:
    """
    对于有扇风机的分支，使用二分法求解q
    
    方程: f(q) = r * q * |q| - h_fan(q) - (delta_p - h_n) = 0
    
    扇风机的风压特性曲线: h_fan(q) = a + b*q + c*q^2 (q >= 0)
    当 q < 0 时，扇风机不提供风压，甚至会产生阻力
    
    我们假设扇风机主要在正风量区域工作，优先搜索正解
    """
    a = branch.fan_params.get('a', 0.0)
    b = branch.fan_params.get('b', 0.0)
    c = branch.fan_params.get('c', 0.0)
    target = delta_p - h_n

    def fan_h(q):
        q_abs = abs(q)
        h = a + b * q_abs + c * q_abs ** 2
        return h if q >= 0 else -h * 0.5

    def f(q):
        return r * q * abs(q) - fan_h(q) - target

    def df_dq(q):
        q_abs = abs(q)
        df_dr = 2 * r * abs(q)
        if q >= 0:
            df_dfan = -(b + 2 * c * q_abs)
        else:
            df_dfan = 0.5 * (b + 2 * c * q_abs)
        return df_dr + df_dfan

    q_solution = None

    q_max = 100.0
    for _ in range(5):
        f_pos = f(q_max)
        f_zero = f(0.0)
        if f_zero * f_pos < 0:
            q_low, q_high = 0.0, q_max
            for _ in range(100):
                q_mid = (q_low + q_high) / 2
                f_mid = f(q_mid)
                if abs(f_mid) < 1e-12 or (q_high - q_low) < 1e-12:
                    q_solution = q_mid
                    break
                if f(q_low) * f_mid < 0:
                    q_high = q_mid
                else:
                    q_low = q_mid
            break
        q_max *= 2

    if q_solution is None:
        q_max = 100.0
        for _ in range(5):
            f_neg = f(-q_max)
            f_zero = f(0.0)
            if f_zero * f_neg < 0:
                q_low, q_high = -q_max, 0.0
                for _ in range(100):
                    q_mid = (q_low + q_high) / 2
                    f_mid = f(q_mid)
                    if abs(f_mid) < 1e-12 or (q_high - q_low) < 1e-12:
                        q_solution = q_mid
                        break
                    if f(q_low) * f_mid < 0:
                        q_high = q_mid
                    else:
                        q_low = q_mid
                break
            q_max *= 2

    if q_solution is None:
        q = 10.0
        for _ in range(100):
            f_val = f(q)
            df_val = df_dq(q)
            if abs(df_val) < 1e-12:
                break
            delta_q = -f_val / df_val
            q_new = q + delta_q
            if q * q_new < 0:
                q = q * 0.5
            else:
                q = q_new
            if abs(delta_q) < 1e-10:
                break
        q_solution = q

    dq_ddp = 1.0 / df_dq(q_solution) if abs(df_dq(q_solution)) > 1e-12 else 0.0

    return q_solution, dq_ddp


def build_node_equations(
    network: VentilationNetwork,
    node_pressures: Dict[int, float],
    resistances: Dict[int, float],
    natural_pressures: Dict[int, float],
    reference_nodes: Optional[List[int]] = None
) -> Tuple[np.ndarray, csr_matrix, List[int], List[int]]:
    node_ids = sorted(network.nodes.keys())
    n_nodes = len(node_ids)
    
    if reference_nodes is None:
        reference_nodes = [node_ids[0]] if node_ids else []
    
    reference_set = set(reference_nodes)
    unknown_nodes = [nid for nid in node_ids if nid not in reference_set]
    n_unknowns = len(unknown_nodes)

    if n_unknowns <= 0:
        return np.array([]), csr_matrix((0, 0)), node_ids, reference_nodes

    node_to_unknown_idx = {nid: i for i, nid in enumerate(unknown_nodes)}

    F = np.zeros(n_unknowns)
    J = lil_matrix((n_unknowns, n_unknowns))

    for idx, node_id in enumerate(unknown_nodes):
        net_flow = 0.0
        adjacent = network.get_adjacent_branches(node_id)

        for branch_id, direction in adjacent:
            branch = network.get_branch(branch_id)
            if branch is None:
                continue
            
            if branch.is_atmospheric:
                continue

            p_from = node_pressures[branch.from_node]
            p_to = node_pressures[branch.to_node]
            delta_p = p_from - p_to

            h_n = natural_pressures.get(branch_id, 0.0)
            r = resistances[branch_id]

            q, dq_ddp = calculate_branch_flow_and_derivatives(delta_p, r, h_n, branch)

            if direction > 0:
                flow_contribution = -q
                dflow_dp_from = -dq_ddp
                dflow_dp_to = dq_ddp
            else:
                flow_contribution = q
                dflow_dp_from = dq_ddp
                dflow_dp_to = -dq_ddp

            net_flow += flow_contribution

            if branch.from_node == node_id:
                if branch.from_node not in reference_set:
                    J[idx, idx] += dflow_dp_from
                other_node = branch.to_node
                if other_node not in reference_set:
                    other_idx = node_to_unknown_idx[other_node]
                    J[idx, other_idx] += dflow_dp_to
            else:
                if branch.to_node not in reference_set:
                    J[idx, idx] += dflow_dp_to
                other_node = branch.from_node
                if other_node not in reference_set:
                    other_idx = node_to_unknown_idx[other_node]
                    J[idx, other_idx] += dflow_dp_from

        F[idx] = net_flow

    return F, csr_matrix(J), unknown_nodes, reference_nodes


def calculate_airflows_from_pressures(
    network: VentilationNetwork,
    node_pressures: Dict[int, float],
    resistances: Dict[int, float],
    natural_pressures: Dict[int, float]
) -> Dict[int, float]:
    airflows: Dict[int, float] = {}
    atmospheric_branches = []

    for branch_id, branch in network.branches.items():
        if branch.is_atmospheric:
            atmospheric_branches.append(branch_id)
            continue
            
        p_from = node_pressures[branch.from_node]
        p_to = node_pressures[branch.to_node]
        delta_p = p_from - p_to

        h_n = natural_pressures.get(branch_id, 0.0)
        r = resistances[branch_id]

        q, _ = calculate_branch_flow_and_derivatives(delta_p, r, h_n, branch)
        airflows[branch_id] = q

    undirected_adj: Dict[int, List[Tuple[int, int, int]]] = {}
    for nid in network.nodes:
        undirected_adj[nid] = []
    
    for bid, branch in network.branches.items():
        undirected_adj[branch.from_node].append((bid, branch.to_node, 1))
        undirected_adj[branch.to_node].append((bid, branch.from_node, -1))

    for atm_bid in atmospheric_branches:
        branch = network.get_branch(atm_bid)
        if branch is None:
            continue
            
        node_id = branch.from_node
        net_flow = 0.0
        
        for bid, neighbor, direction in undirected_adj[node_id]:
            if bid == atm_bid:
                continue
            if bid in airflows:
                q = airflows[bid]
                if direction > 0:
                    net_flow -= q
                else:
                    net_flow += q
        
        atm_branch = network.get_branch(atm_bid)
        if atm_branch.from_node == node_id:
            airflows[atm_bid] = net_flow
        else:
            airflows[atm_bid] = -net_flow

    return airflows


def estimate_initial_pressures(
    network: VentilationNetwork,
    resistances: Dict[int, float],
    natural_pressures: Dict[int, float]
) -> Dict[int, float]:
    """
    使用Hardy-Cross结果作为初始风压估计
    """
    from .hardy_cross import hardy_cross_solve

    try:
        airflows, pressures, info = hardy_cross_solve(
            network, tolerance=0.001, max_iterations=100
        )
        return pressures
    except Exception as e:
        print(f"使用Hardy-Cross初始化失败: {e}")
        pass

    node_ids = sorted(network.nodes.keys())
    pressures = {nid: 0.0 for nid in node_ids}

    ref_node = node_ids[0]
    visited = {ref_node}
    queue = [ref_node]

    while queue:
        current = queue.pop(0)
        adjacent = network.get_adjacent_branches(current)

        for branch_id, direction in adjacent:
            branch = network.get_branch(branch_id)
            if branch is None:
                continue

            other_node = branch.to_node if direction > 0 else branch.from_node

            if other_node not in visited:
                r = resistances[branch_id]
                h_n = natural_pressures.get(branch_id, 0.0)

                q_est = 20.0 if direction > 0 else -20.0

                if direction > 0:
                    delta_p_est = r * q_est * abs(q_est) + h_n
                    if branch.has_fan and branch.fan_params:
                        delta_p_est -= calculate_fan_pressure(branch.fan_params, abs(q_est))
                    pressures[other_node] = pressures[current] - delta_p_est
                else:
                    delta_p_est = r * q_est * abs(q_est) + h_n
                    if branch.has_fan and branch.fan_params:
                        delta_p_est -= calculate_fan_pressure(branch.fan_params, abs(q_est))
                    pressures[other_node] = pressures[current] + delta_p_est

                visited.add(other_node)
                queue.append(other_node)

    return pressures


def newton_raphson_solve(
    network: VentilationNetwork,
    tolerance: float = 0.001,
    max_iterations: int = 500,
    initial_pressures: Optional[Dict[int, float]] = None,
    use_damping: bool = True,
    damping_factor: float = 0.01
) -> Tuple[Dict[int, float], Dict[int, float], Dict]:
    is_valid, errors = network.validate()
    if not is_valid:
        raise ValueError(f"网络验证失败: {errors}")

    resistances = calculate_all_branch_resistances(network)
    natural_pressures = calculate_network_natural_pressures(network)

    node_ids = sorted(network.nodes.keys())
    if not node_ids:
        return {}, {}, {
            'iterations': 0,
            'converged': True,
            'final_residual': 0.0,
            'node_count': 0
        }

    atmospheric_nodes = set()
    for bid, branch in network.branches.items():
        if branch.is_atmospheric:
            atmospheric_nodes.add(branch.from_node)
            atmospheric_nodes.add(branch.to_node)

    if not atmospheric_nodes:
        atmospheric_nodes.add(node_ids[0])

    reference_nodes = sorted(atmospheric_nodes)
    reference_node = reference_nodes[0]

    if initial_pressures is not None:
        node_pressures = initial_pressures.copy()
    else:
        node_pressures = estimate_initial_pressures(network, resistances, natural_pressures)

    for nid in reference_nodes:
        node_pressures[nid] = 0.0

    iteration = 0
    converged = False
    max_residual = float('inf')
    residuals_history = []
    best_residual = float('inf')
    best_pressures = node_pressures.copy()
    no_improvement_count = 0
    
    n_unknowns = len(node_ids) - len(reference_nodes)
    if n_unknowns > 80:
        max_iterations = min(max_iterations, 30)
    elif n_unknowns > 50:
        max_iterations = min(max_iterations, 50)
    elif n_unknowns > 30:
        max_iterations = min(max_iterations, 100)

    while iteration < max_iterations and not converged:
        iteration += 1

        try:
            F, J, unknown_nodes, _ = build_node_equations(
                network, node_pressures, resistances, natural_pressures, reference_nodes
            )
        except Exception as e:
            residuals_history.append(max_residual)
            continue

        if len(F) == 0:
            converged = True
            max_residual = 0.0
            break

        max_residual = np.max(np.abs(F))
        residuals_history.append(max_residual)

        if max_residual < tolerance:
            converged = True
            break

        if max_residual < best_residual * 0.995:
            best_residual = max_residual
            best_pressures = node_pressures.copy()
            no_improvement_count = 0
            if damping_factor < 0.05:
                damping_factor = min(damping_factor * 1.02, 0.05)
        else:
            no_improvement_count += 1
            if no_improvement_count > 5:
                damping_factor = max(damping_factor * 0.7, 0.0005)
                no_improvement_count = 0

        if max_residual > best_residual * 1.2:
            node_pressures = best_pressures.copy()
            damping_factor = max(damping_factor * 0.5, 0.0005)
            continue

        try:
            delta_p = spsolve(J, -F)
        except Exception:
            try:
                delta_p = np.linalg.lstsq(J.toarray(), -F, rcond=None)[0]
            except Exception:
                delta_p = np.zeros_like(F)
                damping_factor = 0.001

        if np.any(np.isnan(delta_p)) or np.any(np.isinf(delta_p)):
            delta_p = np.nan_to_num(delta_p, nan=0.0, posinf=1e3, neginf=-1e3)
            damping_factor = 0.001

        max_delta = np.max(np.abs(delta_p))
        if max_delta > 1e4:
            delta_p = delta_p / max_delta * 1e4

        if use_damping:
            delta_p = delta_p * damping_factor

        for i, nid in enumerate(unknown_nodes):
            if i < len(delta_p):
                node_pressures[nid] += delta_p[i]

        for nid in reference_nodes:
            node_pressures[nid] = 0.0

        if iteration % 20 == 0 and damping_factor < 0.3:
            damping_factor = min(damping_factor * 1.2, 0.3)
        
        if iteration > 20 and max_residual > tolerance * 10:
            break

    if not converged:
        from .hardy_cross import hardy_cross_solve
        try:
            n_loops = network.get_independent_loops_count()
            hc_max_iter = max(1000, n_loops * 10)
            airflows_hc, pressures_hc, info_hc = hardy_cross_solve(
                network, tolerance=tolerance, max_iterations=hc_max_iter
            )
            airflows = airflows_hc
            node_pressures = pressures_hc
            max_residual = info_hc.get('final_residual', max_residual)
            converged = info_hc.get('converged', False)
        except Exception as e:
            print(f"回退到Hardy-Cross失败: {e}")
            airflows = calculate_airflows_from_pressures(
                network, node_pressures, resistances, natural_pressures
            )
    else:
        airflows = calculate_airflows_from_pressures(
            network, node_pressures, resistances, natural_pressures
        )

    for bid in airflows:
        if abs(airflows[bid]) < 1e-10:
            airflows[bid] = 0.0

    return airflows, node_pressures, {
        'iterations': iteration,
        'converged': converged,
        'final_residual': max_residual,
        'node_count': len(node_ids),
        'residuals_history': residuals_history,
        'reference_nodes': reference_nodes
    }


def compare_solutions(
    airflows1: Dict[int, float],
    airflows2: Dict[int, float],
    threshold: float = 0.005
) -> Tuple[bool, float, Dict[int, float]]:
    max_deviation = 0.0
    deviations = {}

    all_keys = set(airflows1.keys()) | set(airflows2.keys())

    for key in all_keys:
        q1 = airflows1.get(key, 0.0)
        q2 = airflows2.get(key, 0.0)

        if abs(q1) < 1e-10 and abs(q2) < 1e-10:
            deviation = 0.0
        elif abs(q1) < 1e-10:
            deviation = 1.0 if abs(q2) > 1e-10 else 0.0
        else:
            deviation = abs(q1 - q2) / max(abs(q1), 1e-10)

        deviations[key] = deviation
        max_deviation = max(max_deviation, deviation)

    is_consistent = max_deviation <= threshold
    return is_consistent, max_deviation, deviations
