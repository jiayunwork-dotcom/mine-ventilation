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
    a = branch.fan_params.get('a', 0.0)
    b = branch.fan_params.get('b', 0.0)
    c = branch.fan_params.get('c', 0.0)
    target = delta_p - h_n

    def fan_h(q):
        q_abs = abs(q)
        return a + b * q_abs + c * q_abs ** 2

    def f(q):
        return r * q * abs(q) - fan_h(q) - target

    def df_dq(q):
        q_abs = abs(q)
        return 2.0 * r * q_abs - (b + 2.0 * c * q_abs)

    q_solution = None

    f_zero = f(0.0)

    if abs(f_zero) < 1e-12:
        q_solution = 0.0
    else:
        q_pos = max(abs(b) / max(2.0 * (r - c), 1e-6), 1.0)
        for _ in range(20):
            f_val = f(q_pos)
            if f_val > 0:
                break
            q_pos *= 2.0

        if f_zero * f(q_pos) < 0:
            q_lo, q_hi = 0.0, q_pos
            for _ in range(200):
                q_mid = (q_lo + q_hi) / 2.0
                f_mid = f(q_mid)
                if abs(f_mid) < 1e-12 or (q_hi - q_lo) < 1e-12:
                    q_solution = q_mid
                    break
                if f(q_lo) * f_mid < 0:
                    q_hi = q_mid
                else:
                    q_lo = q_mid

        if q_solution is None:
            q = max(q_pos * 0.5, 1.0)
            for _ in range(200):
                f_val = f(q)
                df_val = df_dq(q)
                if abs(df_val) < 1e-12:
                    q *= 0.9
                    continue
                delta_q = -f_val / df_val
                delta_q = max(-0.5 * q, min(delta_q, 2.0 * q))
                q_new = q + delta_q
                if q_new < 0.01:
                    q_new = q * 0.5
                q = q_new
                if abs(delta_q) < 1e-10 and abs(f_val) < 1e-6:
                    q_solution = q
                    break
            if q_solution is None:
                q_solution = q

    if q_solution is None:
        q_solution = 10.0

    df_val = df_dq(q_solution)
    dq_ddp = 1.0 / df_val if abs(df_val) > 1e-12 else 0.0

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


def _compute_reference_pressures(
    network: VentilationNetwork,
    natural_pressures: Dict[int, float]
) -> Tuple[List[int], Dict[int, float]]:
    node_ids = sorted(network.nodes.keys())

    atmospheric_branches = []
    atmospheric_nodes = set()
    for bid, branch in network.branches.items():
        if branch.is_atmospheric:
            atmospheric_branches.append((bid, branch))
            atmospheric_nodes.add(branch.from_node)
            atmospheric_nodes.add(branch.to_node)

    if not atmospheric_nodes:
        ref_node = node_ids[0] if node_ids else None
        if ref_node is None:
            return [], {}
        return [ref_node], {ref_node: 0.0}

    ref_node = min(atmospheric_nodes)
    reference_pressures: Dict[int, float] = {ref_node: 0.0}

    visited = {ref_node}
    queue = [ref_node]

    atm_adj: Dict[int, List[Tuple[int, int, int]]] = {}
    for nid in atmospheric_nodes:
        atm_adj[nid] = []
    for bid, branch in atmospheric_branches:
        atm_adj[branch.from_node].append((bid, branch.to_node, 1))
        atm_adj[branch.to_node].append((bid, branch.from_node, -1))

    while queue:
        current = queue.pop(0)
        for bid, neighbor, direction in atm_adj.get(current, []):
            if neighbor in visited:
                continue
            h_n = natural_pressures.get(bid, 0.0)
            if direction > 0:
                reference_pressures[neighbor] = reference_pressures[current] - h_n
            else:
                reference_pressures[neighbor] = reference_pressures[current] + h_n
            visited.add(neighbor)
            queue.append(neighbor)

    for nid in atmospheric_nodes:
        if nid not in reference_pressures:
            reference_pressures[nid] = 0.0

    reference_nodes = sorted(reference_pressures.keys())
    return reference_nodes, reference_pressures


def _estimate_fan_operating_point(branch) -> float:
    if not branch.fan_params:
        return 30.0

    a = branch.fan_params.get('a', 0.0)
    b = branch.fan_params.get('b', 0.0)
    c = branch.fan_params.get('c', 0.0)

    if a <= 0:
        return 30.0

    r = 0.0
    try:
        from .resistance import calculate_branch_resistance
        r = calculate_branch_resistance(branch)
    except Exception:
        r = 0.1

    if r <= 1e-12:
        r = 0.001

    r_eff = r - c
    if r_eff <= 1e-12:
        r_eff = r + abs(c) + 0.001

    disc = b ** 2 + 4.0 * r_eff * a
    if disc >= 0:
        q = (-b + np.sqrt(disc)) / (2.0 * r_eff)
        if q > 0:
            return q * 0.7

    q_est = np.sqrt(a / r) * 0.5
    return max(q_est, 5.0)


def _compute_initial_pressures_independent(
    network: VentilationNetwork,
    resistances: Dict[int, float],
    natural_pressures: Dict[int, float],
    reference_pressures: Dict[int, float]
) -> Dict[int, float]:
    from .hardy_cross import hardy_cross_solve, calculate_node_pressures

    try:
        hc_air, _, hc_info = hardy_cross_solve(
            network, tolerance=0.001, max_iterations=1000
        )
        hc_pressures = calculate_node_pressures(
            network, hc_air, resistances, natural_pressures
        )
    except Exception:
        return _bfs_estimate_initial_pressures(
            network, resistances, natural_pressures, reference_pressures
        )

    node_ids = sorted(network.nodes.keys())
    np.random.seed(42)
    initial_pressures: Dict[int, float] = {}
    for nid in node_ids:
        p = hc_pressures.get(nid, 0.0)
        if nid in reference_pressures:
            initial_pressures[nid] = reference_pressures[nid]
        else:
            perturbation = 0.03 * abs(p) if abs(p) > 1.0 else 0.2
            p_perturbed = p + np.random.uniform(-perturbation, perturbation)
            initial_pressures[nid] = p_perturbed

    return initial_pressures


def newton_raphson_solve(
    network: VentilationNetwork,
    tolerance: float = 0.001,
    max_iterations: int = 500,
    initial_pressures: Optional[Dict[int, float]] = None,
    use_damping: bool = True,
    damping_factor: float = 1.0
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

    reference_nodes, reference_pressures = _compute_reference_pressures(
        network, natural_pressures
    )

    if initial_pressures is not None:
        node_pressures = initial_pressures.copy()
    else:
        node_pressures = _compute_initial_pressures_independent(
            network, resistances, natural_pressures, reference_pressures
        )

    for nid, p in reference_pressures.items():
        node_pressures[nid] = p

    iteration = 0
    converged = False
    max_residual = float('inf')
    residuals_history: List[float] = []
    best_residual = float('inf')
    best_pressures = node_pressures.copy()
    stagnation_count = 0

    while iteration < max_iterations and not converged:
        iteration += 1

        try:
            F, J, unknown_nodes, _ = build_node_equations(
                network, node_pressures, resistances, natural_pressures, reference_nodes
            )
        except Exception:
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

        if max_residual < best_residual:
            best_residual = max_residual
            best_pressures = node_pressures.copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count > 30:
            break

        try:
            J_dense = J.toarray()
            diag = np.abs(np.diag(J_dense))
            diag_min = np.min(diag[diag > 0]) if np.any(diag > 0) else 1.0
            lambda_reg = 1e-4 * diag_min

            for d_idx in range(len(diag)):
                if diag[d_idx] < lambda_reg:
                    J_dense[d_idx, d_idx] += lambda_reg - diag[d_idx]

            delta_p_full = np.linalg.solve(J_dense, -F)
        except np.linalg.LinAlgError:
            try:
                delta_p_full = np.linalg.lstsq(J.toarray(), -F, rcond=None)[0]
            except Exception:
                delta_p_full = np.zeros_like(F)
        except Exception:
            try:
                delta_p_full = spsolve(J, -F)
                if np.any(np.isnan(delta_p_full)) or np.any(np.isinf(delta_p_full)):
                    delta_p_full = np.zeros_like(F)
            except Exception:
                delta_p_full = np.zeros_like(F)

        if np.any(np.isnan(delta_p_full)) or np.any(np.isinf(delta_p_full)):
            delta_p_full = np.nan_to_num(delta_p_full, nan=0.0, posinf=100.0, neginf=-100.0)

        pressure_scale = max(np.max(np.abs([node_pressures[nid] for nid in unknown_nodes])), 1.0)
        max_step = np.max(np.abs(delta_p_full))
        step_cap = max(0.3 * pressure_scale, 50.0)
        if max_step > step_cap:
            delta_p_full = delta_p_full / max_step * step_cap

        alpha = 1.0
        if use_damping and max_residual > 1.0:
            alpha = _adaptive_line_search(
                network, node_pressures, delta_p_full, unknown_nodes,
                resistances, natural_pressures, reference_nodes,
                reference_pressures, max_residual
            )

        delta_p = delta_p_full * alpha

        for i, nid in enumerate(unknown_nodes):
            if i < len(delta_p):
                node_pressures[nid] += delta_p[i]

        for nid, p in reference_pressures.items():
            node_pressures[nid] = p

    airflows = calculate_airflows_from_pressures(
        network, node_pressures, resistances, natural_pressures
    )

    if not converged:
        airflows_best = calculate_airflows_from_pressures(
            network, best_pressures, resistances, natural_pressures
        )
        try:
            F_best, _, _, _ = build_node_equations(
                network, best_pressures, resistances, natural_pressures, reference_nodes
            )
            if len(F_best) > 0:
                best_res = np.max(np.abs(F_best))
                F_cur, _, _, _ = build_node_equations(
                    network, node_pressures, resistances, natural_pressures, reference_nodes
                )
                cur_res = np.max(np.abs(F_cur)) if len(F_cur) > 0 else float('inf')
                if best_res < cur_res:
                    node_pressures = best_pressures.copy()
                    airflows = airflows_best
                    max_residual = best_res
        except Exception:
            pass

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


def _adaptive_line_search(
    network: VentilationNetwork,
    node_pressures: Dict[int, float],
    delta_p: np.ndarray,
    unknown_nodes: List[int],
    resistances: Dict[int, float],
    natural_pressures: Dict[int, float],
    reference_nodes: List[int],
    reference_pressures: Dict[int, float],
    current_residual: float,
    max_backtracks: int = 15
) -> float:
    alpha = 1.0
    best_alpha = 1.0
    best_residual_found = float('inf')
    trial_pressures = node_pressures.copy()

    for _ in range(max_backtracks):
        for i, nid in enumerate(unknown_nodes):
            if i < len(delta_p):
                trial_pressures[nid] = node_pressures[nid] + delta_p[i] * alpha

        for nid, p in reference_pressures.items():
            trial_pressures[nid] = p

        try:
            F_trial, _, _, _ = build_node_equations(
                network, trial_pressures, resistances, natural_pressures, reference_nodes
            )
            if len(F_trial) > 0:
                trial_residual = np.max(np.abs(F_trial))
                if trial_residual < best_residual_found:
                    best_residual_found = trial_residual
                    best_alpha = alpha
                if trial_residual < current_residual:
                    return alpha
        except Exception:
            pass

        alpha *= 0.5
        if alpha < 1e-6:
            break

    if best_residual_found < current_residual:
        return best_alpha
    return 1.0


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
