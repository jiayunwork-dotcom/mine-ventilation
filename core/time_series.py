from __future__ import annotations
import copy
import csv
import io
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .network import VentilationNetwork, Branch
from .resistance import (
    calculate_all_branch_resistances,
    calculate_network_natural_pressures,
    update_branch_pressure_drops,
)
from .hardy_cross import hardy_cross_solve


class ChangeMode(Enum):
    STEP = "step"
    LINEAR = "linear"
    SINE = "sine"


class ParameterType(Enum):
    RESISTANCE = "resistance"
    FAN_SPEED = "fan_speed"


@dataclass
class ChangeRule:
    id: str
    branch_id: int
    parameter_type: ParameterType
    mode: ChangeMode
    base_value: float = 1.0
    target_value: float = 1.0
    start_time: float = 0.0
    end_time: float = 24.0
    period: float = 24.0
    amplitude: float = 0.1
    phase: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "branch_id": self.branch_id,
            "parameter_type": self.parameter_type.value,
            "mode": self.mode.value,
            "base_value": self.base_value,
            "target_value": self.target_value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "period": self.period,
            "amplitude": self.amplitude,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ChangeRule":
        return cls(
            id=data["id"],
            branch_id=data["branch_id"],
            parameter_type=ParameterType(data["parameter_type"]),
            mode=ChangeMode(data["mode"]),
            base_value=data.get("base_value", 1.0),
            target_value=data.get("target_value", 1.0),
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 24.0),
            period=data.get("period", 24.0),
            amplitude=data.get("amplitude", 0.1),
            phase=data.get("phase", 0.0),
        )


@dataclass
class TimeSeriesResult:
    timestamps: List[float] = field(default_factory=list)
    branch_airflows: Dict[int, List[float]] = field(default_factory=dict)
    node_pressures: Dict[int, List[float]] = field(default_factory=dict)
    branch_resistances: Dict[int, List[float]] = field(default_factory=dict)
    solver_infos: List[Dict] = field(default_factory=list)
    branch_ids: List[int] = field(default_factory=list)
    node_ids: List[int] = field(default_factory=list)

    def to_dataframe_dict(self) -> Dict:
        data = {"timestamp_h": self.timestamps}
        for bid in self.branch_ids:
            data[f"branch_{bid}_airflow_m3s"] = self.branch_airflows.get(bid, [])
            data[f"branch_{bid}_resistance"] = self.branch_resistances.get(bid, [])
        for nid in self.node_ids:
            data[f"node_{nid}_pressure_pa"] = self.node_pressures.get(nid, [])
        return data

    def to_csv(self) -> str:
        output = io.StringIO()
        max_len = len(self.timestamps)

        fieldnames = ["timestamp_h"]
        for bid in self.branch_ids:
            fieldnames.append(f"branch_{bid}_airflow_m3s")
            fieldnames.append(f"branch_{bid}_resistance")
        for nid in self.node_ids:
            fieldnames.append(f"node_{nid}_pressure_pa")

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(max_len):
            row = {"timestamp_h": f"{self.timestamps[i]:.4f}"}
            for bid in self.branch_ids:
                af_list = self.branch_airflows.get(bid, [])
                r_list = self.branch_resistances.get(bid, [])
                row[f"branch_{bid}_airflow_m3s"] = f"{af_list[i]:.6f}" if i < len(af_list) else ""
                row[f"branch_{bid}_resistance"] = f"{r_list[i]:.8f}" if i < len(r_list) else ""
            for nid in self.node_ids:
                p_list = self.node_pressures.get(nid, [])
                row[f"node_{nid}_pressure_pa"] = f"{p_list[i]:.4f}" if i < len(p_list) else ""
            writer.writerow(row)

        return output.getvalue()


def compute_parameter_factor(rule: ChangeRule, current_time: float) -> float:
    if rule.mode == ChangeMode.STEP:
        if current_time < rule.start_time:
            return rule.base_value
        else:
            return rule.target_value

    elif rule.mode == ChangeMode.LINEAR:
        if current_time <= rule.start_time:
            return rule.base_value
        elif current_time >= rule.end_time:
            return rule.target_value
        else:
            t_ratio = (current_time - rule.start_time) / (rule.end_time - rule.start_time)
            return rule.base_value + (rule.target_value - rule.base_value) * t_ratio

    elif rule.mode == ChangeMode.SINE:
        t = current_time - rule.phase
        sine_val = math.sin(2 * math.pi * t / rule.period)
        return rule.base_value + rule.amplitude * sine_val

    return rule.base_value


def apply_rules_to_network(
    network: VentilationNetwork,
    rules: List[ChangeRule],
    current_time: float,
    base_branch_params: Dict[int, Dict]
) -> Dict[int, float]:
    applied_factors: Dict[int, float] = {}

    for rule in rules:
        factor = compute_parameter_factor(rule, current_time)
        applied_factors[rule.id] = factor

        branch = network.get_branch(rule.branch_id)
        if branch is None:
            continue

        base_params = base_branch_params.get(rule.branch_id, {})

        if rule.parameter_type == ParameterType.RESISTANCE:
            base_friction = base_params.get("friction_coeff", branch.friction_coeff)
            base_local = base_params.get("local_coeff", branch.local_coeff)
            base_damper = base_params.get("damper_resistance", branch.damper_resistance)

            branch.friction_coeff = base_friction * factor
            branch.local_coeff = base_local * factor
            branch.damper_resistance = base_damper * factor

        elif rule.parameter_type == ParameterType.FAN_SPEED:
            if branch.has_fan and branch.fan_params is not None:
                base_fan_params = base_params.get("fan_params", branch.fan_params.copy())
                fan_scale = factor * factor

                branch.fan_params["a"] = base_fan_params.get("a", 0.0) * fan_scale
                branch.fan_params["b"] = base_fan_params.get("b", 0.0) * factor
                branch.fan_params["c"] = base_fan_params.get("c", 0.0)

    return applied_factors


def snapshot_base_params(network: VentilationNetwork) -> Dict[int, Dict]:
    base_params: Dict[int, Dict] = {}
    for bid, branch in network.branches.items():
        base_params[bid] = {
            "friction_coeff": branch.friction_coeff,
            "local_coeff": branch.local_coeff,
            "damper_resistance": branch.damper_resistance,
        }
        if branch.has_fan and branch.fan_params is not None:
            base_params[bid]["fan_params"] = branch.fan_params.copy()
    return base_params


def run_time_series_simulation(
    network: VentilationNetwork,
    rules: List[ChangeRule],
    total_hours: float = 24.0,
    time_step_minutes: float = 15.0,
    tolerance: float = 0.001,
    max_iterations: int = 1000,
    use_warm_start: bool = True,
    progress_callback=None,
) -> Tuple[TimeSeriesResult, Dict]:
    is_valid, errors = network.validate()
    if not is_valid:
        raise ValueError(f"网络验证失败: {errors}")

    time_step_hours = time_step_minutes / 60.0
    n_steps = int(math.ceil(total_hours / time_step_hours)) + 1

    branch_ids = sorted(network.branches.keys())
    node_ids = sorted(network.nodes.keys())

    result = TimeSeriesResult(branch_ids=branch_ids, node_ids=node_ids)
    result.branch_airflows = {bid: [] for bid in branch_ids}
    result.node_pressures = {nid: [] for nid in node_ids}
    result.branch_resistances = {bid: [] for bid in branch_ids}

    base_branch_params = snapshot_base_params(network)

    working_network = copy.deepcopy(network)
    previous_airflows: Optional[Dict[int, float]] = None

    import time
    sim_start = time.time()
    step_times: List[float] = []

    for step_idx in range(n_steps):
        step_start = time.time()
        current_time = step_idx * time_step_hours
        if current_time > total_hours:
            current_time = total_hours

        apply_rules_to_network(working_network, rules, current_time, base_branch_params)

        initial_guess = previous_airflows if (use_warm_start and previous_airflows is not None) else None

        try:
            airflows, pressures, solver_info = hardy_cross_solve(
                working_network,
                tolerance=tolerance,
                max_iterations=max_iterations,
                initial_guess=initial_guess,
            )
        except Exception as e:
            raise RuntimeError(f"时间步 {step_idx} (t={current_time:.2f}h) 求解失败: {e}")

        calculate_all_branch_resistances(working_network)
        natural_pressures = calculate_network_natural_pressures(working_network)
        update_branch_pressure_drops(working_network, natural_pressures)

        result.timestamps.append(current_time)
        for bid in branch_ids:
            result.branch_airflows[bid].append(airflows.get(bid, 0.0))
            br = working_network.get_branch(bid)
            result.branch_resistances[bid].append(br.resistance if br else 0.0)
        for nid in node_ids:
            result.node_pressures[nid].append(pressures.get(nid, 0.0))
        result.solver_infos.append(solver_info)

        previous_airflows = airflows.copy()

        step_times.append(time.time() - step_start)
        if progress_callback is not None:
            try:
                progress_callback(step_idx + 1, n_steps, current_time)
            except Exception:
                pass

    total_time = time.time() - sim_start
    perf_stats = {
        "total_simulation_time_s": total_time,
        "avg_step_time_s": sum(step_times) / len(step_times) if step_times else 0,
        "max_step_time_s": max(step_times) if step_times else 0,
        "min_step_time_s": min(step_times) if step_times else 0,
        "n_steps": n_steps,
        "total_hours": total_hours,
        "time_step_minutes": time_step_minutes,
    }

    return result, perf_stats


def get_solution_at_time(
    result: TimeSeriesResult,
    target_time: float,
) -> Tuple[int, Dict[int, float], Dict[int, float], Dict[int, float]]:
    if not result.timestamps:
        return 0, {}, {}, {}

    closest_idx = 0
    min_diff = float("inf")
    for i, t in enumerate(result.timestamps):
        diff = abs(t - target_time)
        if diff < min_diff:
            min_diff = diff
            closest_idx = i

    airflows = {}
    pressures = {}
    resistances = {}
    for bid in result.branch_ids:
        af_list = result.branch_airflows.get(bid, [])
        r_list = result.branch_resistances.get(bid, [])
        airflows[bid] = af_list[closest_idx] if closest_idx < len(af_list) else 0.0
        resistances[bid] = r_list[closest_idx] if closest_idx < len(r_list) else 0.0
    for nid in result.node_ids:
        p_list = result.node_pressures.get(nid, [])
        pressures[nid] = p_list[closest_idx] if closest_idx < len(p_list) else 0.0

    return closest_idx, airflows, pressures, resistances
