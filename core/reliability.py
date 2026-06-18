from __future__ import annotations
import copy
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count

from .network import VentilationNetwork, Branch
from .hardy_cross import hardy_cross_solve
from .resistance import calculate_all_branch_resistances, calculate_network_natural_pressures


@dataclass
class FailureScenario:
    scenario_id: int
    failed_branches: List[int]
    failed_fans: List[int]
    resistance_multipliers: Dict[int, float]
    fan_disabled: Dict[int, bool]


@dataclass
class SimulationResult:
    scenario_id: int
    is_valid: bool
    converged: bool
    min_airflow: float
    min_airflow_branch: int
    workface_airflows: Dict[int, float]
    all_airflows: Dict[int, float]
    failed_branches: List[int]
    failed_fans: List[int]


@dataclass
class ReliabilityAnalysisResult:
    total_simulations: int
    valid_count: int
    reliability: float
    weak_branch_frequency: Dict[int, int]
    weak_branch_distribution: Dict[int, float]
    failure_min_airflows: List[float]
    failure_stats: Dict[str, float]
    simulation_results: List[SimulationResult]
    parameters: Dict
    heatmap_data: Optional[Dict] = None
    critical_branches: Optional[List[Dict]] = None


def generate_failure_scenarios(
    network: VentilationNetwork,
    n_simulations: int,
    branch_failure_prob: float = 0.05,
    fan_failure_prob: float = 0.02,
    branch_failure_probs: Optional[Dict[int, float]] = None,
    resistance_multiplier: float = 10.0,
    random_seed: Optional[int] = None
) -> List[FailureScenario]:
    if random_seed is not None:
        np.random.seed(random_seed)

    branch_ids = sorted(network.branches.keys())
    fan_branch_ids = [b.id for b in network.get_fan_branches()]

    if branch_failure_probs is None:
        branch_failure_probs = {bid: branch_failure_prob for bid in branch_ids}

    scenarios = []

    for sim_id in range(n_simulations):
        failed_branches = []
        failed_fans = []
        resistance_multipliers = {}
        fan_disabled = {}

        for bid in branch_ids:
            branch = network.get_branch(bid)
            if branch and branch.is_atmospheric:
                resistance_multipliers[bid] = 1.0
                fan_disabled[bid] = False
                continue

            prob = branch_failure_probs.get(bid, branch_failure_prob)
            if np.random.random() < prob:
                failed_branches.append(bid)
                resistance_multipliers[bid] = resistance_multiplier
            else:
                resistance_multipliers[bid] = 1.0

            fan_disabled[bid] = False

        for bid in fan_branch_ids:
            if np.random.random() < fan_failure_prob:
                failed_fans.append(bid)
                fan_disabled[bid] = True

        scenario = FailureScenario(
            scenario_id=sim_id,
            failed_branches=failed_branches,
            failed_fans=failed_fans,
            resistance_multipliers=resistance_multipliers,
            fan_disabled=fan_disabled
        )
        scenarios.append(scenario)

    return scenarios


def apply_failure_scenario(
    network: VentilationNetwork,
    scenario: FailureScenario
) -> VentilationNetwork:
    modified_network = copy.deepcopy(network)

    for bid, multiplier in scenario.resistance_multipliers.items():
        branch = modified_network.get_branch(bid)
        if branch and not branch.is_atmospheric:
            branch.friction_coeff *= multiplier
            branch.local_coeff *= multiplier
            if branch.has_damper:
                branch.damper_resistance *= multiplier

    for bid, disabled in scenario.fan_disabled.items():
        if disabled:
            branch = modified_network.get_branch(bid)
            if branch and branch.has_fan and branch.fan_params:
                branch.fan_params['a'] = 0.0
                branch.fan_params['b'] = 0.0
                branch.fan_params['c'] = 0.0

    return modified_network


def run_single_simulation(
    args: Tuple
) -> SimulationResult:
    network, scenario, workface_branch_ids, min_airflow_threshold, tolerance, max_iterations = args

    try:
        modified_network = apply_failure_scenario(network, scenario)
        airflows, pressures, info = hardy_cross_solve(
            modified_network,
            tolerance=tolerance,
            max_iterations=max_iterations
        )

        if not info.get('converged', False):
            return SimulationResult(
                scenario_id=scenario.scenario_id,
                is_valid=False,
                converged=False,
                min_airflow=0.0,
                min_airflow_branch=-1,
                workface_airflows={},
                all_airflows={},
                failed_branches=scenario.failed_branches,
                failed_fans=scenario.failed_fans
            )

        workface_airflows = {}
        min_q = float('inf')
        min_q_branch = -1

        for bid in workface_branch_ids:
            q = abs(airflows.get(bid, 0.0))
            workface_airflows[bid] = q
            if q < min_q:
                min_q = q
                min_q_branch = bid

        is_valid = min_q >= min_airflow_threshold

        return SimulationResult(
            scenario_id=scenario.scenario_id,
            is_valid=is_valid,
            converged=True,
            min_airflow=min_q,
            min_airflow_branch=min_q_branch,
            workface_airflows=workface_airflows,
            all_airflows=airflows,
            failed_branches=scenario.failed_branches,
            failed_fans=scenario.failed_fans
        )

    except Exception as e:
        return SimulationResult(
            scenario_id=scenario.scenario_id,
            is_valid=False,
            converged=False,
            min_airflow=0.0,
            min_airflow_branch=-1,
            workface_airflows={},
            all_airflows={},
            failed_branches=scenario.failed_branches,
            failed_fans=scenario.failed_fans
        )


def run_monte_carlo_simulation(
    network: VentilationNetwork,
    n_simulations: int = 1000,
    workface_branch_ids: Optional[List[int]] = None,
    min_airflow_threshold: float = 4.0,
    branch_failure_prob: float = 0.05,
    fan_failure_prob: float = 0.02,
    branch_failure_probs: Optional[Dict[int, float]] = None,
    resistance_multiplier: float = 10.0,
    random_seed: Optional[int] = None,
    tolerance: float = 0.001,
    max_iterations: int = 500,
    use_parallel: bool = True,
    n_processes: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> ReliabilityAnalysisResult:
    if workface_branch_ids is None:
        non_atm = [bid for bid in sorted(network.branches.keys())
                   if not network.get_branch(bid).is_atmospheric]
        workface_branch_ids = non_atm[-3:] if len(non_atm) >= 3 else non_atm

    scenarios = generate_failure_scenarios(
        network=network,
        n_simulations=n_simulations,
        branch_failure_prob=branch_failure_prob,
        fan_failure_prob=fan_failure_prob,
        branch_failure_probs=branch_failure_probs,
        resistance_multiplier=resistance_multiplier,
        random_seed=random_seed
    )

    args_list = [
        (network, scenario, workface_branch_ids, min_airflow_threshold, tolerance, max_iterations)
        for scenario in scenarios
    ]

    simulation_results = []

    if use_parallel:
        if n_processes is None:
            n_processes = min(cpu_count(), 8)

        with Pool(processes=n_processes) as pool:
            for i, result in enumerate(pool.imap(run_single_simulation, args_list), 1):
                simulation_results.append(result)
                if progress_callback is not None:
                    progress_callback(i, n_simulations)
    else:
        for i, args in enumerate(args_list, 1):
            result = run_single_simulation(args)
            simulation_results.append(result)
            if progress_callback is not None:
                progress_callback(i, n_simulations)

    return analyze_simulation_results(
        simulation_results=simulation_results,
        n_simulations=n_simulations,
        workface_branch_ids=workface_branch_ids,
        parameters={
            'n_simulations': n_simulations,
            'min_airflow_threshold': min_airflow_threshold,
            'branch_failure_prob': branch_failure_prob,
            'fan_failure_prob': fan_failure_prob,
            'resistance_multiplier': resistance_multiplier,
            'random_seed': random_seed,
            'workface_branch_ids': workface_branch_ids
        }
    )


def analyze_simulation_results(
    simulation_results: List[SimulationResult],
    n_simulations: int,
    workface_branch_ids: List[int],
    parameters: Dict
) -> ReliabilityAnalysisResult:
    valid_count = sum(1 for r in simulation_results if r.is_valid and r.converged)
    reliability = valid_count / n_simulations if n_simulations > 0 else 0.0

    weak_branch_frequency: Dict[int, int] = {}
    failure_min_airflows: List[float] = []

    for r in simulation_results:
        if r.converged and not r.is_valid:
            failure_min_airflows.append(r.min_airflow)
            bid = r.min_airflow_branch
            if bid > 0:
                weak_branch_frequency[bid] = weak_branch_frequency.get(bid, 0) + 1

    total_failures = len(failure_min_airflows)
    weak_branch_distribution: Dict[int, float] = {}
    if total_failures > 0:
        for bid, count in weak_branch_frequency.items():
            weak_branch_distribution[bid] = count / total_failures

    failure_stats = {}
    if failure_min_airflows:
        arr = np.array(failure_min_airflows)
        failure_stats = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            '5th_percentile': float(np.percentile(arr, 5)),
            '25th_percentile': float(np.percentile(arr, 25)),
            '75th_percentile': float(np.percentile(arr, 75)),
            '95th_percentile': float(np.percentile(arr, 95))
        }

    return ReliabilityAnalysisResult(
        total_simulations=n_simulations,
        valid_count=valid_count,
        reliability=reliability,
        weak_branch_frequency=weak_branch_frequency,
        weak_branch_distribution=weak_branch_distribution,
        failure_min_airflows=failure_min_airflows,
        failure_stats=failure_stats,
        simulation_results=simulation_results,
        parameters=parameters
    )


def generate_reliability_heatmap(
    network: VentilationNetwork,
    workface_branch_ids: List[int],
    min_airflow_threshold: float = 4.0,
    base_branch_failure_prob: float = 0.05,
    fan_failure_prob: float = 0.02,
    resistance_multiplier: float = 10.0,
    random_seed: Optional[int] = 42,
    n_simulations_per_point: int = 300,
    tolerance: float = 0.001,
    max_iterations: int = 500,
    use_parallel: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict:
    branch_ids = sorted([bid for bid in network.branches.keys()
                        if not network.get_branch(bid).is_atmospheric])

    failure_probs = np.arange(0.01, 0.21, 0.02)

    base_result = run_monte_carlo_simulation(
        network=network,
        n_simulations=n_simulations_per_point,
        workface_branch_ids=workface_branch_ids,
        min_airflow_threshold=min_airflow_threshold,
        branch_failure_prob=0.0,
        fan_failure_prob=0.0,
        random_seed=random_seed,
        tolerance=tolerance,
        max_iterations=max_iterations,
        use_parallel=use_parallel
    )
    base_reliability = base_result.reliability

    n_points = len(branch_ids) * len(failure_probs)
    current_point = 0

    heatmap_data = {
        'branch_ids': branch_ids,
        'failure_probs': failure_probs.tolist(),
        'reliability_drops': np.zeros((len(branch_ids), len(failure_probs))),
        'base_reliability': base_reliability
    }

    for i, bid in enumerate(branch_ids):
        for j, fp in enumerate(failure_probs):
            branch_probs = {b: 0.0 for b in branch_ids}
            branch_probs[bid] = fp

            result = run_monte_carlo_simulation(
                network=network,
                n_simulations=n_simulations_per_point,
                workface_branch_ids=workface_branch_ids,
                min_airflow_threshold=min_airflow_threshold,
                branch_failure_prob=0.0,
                fan_failure_prob=0.0,
                branch_failure_probs=branch_probs,
                resistance_multiplier=resistance_multiplier,
                random_seed=random_seed,
                tolerance=tolerance,
                max_iterations=max_iterations,
                use_parallel=use_parallel
            )

            reliability_drop = base_reliability - result.reliability
            heatmap_data['reliability_drops'][i, j] = reliability_drop

            current_point += 1
            if progress_callback is not None:
                progress_callback(current_point, n_points)

    heatmap_data['reliability_drops'] = heatmap_data['reliability_drops'].tolist()

    return heatmap_data


def identify_critical_branches(
    network: VentilationNetwork,
    workface_branch_ids: List[int],
    min_airflow_threshold: float = 4.0,
    base_branch_failure_prob: float = 0.05,
    fan_failure_prob: float = 0.02,
    resistance_multiplier: float = 10.0,
    random_seed: Optional[int] = 42,
    n_simulations_per_branch: int = 500,
    top_k: int = 3,
    tolerance: float = 0.001,
    max_iterations: int = 500,
    use_parallel: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict]:
    branch_ids = sorted([bid for bid in network.branches.keys()
                        if not network.get_branch(bid).is_atmospheric])

    base_result = run_monte_carlo_simulation(
        network=network,
        n_simulations=n_simulations_per_branch,
        workface_branch_ids=workface_branch_ids,
        min_airflow_threshold=min_airflow_threshold,
        branch_failure_prob=base_branch_failure_prob,
        fan_failure_prob=fan_failure_prob,
        random_seed=random_seed,
        tolerance=tolerance,
        max_iterations=max_iterations,
        use_parallel=use_parallel
    )
    base_reliability = base_result.reliability

    branch_impact = []
    n_branches = len(branch_ids)

    for i, bid in enumerate(branch_ids):
        branch_probs = {b: base_branch_failure_prob for b in branch_ids}
        branch_probs[bid] = 1.0

        result = run_monte_carlo_simulation(
            network=network,
            n_simulations=n_simulations_per_branch,
            workface_branch_ids=workface_branch_ids,
            min_airflow_threshold=min_airflow_threshold,
            branch_failure_prob=base_branch_failure_prob,
            fan_failure_prob=fan_failure_prob,
            branch_failure_probs=branch_probs,
            resistance_multiplier=resistance_multiplier,
            random_seed=random_seed,
            tolerance=tolerance,
            max_iterations=max_iterations,
            use_parallel=use_parallel
        )

        reliability_drop = base_reliability - result.reliability
        branch_impact.append({
            'branch_id': bid,
            'base_reliability': base_reliability,
            'branch_failure_reliability': result.reliability,
            'reliability_drop': reliability_drop,
            'failure_count': result.total_simulations - result.valid_count
        })

        if progress_callback is not None:
            progress_callback(i + 1, n_branches)

    branch_impact.sort(key=lambda x: x['reliability_drop'], reverse=True)

    return branch_impact[:top_k]


def export_reliability_report_to_json(
    result: ReliabilityAnalysisResult,
    indent: int = 2
) -> str:
    report = {
        'version': '1.0',
        'analysis_type': 'ventilation_network_reliability',
        'parameters': result.parameters,
        'summary': {
            'total_simulations': result.total_simulations,
            'valid_count': result.valid_count,
            'reliability': result.reliability,
            'failure_rate': 1.0 - result.reliability
        },
        'failure_statistics': result.failure_stats,
        'weak_branch_analysis': {
            'frequency': {str(k): v for k, v in result.weak_branch_frequency.items()},
            'distribution': {str(k): v for k, v in result.weak_branch_distribution.items()}
        },
        'simulation_details': [
            {
                'scenario_id': r.scenario_id,
                'is_valid': r.is_valid,
                'converged': r.converged,
                'min_airflow': r.min_airflow,
                'min_airflow_branch': r.min_airflow_branch,
                'failed_branches': r.failed_branches,
                'failed_fans': r.failed_fans,
                'workface_airflows': r.workface_airflows
            }
            for r in result.simulation_results
        ]
    }

    if result.heatmap_data is not None:
        report['heatmap_data'] = result.heatmap_data

    if result.critical_branches is not None:
        report['critical_branches'] = result.critical_branches

    return json.dumps(report, indent=indent, ensure_ascii=False)
