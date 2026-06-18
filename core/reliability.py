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
    n_simulations_per_point: int = 80,
    tolerance: float = 0.001,
    max_iterations: int = 500,
    use_parallel: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict:
    branch_ids = sorted([bid for bid in network.branches.keys()
                        if not network.get_branch(bid).is_atmospheric])

    failure_probs = np.array([0.01, 0.05, 0.10, 0.15, 0.20])

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

    heatmap_data = {
        'branch_ids': branch_ids,
        'failure_probs': failure_probs.tolist(),
        'reliability_drops': np.zeros((len(branch_ids), len(failure_probs))),
        'base_reliability': base_reliability
    }

    all_args = []
    scenario_index_map = []

    for i, bid in enumerate(branch_ids):
        branch_specific_seed = (random_seed or 42) + bid * 1000
        for j, fp in enumerate(failure_probs):
            scenarios = generate_failure_scenarios(
                network=network,
                n_simulations=n_simulations_per_point,
                branch_failure_prob=0.0,
                fan_failure_prob=0.0,
                branch_failure_probs={b: (fp if b == bid else 0.0) for b in branch_ids},
                resistance_multiplier=resistance_multiplier,
                random_seed=branch_specific_seed + j
            )
            for scenario in scenarios:
                all_args.append((
                    network, scenario, workface_branch_ids,
                    min_airflow_threshold, tolerance, max_iterations
                ))
                scenario_index_map.append((i, j))

    total_sims = len(all_args)
    all_sim_results = []

    if use_parallel:
        n_proc = min(cpu_count(), 8)
        batch_size = max(1, total_sims // 20)
        current_count = 0
        with Pool(processes=n_proc) as pool:
            for batch_start in range(0, total_sims, batch_size):
                batch_end = min(batch_start + batch_size, total_sims)
                batch = all_args[batch_start:batch_end]
                batch_results = pool.map(run_single_simulation, batch)
                all_sim_results.extend(batch_results)
                current_count = batch_end
                if progress_callback is not None:
                    progress = int((current_count / total_sims) * n_points)
                    progress_callback(max(1, min(progress, n_points)), n_points)
    else:
        for idx, args in enumerate(all_args, 1):
            all_sim_results.append(run_single_simulation(args))
            if idx % 100 == 0 or idx == total_sims:
                if progress_callback is not None:
                    progress = int((idx / total_sims) * n_points)
                    progress_callback(max(1, min(progress, n_points)), n_points)

    valid_counts = np.zeros((len(branch_ids), len(failure_probs)), dtype=int)
    total_counts = np.zeros((len(branch_ids), len(failure_probs)), dtype=int)

    for idx, sim_result in enumerate(all_sim_results):
        i, j = scenario_index_map[idx]
        total_counts[i, j] += 1
        if sim_result.is_valid and sim_result.converged:
            valid_counts[i, j] += 1

    for i in range(len(branch_ids)):
        for j in range(len(failure_probs)):
            if total_counts[i, j] > 0:
                reliability = valid_counts[i, j] / total_counts[i, j]
                reliability_drop = max(0.0, base_reliability - reliability)
                heatmap_data['reliability_drops'][i, j] = reliability_drop

    if progress_callback is not None:
        progress_callback(n_points, n_points)

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
    n_simulations_per_branch: int = 200,
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

    n_branches = len(branch_ids)
    all_args = []
    branch_index_map = []

    for i, bid in enumerate(branch_ids):
        branch_specific_seed = (random_seed or 42) + bid * 2000
        branch_probs = {b: base_branch_failure_prob for b in branch_ids}
        branch_probs[bid] = 1.0

        scenarios = generate_failure_scenarios(
            network=network,
            n_simulations=n_simulations_per_branch,
            branch_failure_prob=base_branch_failure_prob,
            fan_failure_prob=fan_failure_prob,
            branch_failure_probs=branch_probs,
            resistance_multiplier=resistance_multiplier,
            random_seed=branch_specific_seed
        )

        for scenario in scenarios:
            all_args.append((
                network, scenario, workface_branch_ids,
                min_airflow_threshold, tolerance, max_iterations
            ))
            branch_index_map.append(i)

    total_sims = len(all_args)
    all_sim_results = []

    if use_parallel:
        n_proc = min(cpu_count(), 8)
        batch_size = max(1, total_sims // max(n_branches, 1))
        with Pool(processes=n_proc) as pool:
            processed = 0
            for batch_start in range(0, total_sims, batch_size):
                batch_end = min(batch_start + batch_size, total_sims)
                batch = all_args[batch_start:batch_end]
                batch_results = pool.map(run_single_simulation, batch)
                all_sim_results.extend(batch_results)
                processed = batch_end
                if progress_callback is not None:
                    progress = int((processed / total_sims) * n_branches)
                    progress_callback(max(1, min(progress, n_branches)), n_branches)
    else:
        for idx, args in enumerate(all_args, 1):
            all_sim_results.append(run_single_simulation(args))
            if idx % 50 == 0 or idx == total_sims:
                if progress_callback is not None:
                    progress = int((idx / total_sims) * n_branches)
                    progress_callback(max(1, min(progress, n_branches)), n_branches)

    valid_counts = np.zeros(n_branches, dtype=int)
    total_counts = np.zeros(n_branches, dtype=int)

    for idx, sim_result in enumerate(all_sim_results):
        branch_idx = branch_index_map[idx]
        total_counts[branch_idx] += 1
        if sim_result.is_valid and sim_result.converged:
            valid_counts[branch_idx] += 1

    branch_impact = []
    for i, bid in enumerate(branch_ids):
        if total_counts[i] > 0:
            reliability = valid_counts[i] / total_counts[i]
        else:
            reliability = 0.0
        reliability_drop = max(0.0, base_reliability - reliability)
        branch_impact.append({
            'branch_id': bid,
            'base_reliability': base_reliability,
            'branch_failure_reliability': reliability,
            'reliability_drop': reliability_drop,
            'failure_count': int(total_counts[i] - valid_counts[i])
        })

    if progress_callback is not None:
        progress_callback(n_branches, n_branches)

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


@dataclass
class RedundantBranchCandidate:
    candidate_id: str
    original_branch_id: int
    from_node: int
    to_node: int
    length: float
    area: float
    perimeter: float
    friction_coeff: float
    local_coeff: float
    estimated_cost: float
    direction_note: str


@dataclass
class CandidateEvaluation:
    candidate_id: str
    original_branch_id: int
    added_branch_params: Dict
    reliability_before: float
    reliability_after: float
    reliability_gain: float
    estimated_cost: float
    benefit_cost_ratio: float
    simulation_count: int


@dataclass
class GreedyStep:
    step: int
    added_candidate_id: Optional[str]
    cumulative_cost: float
    cumulative_reliability: float
    reliability_increment: float
    added_branch: Optional[Dict]


@dataclass
class RedundancyDesignResult:
    bottleneck_branches: List[Dict]
    candidate_evaluations: List[CandidateEvaluation]
    top_candidates: List[CandidateEvaluation]
    greedy_steps: List[GreedyStep]
    final_reliability: float
    total_cost: float
    target_reliability: float
    target_met: bool
    recommended_branches: List[Dict]
    base_reliability: float
    random_seed: int


def identify_bottleneck_branches(
    network: VentilationNetwork,
    critical_branches: Optional[List[Dict]] = None,
    weak_branch_distribution: Optional[Dict[int, float]] = None,
    reliability_heatmap: Optional[Dict] = None,
    top_k: int = 5
) -> List[Dict]:
    bottleneck_scores: Dict[int, float] = {}

    if critical_branches:
        for cb in critical_branches:
            bid = cb['branch_id']
            score = cb.get('reliability_drop', 0) * 2.0
            bottleneck_scores[bid] = bottleneck_scores.get(bid, 0) + score

    if weak_branch_distribution:
        for bid, freq in weak_branch_distribution.items():
            bottleneck_scores[bid] = bottleneck_scores.get(bid, 0) + freq

    if reliability_heatmap and 'reliability_drops' in reliability_heatmap:
        branch_ids_hm = reliability_heatmap.get('branch_ids', [])
        drops = reliability_heatmap['reliability_drops']
        for i, bid in enumerate(branch_ids_hm):
            if i < len(drops):
                avg_drop = float(np.mean(drops[i]))
                bottleneck_scores[bid] = bottleneck_scores.get(bid, 0) + avg_drop * 1.5

    if not bottleneck_scores:
        non_atm_branch_ids = [bid for bid in sorted(network.branches.keys())
                              if not network.get_branch(bid).is_atmospheric]
        for bid in non_atm_branch_ids:
            bottleneck_scores[bid] = 0.01

    sorted_bottlenecks = sorted(
        bottleneck_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    result = []
    for bid, score in sorted_bottlenecks[:top_k]:
        branch = network.get_branch(bid)
        if branch:
            result.append({
                'branch_id': bid,
                'from_node': branch.from_node,
                'to_node': branch.to_node,
                'score': float(score),
                'has_fan': branch.has_fan,
                'is_atmospheric': branch.is_atmospheric
            })

    return result


def generate_redundant_candidates(
    network: VentilationNetwork,
    bottleneck_branches: List[Dict],
    area_shrink_ratio: float = 0.7,
    length_increase_ratio: float = 1.2
) -> List[RedundantBranchCandidate]:
    candidates = []
    generated_key_set = set()

    for bottleneck in bottleneck_branches:
        bid = bottleneck['branch_id']
        orig_branch = network.get_branch(bid)
        if not orig_branch:
            continue
        if orig_branch.is_atmospheric:
            continue

        fn = orig_branch.from_node
        tn = orig_branch.to_node
        orig_len = orig_branch.length
        orig_area = orig_branch.area
        orig_perim = orig_branch.perimeter
        orig_fric = orig_branch.friction_coeff
        orig_loc = orig_branch.local_coeff

        directions = [
            (fn, tn, '同向并联'),
            (tn, fn, '反向并联')
        ]

        for from_n, to_n, dir_note in directions:
            cand_key = (from_n, to_n)
            if cand_key in generated_key_set:
                continue

            new_len = orig_len * length_increase_ratio
            new_area = orig_area * area_shrink_ratio
            new_perim = orig_perim * (area_shrink_ratio ** 0.5)
            cost_est = new_len * new_area

            cand_id = f"RED_{bid}_{from_n}_{to_n}"
            candidate = RedundantBranchCandidate(
                candidate_id=cand_id,
                original_branch_id=bid,
                from_node=from_n,
                to_node=to_n,
                length=new_len,
                area=new_area,
                perimeter=new_perim,
                friction_coeff=orig_fric,
                local_coeff=orig_loc,
                estimated_cost=cost_est,
                direction_note=dir_note
            )
            candidates.append(candidate)
            generated_key_set.add(cand_key)

    return candidates


def evaluate_single_candidate(
    base_network: VentilationNetwork,
    candidate: RedundantBranchCandidate,
    reliability_params: Dict,
    n_simulations: int = 500,
    fixed_seed: int = 12345,
    tolerance: float = 0.001,
    max_iterations: int = 500,
    use_parallel: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> CandidateEvaluation:
    test_network = copy.deepcopy(base_network)

    used_ids = set(test_network.branches.keys())
    new_id = max(used_ids) + 1 if used_ids else 1

    new_branch = Branch(
        id=new_id,
        from_node=candidate.from_node,
        to_node=candidate.to_node,
        length=candidate.length,
        area=candidate.area,
        perimeter=candidate.perimeter,
        friction_coeff=candidate.friction_coeff,
        local_coeff=candidate.local_coeff,
        has_fan=False,
        fan_params=None,
        has_damper=False,
        damper_resistance=0.0,
        is_atmospheric=False
    )
    test_network.add_branch(new_branch)

    workface_ids = reliability_params.get('workface_branch_ids', [])
    branch_fail_prob = reliability_params.get('branch_failure_prob', 0.05)
    fan_fail_prob = reliability_params.get('fan_failure_prob', 0.02)
    min_q_thresh = reliability_params.get('min_airflow_threshold', 4.0)
    res_mult = reliability_params.get('resistance_multiplier', 10.0)

    result = run_monte_carlo_simulation(
        network=test_network,
        n_simulations=n_simulations,
        workface_branch_ids=workface_ids,
        min_airflow_threshold=min_q_thresh,
        branch_failure_prob=branch_fail_prob,
        fan_failure_prob=fan_fail_prob,
        resistance_multiplier=res_mult,
        random_seed=fixed_seed,
        tolerance=tolerance,
        max_iterations=max_iterations,
        use_parallel=use_parallel,
        progress_callback=progress_callback
    )

    reliability_after = result.reliability
    base_reliability = reliability_params.get('base_reliability', 0.0)
    gain = max(0.0, reliability_after - base_reliability)
    cost = candidate.estimated_cost
    ratio = gain / cost if cost > 0 else 0.0

    added_branch_params = {
        'id': new_id,
        'from_node': candidate.from_node,
        'to_node': candidate.to_node,
        'length': candidate.length,
        'area': candidate.area,
        'perimeter': candidate.perimeter,
        'friction_coeff': candidate.friction_coeff,
        'local_coeff': candidate.local_coeff,
        'candidate_id': candidate.candidate_id,
        'original_branch_id': candidate.original_branch_id,
        'direction_note': candidate.direction_note
    }

    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        original_branch_id=candidate.original_branch_id,
        added_branch_params=added_branch_params,
        reliability_before=base_reliability,
        reliability_after=reliability_after,
        reliability_gain=gain,
        estimated_cost=cost,
        benefit_cost_ratio=ratio,
        simulation_count=n_simulations
    )


def evaluate_all_candidates(
    base_network: VentilationNetwork,
    candidates: List[RedundantBranchCandidate],
    reliability_params: Dict,
    n_simulations_per_candidate: int = 500,
    fixed_seed: int = 12345,
    use_parallel: bool = True,
    overall_progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[CandidateEvaluation]:
    evaluations = []
    total_candidates = len(candidates)

    for idx, candidate in enumerate(candidates, 1):
        eval_result = evaluate_single_candidate(
            base_network=base_network,
            candidate=candidate,
            reliability_params=reliability_params,
            n_simulations=n_simulations_per_candidate,
            fixed_seed=fixed_seed,
            use_parallel=use_parallel
        )
        evaluations.append(eval_result)

        if overall_progress_callback is not None:
            overall_progress_callback(idx, total_candidates)

    return evaluations


def greedy_combine_redundancy(
    base_network: VentilationNetwork,
    candidate_evaluations: List[CandidateEvaluation],
    reliability_params: Dict,
    target_reliability: float,
    max_branches: int = 5,
    n_simulations: int = 500,
    fixed_seed: int = 12345,
    use_parallel: bool = True,
    overall_progress_callback: Optional[Callable[[int, int], None]] = None
) -> RedundancyDesignResult:
    sorted_evals = sorted(
        candidate_evaluations,
        key=lambda x: x.benefit_cost_ratio,
        reverse=True
    )
    top5 = sorted_evals[:5]

    base_reliability = reliability_params.get('base_reliability', 0.0)
    greedy_steps: List[GreedyStep] = [
        GreedyStep(
            step=0,
            added_candidate_id=None,
            cumulative_cost=0.0,
            cumulative_reliability=base_reliability,
            reliability_increment=0.0,
            added_branch=None
        )
    ]

    current_network = copy.deepcopy(base_network)
    added_branches: List[Dict] = []
    cumulative_cost = 0.0
    current_reliability = base_reliability
    available_evals = list(candidate_evaluations)
    target_met = current_reliability >= target_reliability

    workface_ids = reliability_params.get('workface_branch_ids', [])
    branch_fail_prob = reliability_params.get('branch_failure_prob', 0.05)
    fan_fail_prob = reliability_params.get('fan_failure_prob', 0.02)
    min_q_thresh = reliability_params.get('min_airflow_threshold', 4.0)
    res_mult = reliability_params.get('resistance_multiplier', 10.0)

    max_steps = min(max_branches, len(available_evals))

    for step_num in range(1, max_steps + 1):
        if current_reliability >= target_reliability:
            target_met = True
            break
        if not available_evals:
            break

        best_eval = None
        best_gain = 0.0
        best_test_reliability = 0.0

        for cand_eval in available_evals:
            test_network = copy.deepcopy(current_network)

            used_ids = set(test_network.branches.keys())
            new_id = max(used_ids) + 1 if used_ids else 1

            branch_params = cand_eval.added_branch_params
            new_branch = Branch(
                id=new_id,
                from_node=branch_params['from_node'],
                to_node=branch_params['to_node'],
                length=branch_params['length'],
                area=branch_params['area'],
                perimeter=branch_params['perimeter'],
                friction_coeff=branch_params['friction_coeff'],
                local_coeff=branch_params['local_coeff'],
                has_fan=False,
                fan_params=None,
                has_damper=False,
                damper_resistance=0.0,
                is_atmospheric=False
            )
            test_network.add_branch(new_branch)

            result = run_monte_carlo_simulation(
                network=test_network,
                n_simulations=n_simulations,
                workface_branch_ids=workface_ids,
                min_airflow_threshold=min_q_thresh,
                branch_failure_prob=branch_fail_prob,
                fan_failure_prob=fan_fail_prob,
                resistance_multiplier=res_mult,
                random_seed=fixed_seed,
                tolerance=0.001,
                max_iterations=500,
                use_parallel=use_parallel
            )

            test_reliability = result.reliability
            gain = max(0.0, test_reliability - current_reliability)

            if gain > best_gain:
                best_gain = gain
                best_eval = cand_eval
                best_test_reliability = test_reliability

        if best_eval is None or best_gain <= 0:
            break

        used_ids = set(current_network.branches.keys())
        new_id = max(used_ids) + 1 if used_ids else 1

        branch_params_copy = dict(best_eval.added_branch_params)
        branch_params_copy['id'] = new_id
        new_branch = Branch(
            id=new_id,
            from_node=branch_params_copy['from_node'],
            to_node=branch_params_copy['to_node'],
            length=branch_params_copy['length'],
            area=branch_params_copy['area'],
            perimeter=branch_params_copy['perimeter'],
            friction_coeff=branch_params_copy['friction_coeff'],
            local_coeff=branch_params_copy['local_coeff'],
            has_fan=False,
            fan_params=None,
            has_damper=False,
            damper_resistance=0.0,
            is_atmospheric=False
        )
        current_network.add_branch(new_branch)

        added_branches.append(branch_params_copy)
        cumulative_cost += best_eval.estimated_cost
        current_reliability = best_test_reliability

        step_record = GreedyStep(
            step=step_num,
            added_candidate_id=best_eval.candidate_id,
            cumulative_cost=cumulative_cost,
            cumulative_reliability=current_reliability,
            reliability_increment=best_gain,
            added_branch=branch_params_copy
        )
        greedy_steps.append(step_record)

        available_evals = [e for e in available_evals if e.candidate_id != best_eval.candidate_id]

        if overall_progress_callback is not None:
            overall_progress_callback(step_num, max_steps)

        if current_reliability >= target_reliability:
            target_met = True
            break

    final_reliability = current_reliability
    if final_reliability >= target_reliability:
        target_met = True

    bottleneck_branches_info = []
    if reliability_params.get('bottleneck_branches'):
        bottleneck_branches_info = reliability_params['bottleneck_branches']

    result = RedundancyDesignResult(
        bottleneck_branches=bottleneck_branches_info,
        candidate_evaluations=sorted_evals,
        top_candidates=top5,
        greedy_steps=greedy_steps,
        final_reliability=final_reliability,
        total_cost=cumulative_cost,
        target_reliability=target_reliability,
        target_met=target_met,
        recommended_branches=added_branches,
        base_reliability=base_reliability,
        random_seed=fixed_seed
    )

    return result


def export_redundancy_design_to_json(
    design_result: RedundancyDesignResult,
    indent: int = 2
) -> str:
    recommended_with_details = []
    for branch in design_result.recommended_branches:
        rec = dict(branch)
        cand_eval = next(
            (e for e in design_result.candidate_evaluations
             if e.candidate_id == branch.get('candidate_id')),
            None
        )
        if cand_eval:
            rec['reliability_gain'] = cand_eval.reliability_gain
            rec['estimated_cost'] = cand_eval.estimated_cost
            rec['benefit_cost_ratio'] = cand_eval.benefit_cost_ratio
        recommended_with_details.append(rec)

    report = {
        'version': '1.0',
        'analysis_type': 'redundancy_design',
        'base_reliability': design_result.base_reliability,
        'target_reliability': design_result.target_reliability,
        'final_reliability': design_result.final_reliability,
        'target_met': design_result.target_met,
        'total_cost': design_result.total_cost,
        'random_seed': design_result.random_seed,
        'bottleneck_branches': design_result.bottleneck_branches,
        'top_5_candidates': [
            {
                'candidate_id': e.candidate_id,
                'original_branch_id': e.original_branch_id,
                'added_branch_params': e.added_branch_params,
                'reliability_before': e.reliability_before,
                'reliability_after': e.reliability_after,
                'reliability_gain': e.reliability_gain,
                'estimated_cost': e.estimated_cost,
                'benefit_cost_ratio': e.benefit_cost_ratio,
                'simulation_count': e.simulation_count
            }
            for e in design_result.top_candidates
        ],
        'greedy_optimization_steps': [
            {
                'step': s.step,
                'added_candidate_id': s.added_candidate_id,
                'cumulative_cost': s.cumulative_cost,
                'cumulative_reliability': s.cumulative_reliability,
                'reliability_increment': s.reliability_increment,
                'added_branch': s.added_branch
            }
            for s in design_result.greedy_steps
        ],
        'recommended_redundant_branches': recommended_with_details
    }

    return json.dumps(report, indent=indent, ensure_ascii=False)
