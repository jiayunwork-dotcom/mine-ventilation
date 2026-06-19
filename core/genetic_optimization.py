from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

from .network import VentilationNetwork, Branch
from .hardy_cross import hardy_cross_solve
from .fan_operation import calculate_total_power_consumption
from .resistance import (
    calculate_all_branch_resistances,
    calculate_network_natural_pressures,
    calculate_branch_resistance,
    calculate_friction_resistance,
    calculate_local_resistance,
)


@dataclass
class GAParameters:
    population_size: int = 50
    max_generations: int = 100
    crossover_prob: float = 0.8
    mutation_prob: float = 0.1
    elitism_count: int = 2
    tournament_size: int = 3
    sbx_distribution_index: float = 20.0
    pm_distribution_index: float = 20.0
    penalty_coefficient: float = 10000.0
    min_airflow_threshold: float = 4.0
    convergence_generations: int = 20
    convergence_improvement: float = 0.001
    fan_speed_min: float = 0.5
    fan_speed_max: float = 1.2
    damper_open_min: float = 0.0
    damper_open_max: float = 1.0
    damper_max_resistance_multiplier: float = 50.0
    tolerance: float = 0.001
    max_iterations: int = 500
    random_seed: Optional[int] = None
    optimization_mode: str = 'single'
    energy_weight: float = 1.0
    uniformity_weight: float = 0.0


@dataclass
class GenerationHistory:
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    best_individual: np.ndarray


@dataclass
class GAOptimizationResult:
    success: bool
    message: str
    parameters: GAParameters
    best_solution: Optional[np.ndarray] = None
    best_fitness: float = float('inf')
    best_power: float = 0.0
    initial_power: float = 0.0
    energy_saving_percent: float = 0.0
    fan_speeds: Dict[int, float] = field(default_factory=dict)
    damper_openings: Dict[int, float] = field(default_factory=dict)
    workface_airflows: Dict[int, Dict] = field(default_factory=dict)
    all_airflows: Dict[int, float] = field(default_factory=dict)
    all_pressures: Dict[int, float] = field(default_factory=dict)
    history: List[GenerationHistory] = field(default_factory=list)
    total_time: float = 0.0
    generations_run: int = 0
    converged: bool = False
    constraint_satisfied: Dict[int, bool] = field(default_factory=dict)
    total_violation: float = 0.0
    uniformity_value: float = 0.0
    initial_uniformity: float = 0.0
    pareto_efficiency: float = 0.0
    convergence_reason: str = ''


def _deep_copy_network(network: VentilationNetwork) -> VentilationNetwork:
    return copy.deepcopy(network)


def _extract_fan_and_damper_ids(network: VentilationNetwork) -> Tuple[List[int], List[int]]:
    fan_ids = []
    damper_ids = []
    for bid, branch in network.branches.items():
        if branch.has_fan:
            fan_ids.append(bid)
        if branch.has_damper:
            damper_ids.append(bid)
    return sorted(fan_ids), sorted(damper_ids)


def _apply_solution_to_network(
    network: VentilationNetwork,
    solution: np.ndarray,
    fan_ids: List[int],
    damper_ids: List[int],
    params: GAParameters
) -> None:
    n_fans = len(fan_ids)

    for i, fan_id in enumerate(fan_ids):
        speed_factor = solution[i]
        branch = network.get_branch(fan_id)
        if branch and branch.fan_params:
            orig_a = branch.fan_params.get('a', 0.0)
            orig_b = branch.fan_params.get('b', 0.0)
            orig_c = branch.fan_params.get('c', 0.0)
            branch.fan_params['a'] = orig_a * speed_factor * speed_factor
            branch.fan_params['b'] = orig_b * speed_factor
            branch.fan_params['c'] = orig_c

    for j, damper_id in enumerate(damper_ids):
        opening = solution[n_fans + j]
        branch = network.get_branch(damper_id)
        if branch:
            base_r = calculate_friction_resistance(
                branch.friction_coeff,
                branch.length,
                branch.perimeter,
                branch.area
            ) + calculate_local_resistance(branch.local_coeff, branch.area)

            multiplier = params.damper_max_resistance_multiplier * (1.0 - opening)
            branch.damper_resistance = base_r * multiplier


def _restore_network_original(
    network: VentilationNetwork,
    original_network: VentilationNetwork
) -> None:
    for bid in network.branches.keys():
        orig_branch = original_network.get_branch(bid)
        curr_branch = network.get_branch(bid)
        if orig_branch and curr_branch:
            curr_branch.damper_resistance = orig_branch.damper_resistance
            if curr_branch.fan_params and orig_branch.fan_params:
                for k in ['a', 'b', 'c']:
                    if k in orig_branch.fan_params:
                        curr_branch.fan_params[k] = orig_branch.fan_params[k]


def _evaluate_single(
    network: VentilationNetwork,
    original_network: VentilationNetwork,
    solution: np.ndarray,
    fan_ids: List[int],
    damper_ids: List[int],
    workface_ids: List[int],
    params: GAParameters,
) -> Tuple[float, float, Dict[int, float], Dict[int, float], Dict[int, float], bool, float, float]:
    _apply_solution_to_network(network, solution, fan_ids, damper_ids, params)

    try:
        airflows, pressures, info = hardy_cross_solve(
            network,
            tolerance=params.tolerance,
            max_iterations=params.max_iterations
        )
    except Exception:
        _restore_network_original(network, original_network)
        return float('inf'), float('inf'), {}, {}, {}, False, float('inf'), float('inf')

    converged = info.get('converged', False)
    if not converged:
        _restore_network_original(network, original_network)
        return float('inf'), float('inf'), {}, {}, {}, False, float('inf'), float('inf')

    network.update_solution(airflows, pressures)
    calculate_all_branch_resistances(network)
    natural_pressures = calculate_network_natural_pressures(network)
    from .resistance import update_branch_pressure_drops
    update_branch_pressure_drops(network, natural_pressures)

    power_info = calculate_total_power_consumption(network)
    total_power = power_info['total_shaft_power']

    total_violation = 0.0
    workface_airflow_values = {}
    for wf_id in workface_ids:
        wf_branch = network.get_branch(wf_id)
        if wf_branch:
            wf_q = abs(wf_branch.airflow)
            workface_airflow_values[wf_id] = wf_q
            deficit = max(0.0, params.min_airflow_threshold - wf_q)
            total_violation += deficit * deficit

    uniformity = 0.0
    if workface_ids and params.optimization_mode == 'dual':
        deviations = []
        for wf_id in workface_ids:
            wf_q = workface_airflow_values.get(wf_id, 0.0)
            deviation = wf_q - params.min_airflow_threshold
            deviations.append(deviation)
        if len(deviations) > 1:
            uniformity = float(np.std(deviations))
        elif len(deviations) == 1:
            uniformity = abs(deviations[0])

    _restore_network_original(network, original_network)

    if params.optimization_mode == 'dual':
        energy_obj = total_power
        uniformity_obj = uniformity * params.penalty_coefficient
        fitness = (params.energy_weight * energy_obj
                   + params.uniformity_weight * uniformity_obj
                   + params.penalty_coefficient * total_violation)
    else:
        fitness = total_power + params.penalty_coefficient * total_violation

    return fitness, total_power, airflows, pressures, workface_airflow_values, True, total_violation, uniformity


def _initialize_population(
    pop_size: int,
    n_vars: int,
    fan_ids: List[int],
    damper_ids: List[int],
    params: GAParameters,
    rng: np.random.Generator
) -> np.ndarray:
    population = np.zeros((pop_size, n_vars))
    n_fans = len(fan_ids)

    for i in range(pop_size):
        for j in range(n_fans):
            population[i, j] = rng.uniform(params.fan_speed_min, params.fan_speed_max)
        for k in range(len(damper_ids)):
            population[i, n_fans + k] = rng.uniform(params.damper_open_min, params.damper_open_max)

    if n_vars > 0:
        population[0, :n_fans] = 1.0
        population[0, n_fans:] = 1.0

    return population


def _tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator
) -> np.ndarray:
    pop_size = len(population)
    n_vars = population.shape[1]
    selected = np.zeros((pop_size, n_vars))

    for i in range(pop_size):
        candidates = rng.choice(pop_size, size=tournament_size, replace=False)
        best_idx = candidates[np.argmin(fitness[candidates])]
        selected[i] = population[best_idx]

    return selected


def _simulated_binary_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    n_fans: int,
    n_dampers: int,
    params: GAParameters,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    child1 = parent1.copy()
    child2 = parent2.copy()
    eta = params.sbx_distribution_index

    for i in range(len(parent1)):
        if rng.random() > 0.5:
            continue

        if abs(parent1[i] - parent2[i]) < 1e-14:
            continue

        if i < n_fans:
            low, high = params.fan_speed_min, params.fan_speed_max
        else:
            low, high = params.damper_open_min, params.damper_open_max

        x1 = min(parent1[i], parent2[i])
        x2 = max(parent1[i], parent2[i])

        rand = rng.random()

        beta = 1.0 + (2.0 * (x1 - low) / (x2 - x1))
        alpha = 2.0 - np.power(beta, -(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = np.power(rand * alpha, 1.0 / (eta + 1.0))
        else:
            betaq = np.power(1.0 / (2.0 - rand * alpha), 1.0 / (eta + 1.0))
        c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        beta = 1.0 + (2.0 * (high - x2) / (x2 - x1))
        alpha = 2.0 - np.power(beta, -(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = np.power(rand * alpha, 1.0 / (eta + 1.0))
        else:
            betaq = np.power(1.0 / (2.0 - rand * alpha), 1.0 / (eta + 1.0))
        c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

        c1 = np.clip(c1, low, high)
        c2 = np.clip(c2, low, high)

        if rng.random() < 0.5:
            child1[i], child2[i] = c2, c1
        else:
            child1[i], child2[i] = c1, c2

    return child1, child2


def _polynomial_mutation(
    individual: np.ndarray,
    n_fans: int,
    n_dampers: int,
    params: GAParameters,
    rng: np.random.Generator
) -> np.ndarray:
    mutant = individual.copy()
    eta = params.pm_distribution_index

    for i in range(len(individual)):
        if rng.random() > params.mutation_prob:
            continue

        if i < n_fans:
            low, high = params.fan_speed_min, params.fan_speed_max
        else:
            low, high = params.damper_open_min, params.damper_open_max

        x = mutant[i]
        delta1 = (x - low) / (high - low)
        delta2 = (high - x) / (high - low)
        rand = rng.random()
        mut_pow = 1.0 / (eta + 1.0)

        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * np.power(xy, eta + 1.0)
            delta_q = np.power(val, mut_pow) - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * np.power(xy, eta + 1.0)
            delta_q = 1.0 - np.power(val, mut_pow)

        x_new = x + delta_q * (high - low)
        mutant[i] = np.clip(x_new, low, high)

    return mutant


def run_genetic_optimization(
    network: VentilationNetwork,
    workface_branch_ids: List[int],
    params: Optional[GAParameters] = None,
    progress_callback: Optional[Callable[[int, int, float, float, float], None]] = None,
) -> GAOptimizationResult:
    if params is None:
        params = GAParameters()

    start_time = time.time()

    is_valid, errors = network.validate()
    if not is_valid:
        return GAOptimizationResult(
            success=False,
            message=f"网络验证失败: {errors}",
            parameters=params
        )

    if not workface_branch_ids:
        return GAOptimizationResult(
            success=False,
            message="请至少选择一个工作面分支",
            parameters=params
        )

    fan_ids, damper_ids = _extract_fan_and_damper_ids(network)
    n_vars = len(fan_ids) + len(damper_ids)

    if n_vars == 0:
        return GAOptimizationResult(
            success=False,
            message="网络中没有扇风机或调节风门，无需优化",
            parameters=params
        )

    original_network = _deep_copy_network(network)
    work_network = _deep_copy_network(network)

    rng = np.random.default_rng(params.random_seed)

    initial_solution = np.ones(n_vars)
    initial_fitness, initial_power, _, _, _, _, initial_violation, initial_uniformity = _evaluate_single(
        work_network, original_network, initial_solution,
        fan_ids, damper_ids, workface_branch_ids, params
    )

    if initial_power == float('inf'):
        return GAOptimizationResult(
            success=False,
            message="初始方案网络求解失败，无法进行优化",
            parameters=params
        )

    population = _initialize_population(
        params.population_size, n_vars, fan_ids, damper_ids, params, rng
    )

    fitness_array = np.zeros(params.population_size)
    power_array = np.zeros(params.population_size)
    converged_flags = np.zeros(params.population_size, dtype=bool)
    violation_array = np.zeros(params.population_size)

    history: List[GenerationHistory] = []
    best_fitness_overall = float('inf')
    best_solution_overall = None
    best_power_overall = 0.0
    best_airflows_overall = {}
    best_pressures_overall = {}
    best_wf_airflows_overall = {}
    best_total_violation_overall = 0.0
    best_uniformity_overall = 0.0

    no_improve_count = 0
    ga_converged = False
    final_generation = 0
    convergence_reason = ''

    for gen in range(params.max_generations):
        final_generation = gen + 1

        for idx in range(params.population_size):
            fit, pow_val, airflows, pressures, wf_airflows, conv, viol, unif = _evaluate_single(
                work_network, original_network, population[idx],
                fan_ids, damper_ids, workface_branch_ids, params
            )
            fitness_array[idx] = fit
            power_array[idx] = pow_val
            converged_flags[idx] = conv
            violation_array[idx] = viol

            if fit < best_fitness_overall:
                best_fitness_overall = fit
                best_solution_overall = population[idx].copy()
                best_power_overall = pow_val
                best_airflows_overall = airflows.copy()
                best_pressures_overall = pressures.copy()
                best_wf_airflows_overall = wf_airflows.copy()
                best_total_violation_overall = viol
                best_uniformity_overall = unif

        sorted_indices = np.argsort(fitness_array)
        best_gen_fitness = fitness_array[sorted_indices[0]]
        avg_gen_fitness = np.mean(fitness_array)
        worst_gen_fitness = fitness_array[sorted_indices[-1]]
        best_gen_individual = population[sorted_indices[0]].copy()

        history.append(GenerationHistory(
            generation=gen + 1,
            best_fitness=best_gen_fitness,
            avg_fitness=avg_gen_fitness,
            worst_fitness=worst_gen_fitness,
            best_individual=best_gen_individual.copy()
        ))

        if gen > 0:
            prev_best = history[gen - 1].best_fitness
            if prev_best > 0 and abs(prev_best - best_gen_fitness) / prev_best < params.convergence_improvement:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if no_improve_count >= params.convergence_generations:
                ga_converged = True
                convergence_reason = (
                    f'连续{params.convergence_generations}代改善不足'
                    f'{params.convergence_improvement*100:.1f}%'
                )
                if progress_callback:
                    progress_callback(
                        gen + 1, params.max_generations,
                        best_gen_fitness, avg_gen_fitness, worst_gen_fitness
                    )
                break

        if progress_callback:
            progress_callback(
                gen + 1, params.max_generations,
                best_gen_fitness, avg_gen_fitness, worst_gen_fitness
            )

        new_population = np.zeros_like(population)

        for e in range(params.elitism_count):
            if e < params.population_size:
                new_population[e] = population[sorted_indices[e]].copy()

        selected = _tournament_selection(
            population, fitness_array, params.tournament_size, rng
        )

        fill_start = params.elitism_count
        while fill_start < params.population_size:
            p1_idx = rng.integers(params.population_size)
            p2_idx = rng.integers(params.population_size)
            parent1 = selected[p1_idx]
            parent2 = selected[p2_idx]

            if rng.random() < params.crossover_prob and fill_start + 1 < params.population_size:
                child1, child2 = _simulated_binary_crossover(
                    parent1, parent2, len(fan_ids), len(damper_ids), params, rng
                )
                child1 = _polynomial_mutation(
                    child1, len(fan_ids), len(damper_ids), params, rng
                )
                child2 = _polynomial_mutation(
                    child2, len(fan_ids), len(damper_ids), params, rng
                )
                new_population[fill_start] = child1
                new_population[fill_start + 1] = child2
                fill_start += 2
            else:
                child = parent1.copy()
                child = _polynomial_mutation(
                    child, len(fan_ids), len(damper_ids), params, rng
                )
                new_population[fill_start] = child
                fill_start += 1

        population = new_population

    total_time = time.time() - start_time

    if best_solution_overall is None:
        return GAOptimizationResult(
            success=False,
            message="优化过程未找到有效解",
            parameters=params,
            history=history,
            total_time=total_time,
            generations_run=final_generation,
            converged=ga_converged,
            uniformity_value=0.0,
            initial_uniformity=initial_uniformity,
            convergence_reason=convergence_reason or ('达到最大代数' if not ga_converged else ''),
        )

    n_fans = len(fan_ids)
    fan_speeds = {}
    for i, fid in enumerate(fan_ids):
        fan_speeds[fid] = float(best_solution_overall[i])

    damper_openings = {}
    for j, did in enumerate(damper_ids):
        damper_openings[did] = float(best_solution_overall[n_fans + j])

    constraint_satisfied = {}
    for wf_id in workface_branch_ids:
        wf_q = best_wf_airflows_overall.get(wf_id, 0.0)
        constraint_satisfied[wf_id] = wf_q >= params.min_airflow_threshold

    energy_saving_percent = 0.0
    if initial_power > 0 and best_power_overall < initial_power:
        energy_saving_percent = (initial_power - best_power_overall) / initial_power * 100.0

    pareto_efficiency = 0.0
    if params.optimization_mode == 'dual' and initial_power > 0:
        energy_improve = max(0.0, (initial_power - best_power_overall) / initial_power)
        uniformity_improve = 0.0
        if initial_uniformity > 0:
            uniformity_improve = max(0.0, (initial_uniformity - best_uniformity_overall) / initial_uniformity)
        pareto_efficiency = params.energy_weight * energy_improve + params.uniformity_weight * uniformity_improve

    if not convergence_reason and not ga_converged:
        convergence_reason = f'达到最大代数({params.max_generations})'

    return GAOptimizationResult(
        success=True,
        message="优化完成",
        parameters=params,
        best_solution=best_solution_overall,
        best_fitness=best_fitness_overall,
        best_power=best_power_overall,
        initial_power=initial_power,
        energy_saving_percent=energy_saving_percent,
        fan_speeds=fan_speeds,
        damper_openings=damper_openings,
        workface_airflows=best_wf_airflows_overall,
        all_airflows=best_airflows_overall,
        all_pressures=best_pressures_overall,
        history=history,
        total_time=total_time,
        generations_run=final_generation,
        converged=ga_converged,
        constraint_satisfied=constraint_satisfied,
        total_violation=best_total_violation_overall,
        uniformity_value=best_uniformity_overall,
        initial_uniformity=initial_uniformity,
        pareto_efficiency=pareto_efficiency,
        convergence_reason=convergence_reason,
    )


def export_ga_result_to_json(result: GAOptimizationResult) -> str:
    history_data = []
    for h in result.history:
        history_data.append({
            'generation': int(h.generation),
            'best_fitness': float(h.best_fitness),
            'avg_fitness': float(h.avg_fitness),
            'worst_fitness': float(h.worst_fitness),
            'best_individual': h.best_individual.tolist() if h.best_individual is not None else None
        })

    params_data = {
        'population_size': result.parameters.population_size,
        'max_generations': result.parameters.max_generations,
        'crossover_prob': result.parameters.crossover_prob,
        'mutation_prob': result.parameters.mutation_prob,
        'elitism_count': result.parameters.elitism_count,
        'tournament_size': result.parameters.tournament_size,
        'sbx_distribution_index': result.parameters.sbx_distribution_index,
        'pm_distribution_index': result.parameters.pm_distribution_index,
        'penalty_coefficient': result.parameters.penalty_coefficient,
        'min_airflow_threshold': result.parameters.min_airflow_threshold,
        'convergence_generations': result.parameters.convergence_generations,
        'convergence_improvement': result.parameters.convergence_improvement,
        'fan_speed_min': result.parameters.fan_speed_min,
        'fan_speed_max': result.parameters.fan_speed_max,
        'damper_open_min': result.parameters.damper_open_min,
        'damper_open_max': result.parameters.damper_open_max,
        'damper_max_resistance_multiplier': result.parameters.damper_max_resistance_multiplier,
        'tolerance': result.parameters.tolerance,
        'max_iterations': result.parameters.max_iterations,
        'random_seed': result.parameters.random_seed,
        'optimization_mode': result.parameters.optimization_mode,
        'energy_weight': result.parameters.energy_weight,
        'uniformity_weight': result.parameters.uniformity_weight,
    }

    wf_data = {}
    for wf_id, q in result.workface_airflows.items():
        satisfied = bool(result.constraint_satisfied.get(wf_id, False))
        wf_data[str(wf_id)] = {
            'airflow': float(q),
            'threshold': result.parameters.min_airflow_threshold,
            'satisfied': satisfied,
            'deficit': float(max(0.0, result.parameters.min_airflow_threshold - float(q)))
        }

    data = {
        'success': bool(result.success),
        'message': result.message,
        'parameters': params_data,
        'best_solution': result.best_solution.tolist() if result.best_solution is not None else None,
        'best_fitness': float(result.best_fitness),
        'best_power_W': float(result.best_power),
        'initial_power_W': float(result.initial_power),
        'energy_saving_percent': float(result.energy_saving_percent),
        'fan_speeds': {str(k): float(v) for k, v in result.fan_speeds.items()},
        'damper_openings': {str(k): float(v) for k, v in result.damper_openings.items()},
        'workface_airflows': wf_data,
        'all_airflows_m3s': {str(k): float(v) for k, v in result.all_airflows.items()},
        'all_pressures_Pa': {str(k): float(v) for k, v in result.all_pressures.items()},
        'history': history_data,
        'total_time_s': float(result.total_time),
        'generations_run': int(result.generations_run),
        'converged': bool(result.converged),
        'constraint_satisfied': {str(k): bool(v) for k, v in result.constraint_satisfied.items()},
        'total_violation': float(result.total_violation),
        'uniformity_value': float(result.uniformity_value),
        'initial_uniformity': float(result.initial_uniformity),
        'pareto_efficiency': float(result.pareto_efficiency),
        'convergence_reason': result.convergence_reason,
    }

    return json.dumps(data, indent=2, ensure_ascii=False)


def import_ga_result_from_json(json_str: str) -> GAOptimizationResult:
    data = json.loads(json_str)

    p = data.get('parameters', {})
    params = GAParameters(
        population_size=p.get('population_size', 50),
        max_generations=p.get('max_generations', 100),
        crossover_prob=p.get('crossover_prob', 0.8),
        mutation_prob=p.get('mutation_prob', 0.1),
        elitism_count=p.get('elitism_count', 2),
        tournament_size=p.get('tournament_size', 3),
        sbx_distribution_index=p.get('sbx_distribution_index', 20.0),
        pm_distribution_index=p.get('pm_distribution_index', 20.0),
        penalty_coefficient=p.get('penalty_coefficient', 10000.0),
        min_airflow_threshold=p.get('min_airflow_threshold', 4.0),
        convergence_generations=p.get('convergence_generations', 20),
        convergence_improvement=p.get('convergence_improvement', 0.001),
        fan_speed_min=p.get('fan_speed_min', 0.5),
        fan_speed_max=p.get('fan_speed_max', 1.2),
        damper_open_min=p.get('damper_open_min', 0.0),
        damper_open_max=p.get('damper_open_max', 1.0),
        damper_max_resistance_multiplier=p.get('damper_max_resistance_multiplier', 50.0),
        tolerance=p.get('tolerance', 0.001),
        max_iterations=p.get('max_iterations', 500),
        random_seed=p.get('random_seed'),
        optimization_mode=p.get('optimization_mode', 'single'),
        energy_weight=p.get('energy_weight', 1.0),
        uniformity_weight=p.get('uniformity_weight', 0.0),
    )

    history = []
    for h in data.get('history', []):
        best_ind = None
        if h.get('best_individual') is not None:
            best_ind = np.array(h['best_individual'])
        history.append(GenerationHistory(
            generation=int(h['generation']),
            best_fitness=float(h['best_fitness']),
            avg_fitness=float(h['avg_fitness']),
            worst_fitness=float(h['worst_fitness']),
            best_individual=best_ind,
        ))

    best_solution = None
    if data.get('best_solution') is not None:
        best_solution = np.array(data['best_solution'])

    wf_airflows = {}
    wf_data = data.get('workface_airflows', {})
    for k, v in wf_data.items():
        wf_airflows[int(k)] = v.get('airflow', 0.0) if isinstance(v, dict) else float(v)

    constraint_satisfied = {}
    for k, v in data.get('constraint_satisfied', {}).items():
        constraint_satisfied[int(k)] = bool(v)

    return GAOptimizationResult(
        success=bool(data.get('success', False)),
        message=data.get('message', ''),
        parameters=params,
        best_solution=best_solution,
        best_fitness=float(data.get('best_fitness', float('inf'))),
        best_power=float(data.get('best_power_W', 0.0)),
        initial_power=float(data.get('initial_power_W', 0.0)),
        energy_saving_percent=float(data.get('energy_saving_percent', 0.0)),
        fan_speeds={int(k): float(v) for k, v in data.get('fan_speeds', {}).items()},
        damper_openings={int(k): float(v) for k, v in data.get('damper_openings', {}).items()},
        workface_airflows=wf_airflows,
        all_airflows={int(k): float(v) for k, v in data.get('all_airflows_m3s', {}).items()},
        all_pressures={int(k): float(v) for k, v in data.get('all_pressures_Pa', {}).items()},
        history=history,
        total_time=float(data.get('total_time_s', 0.0)),
        generations_run=int(data.get('generations_run', 0)),
        converged=bool(data.get('converged', False)),
        constraint_satisfied=constraint_satisfied,
        total_violation=float(data.get('total_violation', 0.0)),
        uniformity_value=float(data.get('uniformity_value', 0.0)),
        initial_uniformity=float(data.get('initial_uniformity', 0.0)),
        pareto_efficiency=float(data.get('pareto_efficiency', 0.0)),
        convergence_reason=data.get('convergence_reason', ''),
    )
