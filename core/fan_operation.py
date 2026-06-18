from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from .network import VentilationNetwork, Branch
from .resistance import (
    calculate_all_branch_resistances,
    calculate_network_natural_pressures,
    calculate_pressure_drop,
    calculate_fan_pressure
)


def fan_series(fan1_params: Dict, fan2_params: Dict) -> Dict:
    a = fan1_params.get('a', 0.0) + fan2_params.get('a', 0.0)
    b = fan1_params.get('b', 0.0) + fan2_params.get('b', 0.0)
    c = fan1_params.get('c', 0.0) + fan2_params.get('c', 0.0)
    return {'a': a, 'b': b, 'c': c, 'type': 'series'}


def fan_parallel(fan1_params: Dict, fan2_params: Dict) -> Dict:
    return {
        'fan1': fan1_params,
        'fan2': fan2_params,
        'type': 'parallel'
    }


def evaluate_parallel_fan_pressure(fan_params: Dict, airflow: float) -> float:
    fan1 = fan_params['fan1']
    fan2 = fan_params['fan2']
    
    q_half = airflow / 2.0
    h1 = calculate_fan_pressure(fan1, abs(q_half))
    h2 = calculate_fan_pressure(fan2, abs(q_half))
    
    return min(h1, h2) if airflow >= 0 else -min(h1, h2)


def evaluate_fan_pressure(fan_params: Dict, airflow: float) -> float:
    if fan_params.get('type') == 'parallel':
        return evaluate_parallel_fan_pressure(fan_params, airflow)
    return calculate_fan_pressure(fan_params, abs(airflow))


def calculate_system_curve(
    network: VentilationNetwork,
    branch_id: int,
    airflow_range: np.ndarray
) -> np.ndarray:
    branch = network.get_branch(branch_id)
    if branch is None:
        return np.zeros_like(airflow_range)

    r = branch.resistance
    if r == 0:
        from .resistance import calculate_branch_resistance
        r = calculate_branch_resistance(branch)

    return r * np.abs(airflow_range) * airflow_range


def find_operating_point(
    fan_params: Dict,
    system_resistance: float,
    q_min: float = 0.0,
    q_max: float = 1000.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100
) -> Tuple[Optional[float], Optional[float], bool]:
    def equation(q):
        h_fan = evaluate_fan_pressure(fan_params, q)
        h_system = system_resistance * abs(q) * q
        return h_fan - h_system

    def derivative(q):
        eps = 1e-6
        return (equation(q + eps) - equation(q - eps)) / (2 * eps)

    q = q_max / 2.0
    for _ in range(max_iterations):
        f = equation(q)
        df = derivative(q)
        
        if abs(f) < tolerance:
            h = evaluate_fan_pressure(fan_params, q)
            if q >= 0 and h >= 0:
                return q, h, True
            return None, None, False
        
        if abs(df) < 1e-10:
            break
            
        q_new = q - f / df
        
        if q_new < q_min:
            q_new = q_min
        elif q_new > q_max:
            q_new = q_max
            
        if abs(q_new - q) < tolerance:
            break
            
        q = q_new

    q_values = np.linspace(q_min, q_max, 1000)
    h_fan = np.array([evaluate_fan_pressure(fan_params, q) for q in q_values])
    h_system = system_resistance * np.abs(q_values) * q_values
    
    diff = h_fan - h_system
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    if len(sign_changes) > 0:
        for idx in sign_changes:
            q1, q2 = q_values[idx], q_values[idx + 1]
            f1, f2 = diff[idx], diff[idx + 1]
            
            if f2 - f1 != 0:
                q_intersect = q1 - f1 * (q2 - q1) / (f2 - f1)
                h_intersect = evaluate_fan_pressure(fan_params, q_intersect)
                
                if q_intersect >= 0 and h_intersect >= 0:
                    return q_intersect, h_intersect, True
    
    return None, None, False


def calculate_fan_operating_point(
    network: VentilationNetwork,
    branch_id: int,
    airflow: Optional[float] = None
) -> Dict:
    branch = network.get_branch(branch_id)
    if branch is None:
        return {'error': '分支不存在'}

    if not branch.has_fan or branch.fan_params is None:
        return {'error': '该分支没有安装扇风机'}

    if airflow is None:
        airflow = branch.airflow

    r = branch.resistance
    if r == 0:
        from .resistance import calculate_branch_resistance
        r = calculate_branch_resistance(branch)

    fan_params = branch.fan_params
    q_abs = abs(airflow)

    h_fan = evaluate_fan_pressure(fan_params, q_abs)
    h_system = r * q_abs * q_abs

    operating_q, operating_h, found = find_operating_point(fan_params, r)

    design_q = fan_params.get('design_q', None)
    efficiency_range = fan_params.get('efficiency_range', (0.7, 1.1))
    in_efficiency_range = False

    if design_q and operating_q is not None:
        ratio = operating_q / design_q
        in_efficiency_range = efficiency_range[0] <= ratio <= efficiency_range[1]

    shaft_power = 0.0
    static_efficiency = 0.0
    if q_abs > 0 and h_fan > 0:
        air_power = h_fan * q_abs
        efficiency = fan_params.get('efficiency', 0.75)
        if efficiency > 0:
            shaft_power = air_power / efficiency
            static_efficiency = h_system * q_abs / shaft_power if shaft_power > 0 else 0.0

    result = {
        'branch_id': branch_id,
        'current_airflow': airflow,
        'current_fan_pressure': h_fan if airflow >= 0 else -h_fan,
        'system_resistance': r,
        'system_pressure_drop': h_system if airflow >= 0 else -h_system,
        'operating_point_found': found,
        'operating_airflow': operating_q,
        'operating_pressure': operating_h,
        'in_efficiency_range': in_efficiency_range,
        'design_airflow': design_q,
        'efficiency_range': efficiency_range,
        'shaft_power': shaft_power,
        'static_efficiency': static_efficiency,
        'air_power': h_fan * q_abs
    }

    if fan_params.get('type') == 'parallel':
        result['fan_type'] = 'parallel'
    elif fan_params.get('type') == 'series':
        result['fan_type'] = 'series'
    else:
        result['fan_type'] = 'single'

    return result


def calculate_all_fan_operating_points(network: VentilationNetwork) -> Dict[int, Dict]:
    results = {}
    for branch in network.get_fan_branches():
        results[branch.id] = calculate_fan_operating_point(network, branch.id)
    return results


def calculate_total_power_consumption(network: VentilationNetwork) -> Dict:
    fan_points = calculate_all_fan_operating_points(network)
    
    total_shaft_power = 0.0
    total_air_power = 0.0
    fan_details = []

    for branch_id, point in fan_points.items():
        if 'error' not in point:
            total_shaft_power += point.get('shaft_power', 0.0)
            total_air_power += point.get('air_power', 0.0)
            fan_details.append({
                'branch_id': branch_id,
                'shaft_power': point.get('shaft_power', 0.0),
                'air_power': point.get('air_power', 0.0),
                'efficiency': point.get('static_efficiency', 0.0)
            })

    total_airflow = network.get_total_airflow()
    specific_power = 0.0
    if total_airflow > 0:
        specific_power = total_shaft_power / total_airflow

    return {
        'total_shaft_power': total_shaft_power,
        'total_air_power': total_air_power,
        'total_efficiency': total_air_power / total_shaft_power if total_shaft_power > 0 else 0.0,
        'total_airflow': total_airflow,
        'specific_power': specific_power,
        'fan_details': fan_details
    }


def check_fan_adequacy(network: VentilationNetwork) -> List[Dict]:
    warnings = []
    fan_points = calculate_all_fan_operating_points(network)

    for branch_id, point in fan_points.items():
        if 'error' in point:
            continue

        if not point['operating_point_found']:
            branch = network.get_branch(branch_id)
            warnings.append({
                'branch_id': branch_id,
                'type': 'insufficient_capacity',
                'message': f'分支 {branch_id} 的扇风机能力不足以克服系统阻力',
                'system_resistance': point['system_resistance'],
                'max_pressure': evaluate_fan_pressure(
                    network.get_branch(branch_id).fan_params, 
                    network.get_branch(branch_id).fan_params.get('design_q', 50)
                )
            })

    return warnings
