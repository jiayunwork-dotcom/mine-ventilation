from __future__ import annotations
from typing import Dict, Optional
import numpy as np

from .network import VentilationNetwork, Branch, Node

AIR_DENSITY = 1.2
GRAVITY = 9.81


def calculate_friction_resistance(
    friction_coeff: float,
    length: float,
    perimeter: float,
    area: float
) -> float:
    if area <= 0:
        raise ValueError("断面积必须大于0")
    return friction_coeff * length * perimeter / (area ** 3)


def calculate_local_resistance(
    local_coeff: float,
    area: float
) -> float:
    if area <= 0:
        raise ValueError("断面积必须大于0")
    return local_coeff / (2 * area ** 2)


def calculate_branch_resistance(branch: Branch) -> float:
    if branch.is_atmospheric:
        return 0.0
    
    r_friction = calculate_friction_resistance(
        branch.friction_coeff,
        branch.length,
        branch.perimeter,
        branch.area
    )
    r_local = calculate_local_resistance(
        branch.local_coeff,
        branch.area
    )
    r_total = r_friction + r_local + branch.damper_resistance
    return r_total


def calculate_all_branch_resistances(network: VentilationNetwork) -> Dict[int, float]:
    resistances = {}
    for branch_id, branch in network.branches.items():
        r = calculate_branch_resistance(branch)
        branch.resistance = r
        resistances[branch_id] = r
    return resistances


def calculate_resistance(
    friction_coeff: float,
    length: float,
    perimeter: float,
    area: float,
    local_coeff: float = 0.0,
    damper_resistance: float = 0.0
) -> float:
    r_friction = calculate_friction_resistance(friction_coeff, length, perimeter, area)
    r_local = calculate_local_resistance(local_coeff, area)
    return r_friction + r_local + damper_resistance


def calculate_pressure_drop(
    resistance: float,
    airflow: float
) -> float:
    return resistance * airflow * abs(airflow)


def calculate_natural_pressure(
    node1: Node,
    node2: Node,
    air_density: float = AIR_DENSITY,
    gravity: float = GRAVITY
) -> float:
    delta_z = node2.elevation - node1.elevation
    t1 = node1.temperature + 273.15
    t2 = node2.temperature + 273.15
    t_avg = (t1 + t2) / 2.0
    
    if t_avg <= 0:
        return 0.0
    
    h_n = air_density * gravity * delta_z * (t1 - t2) / t_avg
    return h_n


def calculate_network_natural_pressures(network: VentilationNetwork) -> Dict[int, float]:
    natural_pressures = {}
    for branch_id, branch in network.branches.items():
        node_from = network.get_node(branch.from_node)
        node_to = network.get_node(branch.to_node)
        if node_from and node_to:
            h_n = calculate_natural_pressure(node_from, node_to)
            natural_pressures[branch_id] = h_n
    return natural_pressures


def calculate_fan_pressure(
    fan_params: Dict,
    airflow: float
) -> float:
    a = fan_params.get('a', 0.0)
    b = fan_params.get('b', 0.0)
    c = fan_params.get('c', 0.0)
    return a + b * airflow + c * airflow ** 2


def calculate_branch_pressure_drop(
    branch: Branch,
    airflow: Optional[float] = None,
    include_fan: bool = True,
    natural_pressure: float = 0.0
) -> float:
    if airflow is None:
        airflow = branch.airflow
    
    r = branch.resistance
    if r == 0:
        r = calculate_branch_resistance(branch)
    
    h_resistance = calculate_pressure_drop(r, airflow)
    
    h_fan = 0.0
    if include_fan and branch.has_fan and branch.fan_params:
        h_fan = calculate_fan_pressure(branch.fan_params, abs(airflow))
        if airflow < 0:
            h_fan = -h_fan
    
    h_total = h_resistance - h_fan + natural_pressure
    return h_total


def update_branch_pressure_drops(
    network: VentilationNetwork,
    natural_pressures: Optional[Dict[int, float]] = None
) -> None:
    if natural_pressures is None:
        natural_pressures = calculate_network_natural_pressures(network)
    
    for branch_id, branch in network.branches.items():
        h_n = natural_pressures.get(branch_id, 0.0)
        branch.pressure_drop = calculate_branch_pressure_drop(
            branch,
            natural_pressure=h_n
        )


def calculate_system_resistance(
    resistances: list,
    airflows: list
) -> float:
    total_pressure = 0.0
    total_airflow_sq = 0.0
    
    for r, q in zip(resistances, airflows):
        total_pressure += r * q * abs(q)
        total_airflow_sq += q * abs(q)
    
    if total_airflow_sq == 0:
        return 0.0
    
    return total_pressure / total_airflow_sq


def calculate_velocity(airflow: float, area: float) -> float:
    if area <= 0:
        return 0.0
    return abs(airflow) / area


def calculate_reynolds_number(
    velocity: float,
    hydraulic_diameter: float,
    kinematic_viscosity: float = 15.11e-6
) -> float:
    if kinematic_viscosity <= 0:
        return 0.0
    return velocity * hydraulic_diameter / kinematic_viscosity


def calculate_hydraulic_diameter(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return 4 * area / perimeter
