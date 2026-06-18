from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import networkx as nx
import numpy as np


@dataclass
class Node:
    id: int
    elevation: float = 0.0
    pressure: float = 0.0
    temperature: float = 15.0

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'elevation': self.elevation,
            'pressure': self.pressure,
            'temperature': self.temperature
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Node':
        return cls(
            id=data['id'],
            elevation=data.get('elevation', 0.0),
            pressure=data.get('pressure', 0.0),
            temperature=data.get('temperature', 15.0)
        )


@dataclass
class Branch:
    id: int
    from_node: int
    to_node: int
    length: float
    area: float
    perimeter: float
    friction_coeff: float
    local_coeff: float = 0.0
    has_fan: bool = False
    fan_params: Optional[Dict] = field(default=None)
    has_damper: bool = False
    damper_resistance: float = 0.0
    is_atmospheric: bool = False
    airflow: float = 0.0
    resistance: float = 0.0
    pressure_drop: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'from_node': self.from_node,
            'to_node': self.to_node,
            'length': self.length,
            'area': self.area,
            'perimeter': self.perimeter,
            'friction_coeff': self.friction_coeff,
            'local_coeff': self.local_coeff,
            'has_fan': self.has_fan,
            'fan_params': self.fan_params,
            'has_damper': self.has_damper,
            'damper_resistance': self.damper_resistance,
            'is_atmospheric': self.is_atmospheric,
            'airflow': self.airflow,
            'resistance': self.resistance,
            'pressure_drop': self.pressure_drop
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Branch':
        return cls(
            id=data['id'],
            from_node=data['from_node'],
            to_node=data['to_node'],
            length=data['length'],
            area=data['area'],
            perimeter=data['perimeter'],
            friction_coeff=data['friction_coeff'],
            local_coeff=data.get('local_coeff', 0.0),
            has_fan=data.get('has_fan', False),
            fan_params=data.get('fan_params', None),
            has_damper=data.get('has_damper', False),
            damper_resistance=data.get('damper_resistance', 0.0),
            is_atmospheric=data.get('is_atmospheric', False),
            airflow=data.get('airflow', 0.0),
            resistance=data.get('resistance', 0.0),
            pressure_drop=data.get('pressure_drop', 0.0)
        )


class VentilationNetwork:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.branches: Dict[int, Branch] = {}
        self.graph: nx.DiGraph = nx.DiGraph()
        self.is_valid: bool = False
        self.validation_errors: List[str] = []

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.graph.add_node(node.id, elevation=node.elevation, 
                           temperature=node.temperature)

    def add_branch(self, branch: Branch) -> None:
        self.branches[branch.id] = branch
        self.graph.add_edge(branch.from_node, branch.to_node, 
                           branch_id=branch.id)

    def get_node(self, node_id: int) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_branch(self, branch_id: int) -> Optional[Branch]:
        return self.branches.get(branch_id)

    def validate(self) -> Tuple[bool, List[str]]:
        errors = []

        if not self.nodes:
            errors.append("网络中没有定义任何节点")

        if not self.branches:
            errors.append("网络中没有定义任何分支")

        for branch_id, branch in self.branches.items():
            if branch.from_node not in self.nodes:
                errors.append(f"分支 {branch_id} 的起始节点 {branch.from_node} 不存在")
            if branch.to_node not in self.nodes:
                errors.append(f"分支 {branch_id} 的终止节点 {branch.to_node} 不存在")
            if branch.from_node == branch.to_node:
                errors.append(f"分支 {branch_id} 的起止节点相同，形成自环")
            if branch.length <= 0:
                errors.append(f"分支 {branch_id} 的长度必须大于0")
            if branch.area <= 0:
                errors.append(f"分支 {branch_id} 的断面积必须大于0")
            if branch.perimeter <= 0:
                errors.append(f"分支 {branch_id} 的断面周长必须大于0")
            if branch.friction_coeff <= 0 and not branch.is_atmospheric:
                errors.append(f"分支 {branch_id} 的摩擦阻力系数必须大于0")
            if branch.has_fan and (branch.fan_params is None or 
                                 not all(k in branch.fan_params for k in ['a', 'b', 'c'])):
                errors.append(f"分支 {branch_id} 安装有扇风机但缺少Q-H特性曲线参数(a, b, c)")

        undirected_graph = self.graph.to_undirected()
        connected_components = list(nx.connected_components(undirected_graph))
        
        if len(connected_components) > 1:
            for i, comp in enumerate(connected_components, 1):
                errors.append(f"检测到不连通子图 {i}: 节点 {sorted(comp)}")

        isolated_nodes = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        if isolated_nodes:
            errors.append(f"检测到孤立节点: {sorted(isolated_nodes)}")

        self.is_valid = len(errors) == 0
        self.validation_errors = errors
        return self.is_valid, errors

    def get_independent_loops_count(self) -> int:
        return len(self.branches) - len(self.nodes) + 1

    def find_independent_loops(self) -> List[List[int]]:
        undirected_graph = self.graph.to_undirected()
        cycles = list(nx.cycle_basis(undirected_graph))
        
        loops = []
        for cycle in cycles:
            loop_branches = []
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                edge_data = self.graph.get_edge_data(u, v)
                if edge_data:
                    loop_branches.append(edge_data['branch_id'])
                else:
                    edge_data = self.graph.get_edge_data(v, u)
                    if edge_data:
                        loop_branches.append(-edge_data['branch_id'])
            loops.append(loop_branches)
        
        return loops

    def get_spanning_tree(self) -> Tuple[List[int], List[int]]:
        undirected_graph = self.graph.to_undirected()
        spanning_tree = nx.minimum_spanning_tree(undirected_graph)
        tree_edges = set(spanning_tree.edges())
        
        tree_branches = []
        chord_branches = []
        
        for branch_id, branch in self.branches.items():
            edge = (branch.from_node, branch.to_node)
            reverse_edge = (branch.to_node, branch.from_node)
            if edge in tree_edges or reverse_edge in tree_edges:
                tree_branches.append(branch_id)
            else:
                chord_branches.append(branch_id)
        
        return tree_branches, chord_branches

    def get_adjacent_branches(self, node_id: int) -> List[Tuple[int, int]]:
        adjacent = []
        for branch_id, branch in self.branches.items():
            if branch.from_node == node_id:
                adjacent.append((branch_id, 1))
            elif branch.to_node == node_id:
                adjacent.append((branch_id, -1))
        return adjacent

    def to_dict(self) -> Dict:
        return {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'branches': [branch.to_dict() for branch in self.branches.values()]
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict) -> 'VentilationNetwork':
        network = cls()
        for node_data in data.get('nodes', []):
            network.add_node(Node.from_dict(node_data))
        for branch_data in data.get('branches', []):
            network.add_branch(Branch.from_dict(branch_data))
        return network

    @classmethod
    def from_json(cls, json_str: str) -> 'VentilationNetwork':
        return cls.from_dict(json.loads(json_str))

    def update_solution(self, airflows: Dict[int, float], 
                       pressures: Dict[int, float]) -> None:
        for branch_id, airflow in airflows.items():
            if branch_id in self.branches:
                self.branches[branch_id].airflow = airflow
        
        for node_id, pressure in pressures.items():
            if node_id in self.nodes:
                self.nodes[node_id].pressure = pressure

    def clear_solution(self) -> None:
        for branch in self.branches.values():
            branch.airflow = 0.0
            branch.pressure_drop = 0.0
        for node in self.nodes.values():
            node.pressure = 0.0

    def get_branch_airflows(self) -> Dict[int, float]:
        return {bid: branch.airflow for bid, branch in self.branches.items()}

    def get_node_pressures(self) -> Dict[int, float]:
        return {nid: node.pressure for nid, node in self.nodes.items()}

    def get_air_velocity(self, branch_id: int) -> float:
        branch = self.get_branch(branch_id)
        if branch and branch.area > 0:
            return abs(branch.airflow) / branch.area
        return 0.0

    def get_total_airflow(self) -> float:
        return sum(abs(b.airflow) for b in self.branches.values())

    def get_fan_branches(self) -> List[Branch]:
        return [b for b in self.branches.values() if b.has_fan]

    def get_damper_branches(self) -> List[Branch]:
        return [b for b in self.branches.values() if b.has_damper]
