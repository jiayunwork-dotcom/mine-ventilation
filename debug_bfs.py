#!/usr/bin/env python3
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.network import VentilationNetwork
from core.hardy_cross import hardy_cross_solve
from core.resistance import calculate_all_branch_resistances, calculate_network_natural_pressures, calculate_pressure_drop, calculate_fan_pressure

with open('data/sample_network.json', 'r') as f:
    data = json.load(f)
network = VentilationNetwork.from_dict(data)

resistances = calculate_all_branch_resistances(network)
natural_pressures = calculate_network_natural_pressures(network)

hc_air, hc_pres, hc_info = hardy_cross_solve(network)

print('Resistances:')
for bid in sorted(resistances.keys()):
    branch = network.get_branch(bid)
    print(f'  分支{bid} ({branch.from_node}->{branch.to_node}): r={resistances[bid]:.6f}, atm={branch.is_atmospheric}, fan={branch.has_fan}')

print('\nNatural pressures:')
for bid in sorted(natural_pressures.keys()):
    print(f'  分支{bid}: h_n={natural_pressures[bid]:.4f}')

print('\nHC airflows and pressure drops:')
for bid in sorted(hc_air.keys()):
    branch = network.get_branch(bid)
    q = hc_air[bid]
    r = resistances[bid]
    h_n = natural_pressures.get(bid, 0.0)
    h_r = calculate_pressure_drop(r, q)
    h_fan = 0.0
    if branch.has_fan and branch.fan_params:
        h_fan = calculate_fan_pressure(branch.fan_params, abs(q))
        if q < 0:
            h_fan = -h_fan
    C = h_r - h_fan + h_n
    print(f'  分支{bid}: Q={q:.6f}, h_r={h_r:.4f}, h_fan={h_fan:.4f}, h_n={h_n:.4f}, C={C:.4f}')

print('\nHC computed pressures (from calculate_node_pressures):')
for nid in sorted(hc_pres.keys()):
    print(f'  节点{nid}: P={hc_pres[nid]:.4f}')

# Now compute pressures manually using BFS from node 1
atm_nodes = set()
for bid, branch in network.branches.items():
    if branch.is_atmospheric:
        atm_nodes.add(branch.from_node)
        atm_nodes.add(branch.to_node)
ref_nodes = sorted(atm_nodes)
ref_node = ref_nodes[0]
print(f'\nReference nodes: {ref_nodes}, ref_node: {ref_node}')

# BFS
pressures = {nid: 0.0 for nid in sorted(network.nodes.keys())}
visited = {ref_node}
queue = [ref_node]

undirected_adj = {}
for nid in network.nodes:
    undirected_adj[nid] = []
for bid, branch in network.branches.items():
    undirected_adj[branch.from_node].append((bid, branch.to_node, 1))
    undirected_adj[branch.to_node].append((bid, branch.from_node, -1))

print('\nBFS traversal:')
while queue:
    current = queue.pop(0)
    print(f'  Visiting node {current} (P={pressures[current]:.4f})')
    for bid, neighbor, direction in undirected_adj[current]:
        if neighbor in visited:
            continue
        branch = network.get_branch(bid)
        q = hc_air[bid]
        r = resistances[bid]
        h_n = natural_pressures.get(bid, 0.0)
        h_r = calculate_pressure_drop(r, q)
        h_fan = 0.0
        if branch.has_fan and branch.fan_params:
            h_fan = calculate_fan_pressure(branch.fan_params, abs(q))
            if q < 0:
                h_fan = -h_fan
        C = h_r - h_fan + h_n
        
        if direction > 0:
            pressures[neighbor] = pressures[current] - C
        else:
            pressures[neighbor] = pressures[current] + C
        
        print(f'    -> Node {neighbor} via branch {bid} (dir={direction}): C={C:.4f}, P={pressures[neighbor]:.4f}')
        visited.add(neighbor)
        queue.append(neighbor)

print(f'\nBefore offset correction:')
for nid in sorted(pressures.keys()):
    print(f'  节点{nid}: P={pressures[nid]:.4f}')

# Offset correction
for nid in ref_nodes:
    if nid != ref_node:
        offset = pressures[nid]
        print(f'  Applying offset {offset:.4f} for reference node {nid}')
        for other_nid in sorted(network.nodes.keys()):
            if other_nid not in set(ref_nodes) or other_nid == nid:
                pressures[other_nid] -= offset
        pressures[nid] = 0.0

print(f'\nAfter offset correction:')
for nid in sorted(pressures.keys()):
    print(f'  节点{nid}: P={pressures[nid]:.4f}')
