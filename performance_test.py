#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试脚本
"""

import sys
import os
import json
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.network import VentilationNetwork, Node, Branch
from core.hardy_cross import hardy_cross_solve
from core.newton_raphson import newton_raphson_solve, compare_solutions


def create_large_network(n_branches: int) -> VentilationNetwork:
    """
    创建一个指定分支数的大型测试网络
    """
    network = VentilationNetwork()
    
    n_nodes = max(n_branches // 2 + 1, 2)
    
    for i in range(1, n_nodes + 1):
        node = Node(
            id=i,
            elevation=random.uniform(0, 500),
            temperature=random.uniform(10, 25)
        )
        network.add_node(node)
    
    branch_id = 1
    
    for i in range(1, n_nodes):
        branch = Branch(
            id=branch_id,
            from_node=i,
            to_node=i + 1,
            length=random.uniform(100, 1000),
            area=random.uniform(5, 20),
            perimeter=random.uniform(8, 20),
            friction_coeff=random.uniform(0.008, 0.02),
            local_coeff=random.uniform(0, 1.0),
            has_fan=False,
            has_damper=random.choice([True, False]),
            damper_resistance=random.uniform(0, 0.05)
        )
        network.add_branch(branch)
        branch_id += 1
    
    while branch_id <= n_branches:
        from_node = random.randint(1, n_nodes)
        to_node = random.randint(1, n_nodes)
        if from_node == to_node:
            continue
        
        branch = Branch(
            id=branch_id,
            from_node=from_node,
            to_node=to_node,
            length=random.uniform(50, 500),
            area=random.uniform(4, 15),
            perimeter=random.uniform(7, 18),
            friction_coeff=random.uniform(0.008, 0.02),
            local_coeff=random.uniform(0, 1.5),
            has_fan=False,
            has_damper=random.choice([True, False]),
            damper_resistance=random.uniform(0, 0.03)
        )
        network.add_branch(branch)
        branch_id += 1
    
    if n_nodes >= 2:
        atm_branch = Branch(
            id=branch_id,
            from_node=n_nodes,
            to_node=1,
            length=100.0,
            area=100.0,
            perimeter=40.0,
            friction_coeff=0.0,
            local_coeff=0.0,
            has_fan=False,
            has_damper=False,
            is_atmospheric=True
        )
        network.add_branch(atm_branch)
    
    is_valid, errors = network.validate()
    if not is_valid:
        print(f"网络验证失败: {errors}")
    
    return network


def test_sample_network():
    """
    测试示例网络
    """
    print("="*60)
    print("测试1: 示例网络")
    print("="*60)
    
    sample_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_network.json')
    with open(sample_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    network = VentilationNetwork.from_dict(data)
    
    print(f"\n网络规模: {len(network.nodes)} 节点, {len(network.branches)} 分支")
    print(f"独立回路数: {network.get_independent_loops_count()}")
    
    start = time.time()
    airflows_hc, pressures_hc, info_hc = hardy_cross_solve(network)
    hc_time = time.time() - start
    
    print(f"\nHardy-Cross:")
    print(f"  收敛: {info_hc['converged']}")
    print(f"  迭代: {info_hc['iterations']}")
    print(f"  时间: {hc_time:.3f} s")
    print(f"  残差: {info_hc['final_residual']:.6e}")
    
    start = time.time()
    airflows_nr, pressures_nr, info_nr = newton_raphson_solve(network)
    nr_time = time.time() - start
    
    print(f"\nNewton-Raphson:")
    print(f"  收敛: {info_nr['converged']}")
    print(f"  迭代: {info_nr['iterations']}")
    print(f"  时间: {nr_time:.3f} s")
    print(f"  残差: {info_nr['final_residual']:.6e}")
    
    is_consistent, max_dev, deviations = compare_solutions(airflows_hc, airflows_nr)
    
    print(f"\n结果对比:")
    print(f"  一致: {is_consistent}")
    print(f"  最大偏差: {max_dev*100:.3f}%")
    
    if is_consistent and max_dev < 0.005:
        print("✓ 示例网络测试通过！")
    else:
        print("✗ 示例网络测试失败！")
    
    return is_consistent


def test_performance(n_branches: int, method: str = 'both'):
    """
    测试指定规模网络的性能
    """
    print(f"\n{'='*60}")
    print(f"测试2: {n_branches} 分支网络")
    print(f"{'='*60}")
    
    network = create_large_network(n_branches)
    is_valid, errors = network.validate()
    
    if not is_valid:
        print(f"网络验证失败，尝试修复...")
        network = create_large_network(n_branches)
        is_valid, errors = network.validate()
    
    if not is_valid:
        print(f"网络验证失败: {errors}")
        return False
    
    print(f"\n网络规模: {len(network.nodes)} 节点, {len(network.branches)} 分支")
    print(f"独立回路数: {network.get_independent_loops_count()}")
    
    hc_ok = True
    nr_ok = True
    
    if method in ['both', 'hc']:
        start = time.time()
        airflows_hc, pressures_hc, info_hc = hardy_cross_solve(network)
        hc_time = time.time() - start
        
        print(f"\nHardy-Cross:")
        print(f"  收敛: {info_hc['converged']}")
        print(f"  迭代: {info_hc['iterations']}")
        print(f"  时间: {hc_time:.3f} s")
        print(f"  残差: {info_hc['final_residual']:.6e}")
        
        if not info_hc['converged']:
            print("✗ Hardy-Cross未收敛！")
            hc_ok = False
        elif hc_time > 3.0 and n_branches <= 200:
            print(f"✗ Hardy-Cross超时！({hc_time:.3f}s > 3s)")
            hc_ok = False
        else:
            print("✓ Hardy-Cross性能达标！")
    
    if method in ['both', 'nr']:
        start = time.time()
        airflows_nr, pressures_nr, info_nr = newton_raphson_solve(network)
        nr_time = time.time() - start
        
        nr_converged = info_nr['converged']
        nr_iterations = info_nr['iterations']
        nr_residual = info_nr['final_residual']
        
        print(f"\nNewton-Raphson:")
        print(f"  收敛: {nr_converged}")
        print(f"  迭代: {nr_iterations}")
        print(f"  时间: {nr_time:.3f} s")
        print(f"  残差: {nr_residual:.6e}")
        
        if nr_time > 1.0 and n_branches <= 200:
            print(f"✗ Newton-Raphson超时！({nr_time:.3f}s > 1s)")
            nr_ok = False
        else:
            print("✓ Newton-Raphson性能达标！")
        
        if method == 'both' and hc_ok:
            is_consistent, max_dev, deviations = compare_solutions(airflows_hc, airflows_nr)
            print(f"\n结果对比:")
            print(f"  一致: {is_consistent}")
            print(f"  最大偏差: {max_dev*100:.3f}%")
            
            if max_dev > 0.005:
                print("✗ 两种方法偏差超过0.5%！")
                return False
            else:
                print("✓ 两种方法一致！")
                nr_ok = True
    
    return hc_ok and nr_ok


def main():
    print("\n" + "="*60)
    print("矿井通风网络计算工具 - 性能测试")
    print("="*60)
    
    all_passed = True
    
    all_passed &= test_sample_network()
    
    for size in [50, 100, 200]:
        try:
            all_passed &= test_performance(size)
        except Exception as e:
            print(f"✗ {size}分支网络测试异常: {e}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败！")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
