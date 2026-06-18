from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import time
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.network import VentilationNetwork, Node, Branch
from core.resistance import (
    calculate_all_branch_resistances,
    calculate_network_natural_pressures,
    update_branch_pressure_drops,
    calculate_branch_resistance
)
from core.hardy_cross import hardy_cross_solve
from core.newton_raphson import newton_raphson_solve, compare_solutions
from core.fan_operation import (
    calculate_fan_operating_point,
    calculate_all_fan_operating_points,
    calculate_total_power_consumption,
    check_fan_adequacy,
    fan_series,
    fan_parallel
)
from core.sensitivity import sensitivity_analysis, short_circuit_analysis
from core.optimization import generate_optimization_suggestions, generate_summary_report
from visualization.network_plot import (
    plot_network,
    plot_sensitivity_results,
    plot_convergence_history,
    plot_airflow_distribution,
    plot_time_series_airflows,
    plot_time_series_pressures,
    plot_rule_preview,
    plot_reliability_heatmap,
    plot_airflow_distribution_histogram,
    plot_weak_branch_distribution,
    plot_critical_branches,
    plot_redundancy_greedy_curve,
    plot_network_with_redundancy,
)
from visualization.fan_plot import (
    plot_fan_operating_point,
    plot_multiple_fan_curves,
    plot_system_curve_comparison
)
from report.pdf_generator import generate_pdf_report
from core.time_series import (
    ChangeMode,
    ParameterType,
    ChangeRule,
    TimeSeriesResult,
    run_time_series_simulation,
    get_solution_at_time,
    compute_parameter_factor,
)
from core.reliability import (
    run_monte_carlo_simulation,
    generate_reliability_heatmap,
    identify_critical_branches,
    export_reliability_report_to_json,
    ReliabilityAnalysisResult,
    identify_bottleneck_branches,
    generate_redundant_candidates,
    evaluate_all_candidates,
    greedy_combine_redundancy,
    export_redundancy_design_to_json,
    RedundancyDesignResult,
)
from core.genetic_optimization import (
    run_genetic_optimization,
    GAParameters,
    GAOptimizationResult,
    export_ga_result_to_json,
)
from visualization.ga_plot import (
    plot_ga_convergence_curve,
    plot_ga_radar_chart,
    plot_ga_airflow_comparison,
    plot_ga_decision_variables_bar,
    plot_ga_power_comparison,
)

st.set_page_config(
    page_title='矿井通风网络模拟系统',
    page_icon='⛏️',
    layout='wide',
    initial_sidebar_state='expanded'
)

if 'network' not in st.session_state:
    st.session_state.network = None
if 'solver_results' not in st.session_state:
    st.session_state.solver_results = None
if 'solver_info' not in st.session_state:
    st.session_state.solver_info = None
if 'solver_method' not in st.session_state:
    st.session_state.solver_method = None
if 'sensitivity_results' not in st.session_state:
    st.session_state.sensitivity_results = None
if 'short_circuit_results' not in st.session_state:
    st.session_state.short_circuit_results = None
if 'optimization_suggestions' not in st.session_state:
    st.session_state.optimization_suggestions = None
if 'ts_rules' not in st.session_state:
    st.session_state.ts_rules = []
if 'ts_result' not in st.session_state:
    st.session_state.ts_result = None
if 'ts_perf_stats' not in st.session_state:
    st.session_state.ts_perf_stats = None
if 'ts_selected_time' not in st.session_state:
    st.session_state.ts_selected_time = 0.0
if 'ts_rule_counter' not in st.session_state:
    st.session_state.ts_rule_counter = 0
if 'ts_compare_branches' not in st.session_state:
    st.session_state.ts_compare_branches = []
if 'ts_compare_nodes' not in st.session_state:
    st.session_state.ts_compare_nodes = []
if 'reliability_result' not in st.session_state:
    st.session_state.reliability_result = None
if 'reliability_heatmap' not in st.session_state:
    st.session_state.reliability_heatmap = None
if 'reliability_critical' not in st.session_state:
    st.session_state.reliability_critical = None
if 'reliability_params' not in st.session_state:
    st.session_state.reliability_params = None
if 'redundancy_design_result' not in st.session_state:
    st.session_state.redundancy_design_result = None
if 'redundancy_params' not in st.session_state:
    st.session_state.redundancy_params = None
if 'ga_result' not in st.session_state:
    st.session_state.ga_result = None
if 'ga_params' not in st.session_state:
    st.session_state.ga_params = None
if 'ga_workface_ids' not in st.session_state:
    st.session_state.ga_workface_ids = None


def load_sample_network():
    sample_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_network.json')
    if os.path.exists(sample_path):
        with open(sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        network = VentilationNetwork.from_dict(data)
        return network
    return None


def create_network_from_dataframes(nodes_df: pd.DataFrame, branches_df: pd.DataFrame) -> VentilationNetwork:
    network = VentilationNetwork()

    for _, row in nodes_df.iterrows():
        node = Node(
            id=int(row['id']),
            elevation=float(row.get('elevation', 0.0)),
            temperature=float(row.get('temperature', 15.0))
        )
        network.add_node(node)

    for _, row in branches_df.iterrows():
        fan_params = None
        if str(row.get('has_fan', False)).lower() in ['true', '1', 'yes']:
            fan_params = {
                'a': float(row.get('fan_a', 0.0)),
                'b': float(row.get('fan_b', 0.0)),
                'c': float(row.get('fan_c', 0.0)),
                'design_q': float(row.get('fan_design_q', 30.0)),
                'efficiency': float(row.get('fan_efficiency', 0.75)),
                'efficiency_range': [0.7, 1.1]
            }

        branch = Branch(
            id=int(row['id']),
            from_node=int(row['from_node']),
            to_node=int(row['to_node']),
            length=float(row['length']),
            area=float(row['area']),
            perimeter=float(row['perimeter']),
            friction_coeff=float(row['friction_coeff']),
            local_coeff=float(row.get('local_coeff', 0.0)),
            has_fan=str(row.get('has_fan', False)).lower() in ['true', '1', 'yes'],
            fan_params=fan_params,
            has_damper=str(row.get('has_damper', False)).lower() in ['true', '1', 'yes'],
            damper_resistance=float(row.get('damper_resistance', 0.0)),
            is_atmospheric=str(row.get('is_atmospheric', False)).lower() in ['true', '1', 'yes']
        )
        network.add_branch(branch)

    return network


def network_definition_tab():
    st.header('网络拓扑定义')

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button('📂 加载示例网络', use_container_width=True, type='primary'):
            network = load_sample_network()
            if network:
                st.session_state.network = network
                st.success('示例网络加载成功！')
            else:
                st.error('未找到示例网络文件')

    with col2:
        uploaded_file = st.file_uploader('📤 导入JSON文件', type=['json'], key='json_upload')
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                network = VentilationNetwork.from_dict(data)
                st.session_state.network = network
                st.success('网络文件导入成功！')
            except Exception as e:
                st.error(f'文件解析错误: {str(e)}')

    with col3:
        if st.session_state.network is not None:
            json_str = st.session_state.network.to_json()
            st.download_button(
                '📥 导出当前网络',
                data=json_str,
                file_name='ventilation_network.json',
                mime='application/json',
                use_container_width=True
            )

    if st.session_state.network is None:
        st.info('请加载示例网络、导入JSON文件或手动定义网络')

        st.subheader('手动定义网络')

        with st.expander('节点定义', expanded=True):
            node_count = st.number_input('节点数量', min_value=1, value=5, step=1)
            node_data = []
            for i in range(node_count):
                col_n1, col_n2, col_n3, col_n4 = st.columns([1, 2, 2, 2])
                with col_n1:
                    st.write(f'节点 {i+1}')
                with col_n2:
                    n_id = st.number_input(f'节点编号', min_value=1, value=i+1, key=f'n_id_{i}')
                with col_n3:
                    elev = st.number_input(f'标高 (m)', value=0.0, key=f'elev_{i}')
                with col_n4:
                    temp = st.number_input(f'温度 (°C)', value=15.0, key=f'temp_{i}')
                node_data.append({'id': n_id, 'elevation': elev, 'temperature': temp})

        with st.expander('分支定义', expanded=True):
            branch_count = st.number_input('分支数量', min_value=1, value=7, step=1)
            branch_data = []
            for i in range(branch_count):
                col_b1, col_b2, col_b3, col_b4, col_b5, col_b6 = st.columns([1, 1, 1, 1, 1, 1])
                with col_b1:
                    b_id = st.number_input(f'分支编号', min_value=1, value=i+1, key=f'b_id_{i}')
                with col_b2:
                    from_n = st.number_input(f'起节点', min_value=1, value=(i % node_count) + 1, key=f'from_{i}')
                with col_b3:
                    to_n = st.number_input(f'止节点', min_value=1, value=((i + 1) % node_count) + 1, key=f'to_{i}')
                with col_b4:
                    length = st.number_input(f'长度 (m)', value=500.0, key=f'len_{i}')
                with col_b5:
                    area = st.number_input(f'断面积 (m²)', value=10.0, key=f'area_{i}')
                with col_b6:
                    perimeter = st.number_input(f'周长 (m)', value=13.0, key=f'peri_{i}')

                col_b7, col_b8, col_b9, col_b10, col_b11 = st.columns([1, 1, 1, 1, 1])
                with col_b7:
                    fric = st.number_input(f'摩擦阻力系数', value=0.012, key=f'fric_{i}', format='%.4f')
                with col_b8:
                    local = st.number_input(f'局部阻力系数', value=0.5, key=f'local_{i}')
                with col_b9:
                    has_fan = st.checkbox(f'有扇风机', key=f'fan_{i}')
                with col_b10:
                    has_damper = st.checkbox(f'有调节风门', key=f'damper_{i}')
                with col_b11:
                    is_atm = st.checkbox(f'大气分支', key=f'atm_{i}')

                if has_fan:
                    col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 1])
                    with col_f1:
                        fa = st.number_input(f'Q-H参数 a', value=500.0, key=f'fa_{i}')
                    with col_f2:
                        fb = st.number_input(f'Q-H参数 b', value=-2.5, key=f'fb_{i}')
                    with col_f3:
                        fc = st.number_input(f'Q-H参数 c', value=-0.005, key=f'fc_{i}', format='%.4f')
                    with col_f4:
                        fdq = st.number_input(f'设计风量 (m³/s)', value=30.0, key=f'fdq_{i}')

                damper_r = 0.0
                if has_damper:
                    damper_r = st.number_input(f'风门阻力 (Ns²/m⁸)', value=0.01, key=f'dr_{i}', format='%.4f')

                branch_data.append({
                    'id': b_id,
                    'from_node': from_n,
                    'to_node': to_n,
                    'length': length,
                    'area': area,
                    'perimeter': perimeter,
                    'friction_coeff': fric,
                    'local_coeff': local,
                    'has_fan': has_fan,
                    'has_damper': has_damper,
                    'damper_resistance': damper_r,
                    'is_atmospheric': is_atm,
                    'fan_a': fa if has_fan else 0.0,
                    'fan_b': fb if has_fan else 0.0,
                    'fan_c': fc if has_fan else 0.0,
                    'fan_design_q': fdq if has_fan else 30.0
                })

        if st.button('🛠️ 构建网络', type='primary', use_container_width=True):
            try:
                nodes_df = pd.DataFrame(node_data)
                branches_df = pd.DataFrame(branch_data)
                network = create_network_from_dataframes(nodes_df, branches_df)
                st.session_state.network = network
                st.success('网络构建成功！')
            except Exception as e:
                st.error(f'网络构建失败: {str(e)}')

    if st.session_state.network is not None:
        network = st.session_state.network

        st.subheader('网络验证')
        is_valid, errors = network.validate()

        if is_valid:
            st.success('✅ 网络拓扑验证通过，网络连通')
        else:
            st.error('❌ 网络拓扑存在问题：')
            for error in errors:
                st.write(f'  - {error}')

        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.metric('节点数量', len(network.nodes))
        with col_info2:
            st.metric('分支数量', len(network.branches))
        with col_info3:
            st.metric('独立回路数', network.get_independent_loops_count())
        with col_info4:
            st.metric('扇风机数量', len(network.get_fan_branches()))

        tab1, tab2 = st.tabs(['节点参数', '分支参数'])

        with tab1:
            node_data = []
            for nid in sorted(network.nodes.keys()):
                node = network.get_node(nid)
                node_data.append({
                    '节点编号': node.id,
                    '标高 (m)': node.elevation,
                    '温度 (°C)': node.temperature,
                    '风压 (Pa)': f'{node.pressure:.2f}'
                })
            st.dataframe(pd.DataFrame(node_data), use_container_width=True, hide_index=True)

        with tab2:
            branch_data = []
            for bid in sorted(network.branches.keys()):
                branch = network.get_branch(bid)
                r = calculate_branch_resistance(branch)
                branch_data.append({
                    '分支编号': branch.id,
                    '起节点': branch.from_node,
                    '止节点': branch.to_node,
                    '长度 (m)': branch.length,
                    '断面积 (m²)': branch.area,
                    '周长 (m)': branch.perimeter,
                    '摩擦阻力系数': f'{branch.friction_coeff:.4f}',
                    '局部阻力系数': f'{branch.local_coeff:.4f}',
                    '综合阻力 (Ns²/m⁸)': f'{r:.6f}',
                    '扇风机': '是' if branch.has_fan else '否',
                    '调节风门': '是' if branch.has_damper else '否',
                    '大气分支': '是' if branch.is_atmospheric else '否'
                })
            st.dataframe(pd.DataFrame(branch_data), use_container_width=True, hide_index=True)

        if st.button('🔄 清除网络', type='secondary'):
            st.session_state.network = None
            st.session_state.solver_results = None
            st.session_state.solver_info = None
            st.session_state.sensitivity_results = None
            st.session_state.short_circuit_results = None
            st.session_state.optimization_suggestions = None
            st.rerun()


def solver_tab():
    st.header('求解计算')

    if st.session_state.network is None:
        st.warning('请先在"网络定义"模块中定义或加载通风网络')
        return

    network = st.session_state.network
    is_valid, errors = network.validate()

    if not is_valid:
        st.error('网络存在拓扑问题，无法进行求解：')
        for error in errors:
            st.write(f'  - {error}')
        return

    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])

    with col_s1:
        method = st.selectbox(
            '选择求解方法',
            ['Hardy-Cross迭代法', 'Newton-Raphson节点风压法', '两种方法对比'],
            index=0
        )

    with col_s2:
        tolerance = st.number_input(
            '收敛阈值',
            min_value=1e-6,
            max_value=0.1,
            value=0.001,
            format='%.6f',
            step=1e-4
        )

    with col_s3:
        max_iter = st.number_input(
            '最大迭代次数',
            min_value=10,
            max_value=10000,
            value=1000,
            step=100
        )

    run_solver = st.button('🚀 开始求解', type='primary', use_container_width=True)

    if run_solver or st.session_state.solver_results is not None:
        if run_solver:
            network.clear_solution()
            st.session_state.solver_results = None
            st.session_state.solver_info = None
            st.session_state.optimization_suggestions = None

            with st.spinner('正在求解通风网络...'):
                start_time = time.time()

                if method in ['Hardy-Cross迭代法', '两种方法对比']:
                    airflows_hc, pressures_hc, info_hc = hardy_cross_solve(
                        network,
                        tolerance=tolerance,
                        max_iterations=max_iter
                    )
                    hc_time = time.time() - start_time
                    info_hc['method'] = 'Hardy-Cross'
                    info_hc['tolerance'] = tolerance
                    info_hc['solve_time'] = hc_time

                if method in ['Newton-Raphson节点风压法', '两种方法对比']:
                    start_time_nr = time.time()
                    airflows_nr, pressures_nr, info_nr = newton_raphson_solve(
                        network,
                        tolerance=tolerance,
                        max_iterations=max_iter
                    )
                    nr_time = time.time() - start_time_nr
                    info_nr['method'] = 'Newton-Raphson'
                    info_nr['tolerance'] = tolerance
                    info_nr['solve_time'] = nr_time

                if method == '两种方法对比':
                    is_consistent, max_dev, deviations = compare_solutions(
                        airflows_hc, airflows_nr, threshold=0.005
                    )

                    if is_consistent:
                        st.success(f'✅ 两种方法求解结果一致，最大偏差 {max_dev*100:.3f}% (< 0.5%)')
                    else:
                        st.warning(f'⚠️ 两种方法求解结果偏差较大，最大偏差 {max_dev*100:.3f}% (> 0.5%)，可能存在数值问题')

                    st.session_state.solver_results = {
                        'hardy_cross': {'airflows': airflows_hc, 'pressures': pressures_hc},
                        'newton_raphson': {'airflows': airflows_nr, 'pressures': pressures_nr}
                    }
                    st.session_state.solver_info = {
                        'hardy_cross': info_hc,
                        'newton_raphson': info_nr,
                        'deviation': max_dev,
                        'is_consistent': is_consistent
                    }

                    network.update_solution(airflows_hc, pressures_hc)
                    st.session_state.solver_method = 'hardy_cross'
                else:
                    if method == 'Hardy-Cross迭代法':
                        st.session_state.solver_results = {
                            'airflows': airflows_hc,
                            'pressures': pressures_hc
                        }
                        st.session_state.solver_info = info_hc
                        network.update_solution(airflows_hc, pressures_hc)
                        st.session_state.solver_method = 'hardy_cross'
                    else:
                        st.session_state.solver_results = {
                            'airflows': airflows_nr,
                            'pressures': pressures_nr
                        }
                        st.session_state.solver_info = info_nr
                        network.update_solution(airflows_nr, pressures_nr)
                        st.session_state.solver_method = 'newton_raphson'

                calculate_all_branch_resistances(network)
                natural_pressures = calculate_network_natural_pressures(network)
                update_branch_pressure_drops(network, natural_pressures)

        if st.session_state.solver_results is not None:
            if st.session_state.solver_method == 'hardy_cross' and 'hardy_cross' in st.session_state.solver_info:
                info = st.session_state.solver_info['hardy_cross']
            elif st.session_state.solver_method == 'newton_raphson' and 'newton_raphson' in st.session_state.solver_info:
                info = st.session_state.solver_info['newton_raphson']
            else:
                info = st.session_state.solver_info

            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            with col_res1:
                st.metric(
                    '求解方法',
                    info.get('method', '未知'),
                    delta=f"{info.get('solve_time', 0)*1000:.1f} ms"
                )
            with col_res2:
                st.metric('迭代次数', info.get('iterations', 0))
            with col_res3:
                status = '✅ 收敛' if info.get('converged', False) else '❌ 未收敛'
                st.metric('收敛状态', status)
            with col_res4:
                st.metric('最终残差', f'{info.get("final_residual", 0):.2e}')

            if 'deviation' in st.session_state.solver_info:
                dev = st.session_state.solver_info['deviation']
                st.info(f'两种方法最大偏差: {dev*100:.3f}%')

            if 'residuals_history' in info and len(info['residuals_history']) > 0:
                with st.expander('收敛历史曲线', expanded=False):
                    fig = plot_convergence_history(info)
                    st.pyplot(fig, use_container_width=True)

            st.subheader('求解结果')

            tab_r1, tab_r2, tab_r3 = st.tabs(['分支风量', '节点风压', '扇风机工况'])

            with tab_r1:
                result_data = []
                for bid in sorted(network.branches.keys()):
                    branch = network.get_branch(bid)
                    velocity = network.get_air_velocity(bid)
                    result_data.append({
                        '分支编号': branch.id,
                        '起→止': f'{branch.from_node}→{branch.to_node}',
                        '风量 (m³/s)': f'{branch.airflow:.3f}',
                        '风压降 (Pa)': f'{branch.pressure_drop:.2f}',
                        '阻力 (Ns²/m⁸)': f'{branch.resistance:.6f}',
                        '风速 (m/s)': f'{velocity:.2f}'
                    })
                st.dataframe(pd.DataFrame(result_data), use_container_width=True, hide_index=True)

            with tab_r2:
                pressure_data = []
                for nid in sorted(network.nodes.keys()):
                    node = network.get_node(nid)
                    pressure_data.append({
                        '节点编号': node.id,
                        '标高 (m)': node.elevation,
                        '温度 (°C)': node.temperature,
                        '风压 (Pa)': f'{node.pressure:.2f}'
                    })
                st.dataframe(pd.DataFrame(pressure_data), use_container_width=True, hide_index=True)

            with tab_r3:
                fan_branches = network.get_fan_branches()
                if fan_branches:
                    fan_data = []
                    fan_points = calculate_all_fan_operating_points(network)
                    for bid, point in fan_points.items():
                        if 'error' not in point:
                            fan_data.append({
                                '分支编号': bid,
                                '工作风量 (m³/s)': f'{point.get("operating_airflow", 0):.2f}',
                                '工作风压 (Pa)': f'{point.get("operating_pressure", 0):.2f}',
                                '设计风量 (m³/s)': f'{point.get("design_airflow", 0):.2f}',
                                '高效区': '是' if point.get('in_efficiency_range', False) else '否',
                                '轴功率 (W)': f'{point.get("shaft_power", 0):.2f}',
                                '静压效率': f'{point.get("static_efficiency", 0)*100:.1f}%'
                            })
                    st.dataframe(pd.DataFrame(fan_data), use_container_width=True, hide_index=True)

                    fan_warnings = check_fan_adequacy(network)
                    if fan_warnings:
                        st.warning('扇风机能力警告：')
                        for warning in fan_warnings:
                            st.write(f'  ⚠️ {warning["message"]}')
                else:
                    st.info('网络中没有安装扇风机')

            power_info = calculate_total_power_consumption(network)
            st.subheader('系统能耗分析')
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1:
                st.metric('总轴功率 (W)', f'{power_info["total_shaft_power"]:.2f}')
            with col_p2:
                st.metric('总有效功率 (W)', f'{power_info["total_air_power"]:.2f}')
            with col_p3:
                st.metric('系统总效率', f'{power_info["total_efficiency"]*100:.1f}%')
            with col_p4:
                st.metric('单位风量能耗 (W/(m³/s))', f'{power_info["specific_power"]:.3f}')

            with st.expander('风量分布图', expanded=False):
                fig = plot_airflow_distribution(network)
                st.pyplot(fig, use_container_width=True)


def visualization_tab():
    st.header('可视化分析')

    if st.session_state.network is None:
        st.warning('请先在"网络定义"模块中定义或加载通风网络')
        return

    network = st.session_state.network
    has_solution = st.session_state.solver_results is not None

    if not has_solution:
        st.warning('请先在"求解计算"模块中进行求解')
        return

    sub_tab1, sub_tab2, sub_tab3 = st.tabs(['网络拓扑图', '扇风机特性', '敏感性分析'])

    with sub_tab1:
        st.subheader('通风网络拓扑图')

        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            show_airflow = st.checkbox('显示风量', value=True)
        with col_v2:
            show_pressure = st.checkbox('显示风压', value=True)
        with col_v3:
            show_fan = st.checkbox('显示扇风机图标', value=True)

        fig = plot_network(
            network,
            show_airflow=show_airflow,
            show_pressure=show_pressure,
            show_fan_icon=show_fan,
            figsize=(12, 9)
        )
        st.pyplot(fig, use_container_width=True)

    with sub_tab2:
        st.subheader('扇风机特性曲线')

        fan_branches = network.get_fan_branches()
        if not fan_branches:
            st.info('网络中没有安装扇风机')
        else:
            fan_ids = [b.id for b in fan_branches]
            selected_fan = st.selectbox('选择扇风机分支', fan_ids, index=0)

            fig = plot_fan_operating_point(network, selected_fan, figsize=(10, 8))
            st.pyplot(fig, use_container_width=True)

            if len(fan_ids) > 1:
                with st.expander('多扇风机特性对比', expanded=False):
                    fan_params_dict = {}
                    for b in fan_branches:
                        fan_params_dict[b.id] = b.fan_params
                    fig2 = plot_multiple_fan_curves(fan_params_dict)
                    st.pyplot(fig2, use_container_width=True)

    with sub_tab3:
        st.subheader('敏感性分析')

        branch_ids = sorted(network.branches.keys())
        selected_branch = st.selectbox('选择分析分支', branch_ids, index=0)

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            analysis_type = st.radio(
                '分析类型',
                ['阻力敏感性分析', '短路分析'],
                horizontal=True
            )

        if analysis_type == '阻力敏感性分析':
            with col_a2:
                resistance_range = st.slider(
                    '阻力变化范围',
                    min_value=0.1,
                    max_value=2.0,
                    value=(0.5, 1.5),
                    step=0.1
                )
            with col_a3:
                step_size = st.number_input(
                    '步长',
                    min_value=0.05,
                    max_value=0.5,
                    value=0.1,
                    step=0.05
                )

            key_branches = st.multiselect(
                '选择关键分支（显示变化曲线）',
                branch_ids,
                default=branch_ids[:5] if len(branch_ids) > 5 else branch_ids
            )

            if st.button('📊 执行敏感性分析', type='primary'):
                with st.spinner('正在进行敏感性分析...'):
                    results = sensitivity_analysis(
                        network,
                        target_branch_id=selected_branch,
                        resistance_range=resistance_range,
                        step_size=step_size,
                        method='hardy_cross',
                        key_branches=key_branches
                    )
                    st.session_state.sensitivity_results = results

            if st.session_state.sensitivity_results is not None and \
               st.session_state.sensitivity_results.get('target_branch_id') == selected_branch:
                results = st.session_state.sensitivity_results

                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.success(f'✅ 敏感性分析完成，共计算 {len(results["resistance_factors"])} 个工况')

                    fig = plot_sensitivity_results(results, key_branches=key_branches)
                    st.pyplot(fig, use_container_width=True)

                    st.subheader('敏感性指数')
                    sens_data = []
                    for bid in sorted(key_branches):
                        if bid in results['sensitivity_indices']:
                            idx = results['sensitivity_indices'][bid]
                            sens_data.append({
                                '分支编号': bid,
                                '平均绝对敏感性': f'{idx["average_absolute_sensitivity"]:.4f}',
                                '最大相对变化': f'{idx["max_relative_change"]*100:.2f}%',
                                '原始风量 (m³/s)': f'{idx["original_airflow"]:.3f}',
                                '风量范围 (m³/s)': f'{idx["airflow_range"][0]:.3f} ~ {idx["airflow_range"][1]:.3f}'
                            })
                    st.dataframe(pd.DataFrame(sens_data), use_container_width=True, hide_index=True)

        else:
            if st.button('⚡ 执行短路分析', type='primary'):
                with st.spinner('正在进行短路分析...'):
                    results = short_circuit_analysis(
                        network,
                        target_branch_id=selected_branch,
                        method='hardy_cross'
                    )
                    st.session_state.short_circuit_results = results

            if st.session_state.short_circuit_results is not None and \
               st.session_state.short_circuit_results.get('target_branch_id') == selected_branch:
                results = st.session_state.short_circuit_results

                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.success(f'✅ 短路分析完成')

                    col_sc1, col_sc2 = st.columns(2)
                    with col_sc1:
                        st.metric(
                            '总风量变化',
                            f'{results["total_airflow_change"]:.2f} m³/s',
                            delta=f'{results["total_airflow_change_percent"]:.2f}%'
                        )
                    with col_sc2:
                        st.metric(
                            '受影响分支数',
                            len(results['affected_branches'])
                        )

                    st.subheader('风量变化详情')
                    change_data = []
                    for bid in sorted(network.branches.keys()):
                        if bid in results['airflow_changes']:
                            ch = results['airflow_changes'][bid]
                            change_data.append({
                                '分支编号': bid,
                                '原始风量 (m³/s)': f'{ch["original"]:.3f}',
                                '短路后风量 (m³/s)': f'{ch["short_circuit"]:.3f}',
                                '绝对变化': f'{ch["absolute_change"]:.3f}',
                                '相对变化': f'{ch["percent_change"]:.2f}%',
                                '受显著影响': '是' if bid in results['affected_branches'] else '否'
                            })

                    df = pd.DataFrame(change_data)
                    def highlight_affected(row):
                        return ['background-color: #ffebee' if row['受显著影响'] == '是' else '' for _ in row]

                    st.dataframe(
                        df.style.apply(highlight_affected, axis=1),
                        use_container_width=True,
                        hide_index=True
                    )


def report_tab():
    st.header('报告导出')

    if st.session_state.network is None:
        st.warning('请先在"网络定义"模块中定义或加载通风网络')
        return

    network = st.session_state.network
    has_solution = st.session_state.solver_results is not None

    if not has_solution:
        st.warning('请先在"求解计算"模块中进行求解')
        return

    if st.session_state.optimization_suggestions is None:
        with st.spinner('正在生成优化建议...'):
            suggestions = generate_optimization_suggestions(network)
            st.session_state.optimization_suggestions = suggestions

    suggestions = st.session_state.optimization_suggestions

    st.subheader('系统优化建议')

    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    with col_sum1:
        st.metric('问题总数', suggestions['total_issues'])
    with col_sum2:
        st.metric('严重问题', suggestions['severity_counts'].get('high', 0), delta_color='inverse')
    with col_sum3:
        st.metric('中等问题', suggestions['severity_counts'].get('medium', 0), delta_color='off')
    with col_sum4:
        st.metric('轻微问题', suggestions['severity_counts'].get('low', 0), delta_color='off')

    with st.expander('问题分类统计', expanded=True):
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.metric('风量不均匀', suggestions['summary']['airflow_imbalance_count'])
        with col_c2:
            st.metric('风机效率问题', suggestions['summary']['fan_efficiency_issues_count'])
        with col_c3:
            st.metric('高阻力分支', suggestions['summary']['high_resistance_branches_count'])
        with col_c4:
            st.metric('风速问题', suggestions['summary']['velocity_issues_count'])

        col_c5, col_c6 = st.columns(2)
        with col_c5:
            st.metric('风门调节建议', suggestions['summary']['damper_adjustments_count'])
        with col_c6:
            st.metric('风机能力警告', suggestions['summary']['fan_capacity_warnings_count'])

    st.subheader('详细建议（按优先级排序）')
    for i, issue in enumerate(suggestions['sorted_suggestions'][:15], 1):
        severity = issue.get('severity', 'low')
        suggestion_text = issue.get('suggestion', '无具体建议')
        issue_type = issue.get('type', 'unknown')

        if severity == 'high':
            st.error(f'**[{i}]** {suggestion_text}')
        elif severity == 'medium':
            st.warning(f'**[{i}]** {suggestion_text}')
        else:
            st.info(f'**[{i}]** {suggestion_text}')

    if len(suggestions['sorted_suggestions']) > 15:
        with st.expander(f'查看更多建议 ({len(suggestions["sorted_suggestions"]) - 15} 条)', expanded=False):
            for i, issue in enumerate(suggestions['sorted_suggestions'][15:], 16):
                severity = issue.get('severity', 'low')
                suggestion_text = issue.get('suggestion', '无具体建议')
                if severity == 'high':
                    st.error(f'**[{i}]** {suggestion_text}')
                elif severity == 'medium':
                    st.warning(f'**[{i}]** {suggestion_text}')
                else:
                    st.info(f'**[{i}]** {suggestion_text}')

    st.divider()
    st.subheader('导出PDF报告')

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        include_network_plot = st.checkbox('包含网络拓扑图', value=True)
    with col_opt2:
        include_fan_plots = st.checkbox('包含扇风机工作点图', value=True)

    report_title = st.text_input('报告标题', value='矿井通风系统分析报告')

    if st.button('📄 生成PDF报告', type='primary', use_container_width=True):
        with st.spinner('正在生成PDF报告...'):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp_path = tmp.name

                solver_info = st.session_state.solver_info
                if isinstance(solver_info, dict) and 'hardy_cross' in solver_info:
                    solver_info_export = solver_info['hardy_cross']
                else:
                    solver_info_export = solver_info

                output_path = generate_pdf_report(
                    network,
                    solver_info=solver_info_export,
                    output_path=tmp_path,
                    include_network_plot=include_network_plot,
                    include_fan_plots=include_fan_plots,
                    title=report_title
                )

                with open(output_path, 'rb') as f:
                    pdf_bytes = f.read()

                st.success('✅ PDF报告生成成功！')

                st.download_button(
                    '📥 下载PDF报告',
                    data=pdf_bytes,
                    file_name='ventilation_analysis_report.pdf',
                    mime='application/pdf',
                    use_container_width=True,
                    type='primary'
                )

                os.unlink(tmp_path)

            except Exception as e:
                st.error(f'PDF生成失败: {str(e)}')
                import traceback
                st.code(traceback.format_exc())


def reliability_analysis_tab():
    st.header('🔒 通风网络可靠性分析')

    if st.session_state.network is None:
        st.warning('请先在"网络定义"模块中定义或加载通风网络')
        return

    network = st.session_state.network
    is_valid, errors = network.validate()

    if not is_valid:
        st.error('网络存在拓扑问题，无法进行分析：')
        for error in errors:
            st.write(f'  - {error}')
        return

    branch_ids = sorted(network.branches.keys())
    non_atm_branch_ids = [bid for bid in branch_ids if not network.get_branch(bid).is_atmospheric]
    fan_branch_ids = [b.id for b in network.get_fan_branches()]

    def format_branch_label(bid):
        br = network.get_branch(bid)
        if not br:
            return f'分支 {bid}'
        label = f'分支 {bid}: {br.from_node}→{br.to_node}'
        if bid in fan_branch_ids:
            label += ' (风机)'
        return label

    if st.session_state.reliability_params is None:
        default_workfaces = non_atm_branch_ids[-3:] if len(non_atm_branch_ids) >= 3 else non_atm_branch_ids
        st.session_state.reliability_params = {
            'n_simulations': 1000,
            'branch_failure_prob': 0.05,
            'fan_failure_prob': 0.02,
            'min_airflow_threshold': 4.0,
            'workface_branch_ids': default_workfaces,
            'resistance_multiplier': 10.0,
            'random_seed': 42,
            'use_parallel': True,
            'generate_heatmap': True,
            'identify_critical': True
        }

    params = st.session_state.reliability_params

    st.subheader('⚙️ 参数设置')

    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        n_simulations = st.number_input(
            '蒙特卡洛模拟次数',
            min_value=100,
            max_value=10000,
            value=params['n_simulations'],
            step=100,
            key='rel_n_sim',
            help='主模拟的次数，建议1000次以上获得稳定结果'
        )
        min_airflow_threshold = st.number_input(
            '最低通风量阈值 (m³/s)',
            min_value=0.5,
            max_value=20.0,
            value=params['min_airflow_threshold'],
            step=0.5,
            key='rel_min_q',
            help='每个工作面分支必须满足的最低风量要求'
        )
        random_seed = st.number_input(
            '随机数种子',
            min_value=0,
            max_value=99999,
            value=params['random_seed'],
            step=1,
            key='rel_seed',
            help='固定种子可复现模拟结果'
        )

    with col_p2:
        branch_failure_prob = st.slider(
            '分支故障概率',
            min_value=0.01,
            max_value=0.5,
            value=params['branch_failure_prob'],
            step=0.01,
            format='%.2f',
            key='rel_branch_prob',
            help='巷道发生冒顶/塌方等故障的概率'
        )
        fan_failure_prob = st.slider(
            '风机故障概率',
            min_value=0.01,
            max_value=0.3,
            value=params['fan_failure_prob'],
            step=0.01,
            format='%.2f',
            key='rel_fan_prob',
            help='扇风机发生停机故障的概率'
        )
        resistance_multiplier = st.number_input(
            '故障阻力倍增系数',
            min_value=2.0,
            max_value=50.0,
            value=params['resistance_multiplier'],
            step=1.0,
            key='rel_res_mult',
            help='分支故障后阻力变为正常值的多少倍'
        )

    with col_p3:
        use_parallel = st.checkbox(
            '使用多进程并行计算',
            value=params['use_parallel'],
            key='rel_parallel',
            help='开启可大幅加速模拟，建议保持开启'
        )
        generate_heatmap = st.checkbox(
            '生成可靠度热力图 (约需10秒)',
            value=params['generate_heatmap'],
            key='rel_heatmap',
            help='分析各分支对系统可靠度的影响程度'
        )
        identify_critical = st.checkbox(
            '识别关键分支 (约需5秒)',
            value=params['identify_critical'],
            key='rel_critical',
            help='找出对系统可靠度影响最大的前3条分支'
        )

    st.subheader('🏭 工作面分支标记')
    st.info('请从下方列表中选择需要检查最低通风量的工作面分支（显示格式：分支编号: 起节点→止节点）')

    workface_branch_ids = st.multiselect(
        '选择工作面分支（可多选）',
        options=non_atm_branch_ids,
        default=params['workface_branch_ids'],
        format_func=format_branch_label,
        key='rel_workfaces',
        help='选择后，系统会检查这些分支的风量是否满足最低通风量要求'
    )

    if workface_branch_ids:
        st.write(f'✅ 已选择 {len(workface_branch_ids)} 个工作面分支:')
        wf_df = pd.DataFrame([{
            '分支编号': bid,
            '起节点': network.get_branch(bid).from_node,
            '止节点': network.get_branch(bid).to_node,
            '长度 (m)': network.get_branch(bid).length,
            '断面积 (m²)': network.get_branch(bid).area,
            '扇风机': '是' if bid in fan_branch_ids else '否'
        } for bid in workface_branch_ids])
        st.dataframe(wf_df, use_container_width=True, hide_index=True)

    st.divider()

    run_analysis = st.button('🚀 开始可靠性分析', type='primary', use_container_width=True)

    if run_analysis:
        st.session_state.reliability_result = None
        st.session_state.reliability_heatmap = None
        st.session_state.reliability_critical = None

        params = {
            'n_simulations': n_simulations,
            'branch_failure_prob': branch_failure_prob,
            'fan_failure_prob': fan_failure_prob,
            'min_airflow_threshold': min_airflow_threshold,
            'workface_branch_ids': workface_branch_ids,
            'resistance_multiplier': resistance_multiplier,
            'random_seed': random_seed,
            'use_parallel': use_parallel,
            'generate_heatmap': generate_heatmap,
            'identify_critical': identify_critical
        }
        st.session_state.reliability_params = params

        if not workface_branch_ids:
            st.error('请至少选择一个工作面分支')
            return

        status_box = st.status('🔄 正在进行可靠性分析...', expanded=True, state='running')
        start_time = time.time()

        try:
            overall_progress = status_box.progress(0.0, text='初始化中...')

            n_phases = 1 + (1 if generate_heatmap else 0) + (1 if identify_critical else 0)
            phase_idx = 0

            overall_progress.progress(
                phase_idx / n_phases + 0.05 / n_phases,
                text=f'阶段 1/{n_phases}: 正在运行蒙特卡洛模拟 ({n_simulations} 次)...'
            )

            mc_progress = status_box.progress(0.0)

            def progress_cb(cur, total):
                mc_progress.progress(cur / total, text=f'蒙特卡洛模拟进度: {cur}/{total}')

            result = run_monte_carlo_simulation(
                network=network,
                n_simulations=n_simulations,
                workface_branch_ids=workface_branch_ids,
                min_airflow_threshold=min_airflow_threshold,
                branch_failure_prob=branch_failure_prob,
                fan_failure_prob=fan_failure_prob,
                resistance_multiplier=resistance_multiplier,
                random_seed=random_seed,
                tolerance=0.001,
                max_iterations=500,
                use_parallel=use_parallel,
                progress_callback=progress_cb
            )
            st.session_state.reliability_result = result

            phase_idx += 1
            overall_progress.progress(
                phase_idx / n_phases,
                text=f'✅ 蒙特卡洛模拟完成 (可靠度: {result.reliability*100:.1f}%)'
            )

            if generate_heatmap:
                non_atm = len([bid for bid in network.branches.keys()
                              if not network.get_branch(bid).is_atmospheric])
                total_heatmap_points = non_atm * 5
                overall_progress.progress(
                    phase_idx / n_phases + 0.05 / n_phases,
                    text=f'阶段 {phase_idx + 1}/{n_phases}: 正在生成可靠度热力图 ({non_atm} 分支 × 5 概率点 = {total_heatmap_points} 点)...'
                )
                hm_progress = status_box.progress(0.0)

                def heatmap_progress_cb(cur, total):
                    hm_progress.progress(cur / total, text=f'热力图计算进度: {cur}/{total} 点')

                heatmap_start = time.time()
                heatmap_data = generate_reliability_heatmap(
                    network=network,
                    workface_branch_ids=workface_branch_ids,
                    min_airflow_threshold=min_airflow_threshold,
                    resistance_multiplier=resistance_multiplier,
                    random_seed=random_seed,
                    n_simulations_per_point=100,
                    use_parallel=use_parallel,
                    progress_callback=heatmap_progress_cb
                )
                result.heatmap_data = heatmap_data
                st.session_state.reliability_heatmap = heatmap_data

                phase_idx += 1
                overall_progress.progress(
                    phase_idx / n_phases,
                    text=f'✅ 热力图生成完成 (耗时: {time.time() - heatmap_start:.1f} 秒)'
                )

            if identify_critical:
                non_atm = len([bid for bid in network.branches.keys()
                              if not network.get_branch(bid).is_atmospheric])
                overall_progress.progress(
                    phase_idx / n_phases + 0.05 / n_phases,
                    text=f'阶段 {phase_idx + 1}/{n_phases}: 正在识别关键分支 ({non_atm} 条分支, 每条 200 次模拟)...'
                )
                crit_progress = status_box.progress(0.0)

                def critical_progress_cb(cur, total):
                    crit_progress.progress(cur / total, text=f'关键分支识别进度: {cur}/{total}')

                critical_start = time.time()
                critical_branches = identify_critical_branches(
                    network=network,
                    workface_branch_ids=workface_branch_ids,
                    min_airflow_threshold=min_airflow_threshold,
                    base_branch_failure_prob=branch_failure_prob,
                    fan_failure_prob=fan_failure_prob,
                    resistance_multiplier=resistance_multiplier,
                    random_seed=random_seed,
                    n_simulations_per_branch=200,
                    top_k=3,
                    use_parallel=use_parallel,
                    progress_callback=critical_progress_cb
                )
                result.critical_branches = critical_branches
                st.session_state.reliability_critical = critical_branches

                phase_idx += 1
                overall_progress.progress(
                    phase_idx / n_phases,
                    text=f'✅ 关键分支识别完成 (耗时: {time.time() - critical_start:.1f} 秒)'
                )

            total_time = time.time() - start_time
            overall_progress.progress(1.0, text=f'🎉 全部分析完成! 总耗时: {total_time:.2f} 秒')
            status_box.update(state='complete', label=f'✅ 分析完成! 总耗时: {total_time:.2f} 秒')

        except Exception as e:
            status_box.update(state='error', label=f'❌ 分析失败: {str(e)}')
            import traceback
            status_box.code(traceback.format_exc())
            return

    if st.session_state.reliability_result is not None:
        result: ReliabilityAnalysisResult = st.session_state.reliability_result

        st.divider()
        st.subheader('📊 分析结果')

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            reliability_pct = result.reliability * 100
            if reliability_pct >= 95:
                delta_color = 'normal'
            elif reliability_pct >= 80:
                delta_color = 'off'
            else:
                delta_color = 'inverse'
            st.metric(
                '系统可靠度',
                f'{reliability_pct:.1f}%',
                delta=f'{result.valid_count}/{result.total_simulations} 合格',
                delta_color=delta_color
            )
        with col_r2:
            failure_rate = (1 - result.reliability) * 100
            st.metric(
                '失效概率',
                f'{failure_rate:.1f}%',
                delta=f'{result.total_simulations - result.valid_count} 次失效',
                delta_color='inverse'
            )
        with col_r3:
            if result.failure_stats:
                st.metric(
                    '失效时平均最小风量',
                    f'{result.failure_stats.get("mean", 0):.2f} m³/s'
                )
            else:
                st.metric('失效时平均最小风量', 'N/A')
        with col_r4:
            if result.failure_stats:
                st.metric(
                    '5%分位最小风量',
                    f'{result.failure_stats.get("5th_percentile", 0):.2f} m³/s',
                    help='95%置信度下的最小风量'
                )
            else:
                st.metric('5%分位最小风量', 'N/A')

        st.divider()

        result_tabs = st.tabs([
            '📈 风量分布',
            '🔗 薄弱分支',
            '🔥 可靠度热力图',
            '🎯 关键路径',
            '�️ 冗余设计',
            '�📋 详细数据'
        ])

        with result_tabs[0]:
            if result.failure_min_airflows:
                fig_hist = plot_airflow_distribution_histogram(
                    result.failure_min_airflows,
                    min_airflow_threshold=result.parameters['min_airflow_threshold']
                )
                st.pyplot(fig_hist, use_container_width=True)

                with st.expander('📊 风量统计详情', expanded=True):
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.write(f'**均值**: {result.failure_stats.get("mean", 0):.3f} m³/s')
                        st.write(f'**中位数**: {result.failure_stats.get("median", 0):.3f} m³/s')
                        st.write(f'**标准差**: {result.failure_stats.get("std", 0):.3f} m³/s')
                    with col_s2:
                        st.write(f'**最小值**: {result.failure_stats.get("min", 0):.3f} m³/s')
                        st.write(f'**25%分位数**: {result.failure_stats.get("25th_percentile", 0):.3f} m³/s')
                        st.write(f'**75%分位数**: {result.failure_stats.get("75th_percentile", 0):.3f} m³/s')
                    with col_s3:
                        st.write(f'**最大值**: {result.failure_stats.get("max", 0):.3f} m³/s')
                        st.write(f'**5%分位数**: {result.failure_stats.get("5th_percentile", 0):.3f} m³/s')
                        st.write(f'**95%分位数**: {result.failure_stats.get("95th_percentile", 0):.3f} m³/s')
            else:
                st.info('所有模拟场景均满足最低通风量要求，无失效场景数据')

        with result_tabs[1]:
            if result.weak_branch_distribution:
                fig_weak = plot_weak_branch_distribution(result.weak_branch_distribution)
                st.pyplot(fig_weak, use_container_width=True)

                with st.expander('📋 薄弱分支频率详情', expanded=True):
                    weak_data = []
                    for bid in sorted(result.weak_branch_frequency.keys()):
                        weak_data.append({
                            '分支编号': bid,
                            '出现次数': result.weak_branch_frequency[bid],
                            '频率 (%)': f'{result.weak_branch_distribution[bid] * 100:.1f}%'
                        })
                    st.dataframe(pd.DataFrame(weak_data), use_container_width=True, hide_index=True)
            else:
                st.info('无失效场景数据，无法分析薄弱分支')

        with result_tabs[2]:
            if result.heatmap_data is not None:
                fig_heatmap = plot_reliability_heatmap(result.heatmap_data)
                st.pyplot(fig_heatmap, use_container_width=True)
                st.caption('热力图显示各分支在不同故障概率下单独故障时，系统可靠度的下降幅度。颜色越深表示该分支对系统可靠性影响越大。')
            else:
                st.info('未生成可靠度热力图。可在参数设置中勾选"生成可靠度热力图"后重新运行分析。')

        with result_tabs[3]:
            if result.critical_branches is not None and len(result.critical_branches) > 0:
                fig_critical = plot_critical_branches(result.critical_branches)
                st.pyplot(fig_critical, use_container_width=True)

                with st.expander('🎯 关键分支详情', expanded=True):
                    crit_data = []
                    for i, cb in enumerate(result.critical_branches, 1):
                        crit_data.append({
                            '排名': i,
                            '分支编号': cb['branch_id'],
                            '基准可靠度': f'{cb["base_reliability"] * 100:.1f}%',
                            '分支故障时可靠度': f'{cb["branch_failure_reliability"] * 100:.1f}%',
                            '可靠度下降': f'{cb["reliability_drop"] * 100:.1f}%',
                            '失效次数': cb['failure_count']
                        })
                    st.dataframe(pd.DataFrame(crit_data), use_container_width=True, hide_index=True)

                    st.info('💡 **关键路径建议**: 排名前3的分支对系统可靠性影响最大，建议优先加强这些分支的维护和备用设施。')
            else:
                st.info('未识别关键分支。可在参数设置中勾选"识别关键分支"后重新运行分析。')

        with result_tabs[4]:
            st.subheader('🛠️ 冗余通风路径自动设计')

            reliability_pct = result.reliability * 100

            if st.session_state.redundancy_params is None:
                st.session_state.redundancy_params = {
                    'target_reliability': 0.95,
                    'area_shrink_ratio': 0.7,
                    'length_increase_ratio': 1.2,
                    'n_simulations_per_candidate': 500,
                    'max_redundant_branches': 5,
                    'bottleneck_top_k': 5,
                    'fixed_random_seed': 12345
                }

            red_params = st.session_state.redundancy_params

            with st.expander('⚙️ 冗余设计参数设置', expanded=True):
                rp_col1, rp_col2, rp_col3 = st.columns(3)
                with rp_col1:
                    target_reliability = st.slider(
                        '目标可靠度',
                        min_value=0.70,
                        max_value=0.999,
                        value=red_params['target_reliability'],
                        step=0.005,
                        format='%.3f',
                        help='期望达到的系统可靠度目标值'
                    )
                    area_shrink_ratio = st.slider(
                        '冗余巷道断面积缩小比例',
                        min_value=0.4,
                        max_value=0.9,
                        value=red_params['area_shrink_ratio'],
                        step=0.05,
                        format='%.2f',
                        help='新巷道断面积 = 原分支断面积 × 此比例'
                    )
                    n_sim_per_cand = st.number_input(
                        '每个候选方案的蒙特卡洛次数',
                        min_value=100,
                        max_value=2000,
                        value=red_params['n_simulations_per_candidate'],
                        step=100,
                        help='建议500次以上获得稳定结果'
                    )
                with rp_col2:
                    length_increase_ratio = st.slider(
                        '冗余巷道长度增加比例',
                        min_value=1.0,
                        max_value=2.0,
                        value=red_params['length_increase_ratio'],
                        step=0.05,
                        format='%.2f',
                        help='新巷道长度 = 原分支长度 × 此比例 (模拟绕行)'
                    )
                    bottleneck_top_k = st.slider(
                        '瓶颈分支识别数量',
                        min_value=2,
                        max_value=10,
                        value=red_params['bottleneck_top_k'],
                        step=1,
                        help='识别前K个最关键的瓶颈分支'
                    )
                    max_red_branches = st.slider(
                        '贪心算法最多添加分支数',
                        min_value=1,
                        max_value=5,
                        value=red_params['max_redundant_branches'],
                        step=1,
                        help='贪心组合优化过程最多添加的冗余分支数'
                    )
                with rp_col3:
                    fixed_seed = st.number_input(
                        '固定随机种子',
                        min_value=0,
                        max_value=999999,
                        value=red_params['fixed_random_seed'],
                        step=1,
                        help='所有评估使用相同种子，保证结果可比性'
                    )
                    st.info(
                        f'📊 **当前系统可靠度**: {reliability_pct:.1f}%\n\n'
                        f'🎯 **目标可靠度**: {target_reliability * 100:.1f}%'
                    )
                    if reliability_pct >= target_reliability * 100:
                        st.success('✅ 当前可靠度已达标，冗余设计为可选项。')
                    else:
                        gap = target_reliability * 100 - reliability_pct
                        st.warning(f'⚠️ 当前可靠度低于目标，需提升约 {gap:.1f}%。')

            red_params = {
                'target_reliability': target_reliability,
                'area_shrink_ratio': area_shrink_ratio,
                'length_increase_ratio': length_increase_ratio,
                'n_simulations_per_candidate': n_sim_per_cand,
                'max_redundant_branches': max_red_branches,
                'bottleneck_top_k': bottleneck_top_k,
                'fixed_random_seed': fixed_seed
            }
            st.session_state.redundancy_params = red_params

            st.divider()

            run_redesign = st.button(
                '🚀 开始冗余路径自动设计',
                type='primary',
                use_container_width=True,
                help='将进行瓶颈识别、候选生成、方案评估和贪心组合优化'
            )

            if run_redesign:
                st.session_state.redundancy_design_result = None

                status_box = st.status('🔄 正在进行冗余路径设计...', expanded=True, state='running')
                start_time = time.time()

                try:
                    overall_progress = status_box.progress(0.0, text='步骤1/4: 识别瓶颈分支...')

                    critical_for_bottleneck = result.critical_branches if result.critical_branches else None
                    weak_for_bottleneck = result.weak_branch_distribution if result.weak_branch_distribution else None
                    heatmap_for_bottleneck = result.heatmap_data if result.heatmap_data else None

                    bottleneck_branches = identify_bottleneck_branches(
                        network=network,
                        critical_branches=critical_for_bottleneck,
                        weak_branch_distribution=weak_for_bottleneck,
                        reliability_heatmap=heatmap_for_bottleneck,
                        top_k=bottleneck_top_k
                    )
                    overall_progress.progress(0.15, text=f'✅ 步骤1完成: 识别出 {len(bottleneck_branches)} 个瓶颈分支')

                    candidates = generate_redundant_candidates(
                        network=network,
                        bottleneck_branches=bottleneck_branches,
                        area_shrink_ratio=area_shrink_ratio,
                        length_increase_ratio=length_increase_ratio
                    )
                    overall_progress.progress(0.25, text=f'✅ 步骤2完成: 生成 {len(candidates)} 个候选冗余路径')

                    if not candidates:
                        status_box.update(state='error', label='❌ 未找到可用的冗余路径候选。可能所有瓶颈分支的起止节点之间已有其他分支。')
                        st.error('未找到可用的冗余路径候选。请尝试调整瓶颈分支数量或检查网络拓扑。')
                    else:
                        rel_params_for_eval = {
                            'base_reliability': result.reliability,
                            'workface_branch_ids': result.parameters.get('workface_branch_ids', []),
                            'branch_failure_prob': result.parameters.get('branch_failure_prob', 0.05),
                            'fan_failure_prob': result.parameters.get('fan_failure_prob', 0.02),
                            'min_airflow_threshold': result.parameters.get('min_airflow_threshold', 4.0),
                            'resistance_multiplier': result.parameters.get('resistance_multiplier', 10.0),
                            'bottleneck_branches': bottleneck_branches
                        }

                        phase2_start = 0.25
                        phase2_end = 0.60
                        total_evals = len(candidates)

                        def eval_progress_cb(cur, total):
                            frac = cur / total
                            overall_progress.progress(
                                phase2_start + (phase2_end - phase2_start) * frac,
                                text=f'步骤3/4: 评估候选方案 ({cur}/{total}) - 每方案 {n_sim_per_cand} 次模拟...'
                            )

                        use_parallel = st.session_state.reliability_params.get('use_parallel', True) if st.session_state.reliability_params else True

                        candidate_evaluations = evaluate_all_candidates(
                            base_network=network,
                            candidates=candidates,
                            reliability_params=rel_params_for_eval,
                            n_simulations_per_candidate=n_sim_per_cand,
                            fixed_seed=fixed_seed,
                            use_parallel=use_parallel,
                            overall_progress_callback=eval_progress_cb
                        )
                        overall_progress.progress(0.60, text=f'✅ 步骤3完成: {len(candidate_evaluations)} 个候选方案评估完毕')

                        phase3_start = 0.60
                        phase3_end = 1.0

                        def greedy_progress_cb(cur, total):
                            frac = cur / total if total > 0 else 1.0
                            overall_progress.progress(
                                phase3_start + (phase3_end - phase3_start) * frac,
                                text=f'步骤4/4: 贪心组合优化 (步骤 {cur}/{total})...'
                            )

                        design_result = greedy_combine_redundancy(
                            base_network=network,
                            candidate_evaluations=candidate_evaluations,
                            reliability_params=rel_params_for_eval,
                            target_reliability=target_reliability,
                            max_branches=max_red_branches,
                            n_simulations=n_sim_per_cand,
                            fixed_seed=fixed_seed,
                            use_parallel=use_parallel,
                            overall_progress_callback=greedy_progress_cb
                        )
                        st.session_state.redundancy_design_result = design_result

                        total_time = time.time() - start_time
                        overall_progress.progress(1.0, text=f'🎉 全部完成! 总耗时: {total_time:.1f} 秒')
                        status_box.update(state='complete', label=f'✅ 冗余设计完成! 总耗时: {total_time:.1f} 秒')

                except Exception as e:
                    status_box.update(state='error', label=f'❌ 冗余设计失败: {str(e)}')
                    import traceback
                    status_box.code(traceback.format_exc())

            if st.session_state.redundancy_design_result is not None:
                dr: RedundancyDesignResult = st.session_state.redundancy_design_result

                st.divider()

                red_metric_col1, red_metric_col2, red_metric_col3, red_metric_col4 = st.columns(4)
                with red_metric_col1:
                    final_pct = dr.final_reliability * 100
                    target_pct = dr.target_reliability * 100
                    if dr.target_met:
                        delta_str = f'✅ 已达标'
                        delta_color = 'normal'
                    else:
                        delta_str = f'未达标 (差{target_pct - final_pct:.1f}%)'
                        delta_color = 'inverse'
                    st.metric(
                        '最终系统可靠度',
                        f'{final_pct:.1f}%',
                        delta=delta_str,
                        delta_color=delta_color
                    )
                with red_metric_col2:
                    gain_pct = (dr.final_reliability - dr.base_reliability) * 100
                    st.metric(
                        '可靠度提升',
                        f'{gain_pct:+.1f}%',
                        delta=f'从 {dr.base_reliability * 100:.1f}% 提升',
                        delta_color='normal' if gain_pct > 0 else 'off'
                    )
                with red_metric_col3:
                    st.metric(
                        '推荐冗余分支数',
                        f'{len(dr.recommended_branches)} 条',
                        delta=f'最多 {dr.greedy_steps[-1].step if dr.greedy_steps else 0} 步',
                    )
                with red_metric_col4:
                    st.metric(
                        '累计成本估算',
                        f'{dr.total_cost:.0f}',
                        delta='长度×断面积',
                        help='成本 ≈ Σ(新巷道长度 × 断面积)'
                    )

                st.divider()

                red_subtabs = st.tabs([
                    '🎯 瓶颈分支识别',
                    '🏆 候选方案排名',
                    '📈 贪心组合收益',
                    '🌐 推荐拓扑图'
                ])

                with red_subtabs[0]:
                    st.info('**单点故障瓶颈**: 这些分支一旦故障会导致系统可靠度大幅下降，是冗余设计优先针对的目标。')
                    bottleneck_data = []
                    bottleneck_ids = []
                    for i, bn in enumerate(dr.bottleneck_branches, 1):
                        bottleneck_ids.append(bn['branch_id'])
                        bottleneck_data.append({
                            '排名': i,
                            '分支编号': bn['branch_id'],
                            '起节点': bn['from_node'],
                            '止节点': bn['to_node'],
                            '关键度评分': f'{bn["score"]:.4f}',
                            '含风机': '是' if bn.get('has_fan') else '否'
                        })
                    st.dataframe(pd.DataFrame(bottleneck_data), use_container_width=True, hide_index=True)

                    bn_styler_data = pd.DataFrame(bottleneck_data).style.applymap(
                        lambda _: 'background-color: #ffcccc; color: #c0392b; font-weight: bold',
                        subset=['分支编号']
                    )
                    st.dataframe(bn_styler_data, use_container_width=True, hide_index=True)

                with red_subtabs[1]:
                    st.info(f'**候选方案排名**: 共 {len(dr.candidate_evaluations)} 个候选，按 可靠度提升/成本比 排序，显示前5个最优方案。所有评估均使用随机种子={dr.random_seed}。')

                    rank_data = []
                    for i, ce in enumerate(dr.top_candidates, 1):
                        branch_params = ce.added_branch_params
                        rank_data.append({
                            '排名': i,
                            '候选ID': ce.candidate_id,
                            '针对瓶颈分支': ce.original_branch_id,
                            '冗余路径': f'{branch_params["from_node"]}→{branch_params["to_node"]}',
                            '方向': branch_params.get('direction_note', 'N/A'),
                            '长度(m)': f'{branch_params["length"]:.1f}',
                            '断面积(m²)': f'{branch_params["area"]:.2f}',
                            '可靠度提升': f'{ce.reliability_gain * 100:.2f}%',
                            '评估后可靠度': f'{ce.reliability_after * 100:.1f}%',
                            '成本估算': f'{ce.estimated_cost:.0f}',
                            '效益/成本比': f'{ce.benefit_cost_ratio:.6f}',
                            '模拟次数': ce.simulation_count
                        })
                    st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)

                    with st.expander('📊 效益比排序详细图表', expanded=False):
                        import matplotlib.pyplot as plt

                        top_n = min(10, len(dr.candidate_evaluations))
                        eval_for_plot = dr.candidate_evaluations[:top_n]
                        labels = [f'候选{i+1}' for i in range(len(eval_for_plot))]
                        gains = [e.reliability_gain * 100 for e in eval_for_plot]
                        ratios = [e.benefit_cost_ratio * 1000 for e in eval_for_plot]

                        fig, ax1 = plt.subplots(figsize=(12, 6))
                        x = np.arange(len(labels))
                        width = 0.35

                        bars1 = ax1.bar(x - width/2, gains, width, label='可靠度提升(%)', color='#3498db', alpha=0.8)
                        ax2 = ax1.twinx()
                        bars2 = ax2.bar(x + width/2, ratios, width, label='效益成本比(×1000)', color='#e67e22', alpha=0.8)

                        ax1.set_xlabel('候选方案', fontsize=11)
                        ax1.set_ylabel('可靠度提升 (%)', fontsize=11, color='#3498db')
                        ax2.set_ylabel('效益/成本比 (放大1000倍)', fontsize=11, color='#e67e22')
                        ax1.set_xticks(x)
                        ax1.set_xticklabels(labels, rotation=45, ha='right')
                        ax1.set_title(f'前{top_n}个候选方案 可靠度提升 & 效益成本比', fontsize=13, fontweight='bold')

                        lines1, labels1 = ax1.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                        ax1.grid(True, alpha=0.3, axis='y')
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)

                with red_subtabs[2]:
                    st.info('**贪心组合优化**: 逐步添加最优冗余分支，每一步选当前提升最大的方案，直到达标或候选用完。最多添加5条。')

                    greedy_dicts = []
                    for gs in dr.greedy_steps:
                        greedy_dicts.append({
                            'step': gs.step,
                            'cumulative_cost': gs.cumulative_cost,
                            'cumulative_reliability': gs.cumulative_reliability,
                            'reliability_increment': gs.reliability_increment
                        })
                    fig_greedy = plot_redundancy_greedy_curve(greedy_dicts, dr.target_reliability)
                    st.pyplot(fig_greedy, use_container_width=True)

                    with st.expander('📋 贪心步骤详情', expanded=True):
                        step_data = []
                        for gs in dr.greedy_steps:
                            added_branch_info = ''
                            if gs.added_branch:
                                ab = gs.added_branch
                                added_branch_info = f'冗余{gs.step}: {ab["from_node"]}→{ab["to_node"]} (L={ab["length"]:.0f}m, A={ab["area"]:.2f}m²)'
                            step_data.append({
                                '步骤': gs.step,
                                '添加候选': gs.added_candidate_id or '(初始)',
                                '新增冗余分支': added_branch_info or '无(基准)',
                                '本步提升': f'{gs.reliability_increment * 100:.2f}%' if gs.reliability_increment > 0 else '—',
                                '累计可靠度': f'{gs.cumulative_reliability * 100:.2f}%',
                                '累计成本': f'{gs.cumulative_cost:.0f}'
                            })
                        st.dataframe(pd.DataFrame(step_data), use_container_width=True, hide_index=True)

                with red_subtabs[3]:
                    st.info('**推荐冗余路径拓扑**: 红色=瓶颈分支，绿色虚线=推荐冗余巷道。')

                    if dr.recommended_branches:
                        fig_topology = plot_network_with_redundancy(
                            network=network,
                            recommended_branches=dr.recommended_branches,
                            bottleneck_branch_ids=bottleneck_ids,
                            show_airflow=True,
                            show_pressure=True
                        )
                        st.pyplot(fig_topology, use_container_width=True)
                    else:
                        st.info('贪心算法未添加任何冗余分支。可能当前可靠度已达标，或没有能带来正向提升的候选方案。')

                st.divider()
                st.subheader('📥 导出冗余设计方案')

                red_json_str = export_redundancy_design_to_json(dr)

                red_exp_col1, red_exp_col2 = st.columns([3, 1])
                with red_exp_col1:
                    st.download_button(
                        label='⬇️ 下载JSON格式冗余设计方案',
                        data=red_json_str,
                        file_name=f'redundancy_design_scheme_{int(time.time())}.json',
                        mime='application/json',
                        use_container_width=True,
                        type='primary'
                    )
                with red_exp_col2:
                    if st.button('🔄 清除冗余设计结果', use_container_width=True, key='clear_redesign'):
                        st.session_state.redundancy_design_result = None
                        st.rerun()

                with st.expander('📄 预览JSON设计方案内容', expanded=False):
                    st.code(red_json_str, language='json')

        with result_tabs[5]:
            st.subheader('前20次模拟结果')
            sim_data = []
            for r in result.simulation_results[:20]:
                sim_data.append({
                    '场景ID': r.scenario_id,
                    '是否合格': '✅ 合格' if r.is_valid else '❌ 不合格',
                    '是否收敛': '✅ 收敛' if r.converged else '❌ 未收敛',
                    '最小风量 (m³/s)': f'{r.min_airflow:.3f}',
                    '最薄弱分支': r.min_airflow_branch if r.min_airflow_branch > 0 else 'N/A',
                    '故障分支数': len(r.failed_branches),
                    '故障风机数': len(r.failed_fans),
                    '故障分支': ', '.join([str(b) for b in r.failed_branches]) if r.failed_branches else '无'
                })
            st.dataframe(pd.DataFrame(sim_data), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader('📥 导出分析报告')

        json_str = export_reliability_report_to_json(result)

        col_d1, col_d2 = st.columns([3, 1])
        with col_d1:
            st.download_button(
                label='⬇️ 下载JSON格式分析报告',
                data=json_str,
                file_name=f'reliability_analysis_report_{int(time.time())}.json',
                mime='application/json',
                use_container_width=True,
                type='primary'
            )
        with col_d2:
            if st.button('🔄 清除结果', use_container_width=True):
                st.session_state.reliability_result = None
                st.session_state.reliability_heatmap = None
                st.session_state.reliability_critical = None
                st.rerun()

        with st.expander('📄 预览JSON报告内容', expanded=False):
            st.code(json_str, language='json')


def genetic_optimization_tab():
    st.header('🧬 遗传算法优化')

    if st.session_state.network is None:
        st.warning('请先在"网络定义"模块中定义或加载通风网络')
        return

    network = st.session_state.network
    is_valid, errors = network.validate()

    if not is_valid:
        st.error('网络存在拓扑问题，无法进行优化：')
        for error in errors:
            st.write(f'  - {error}')
        return

    branch_ids = sorted(network.branches.keys())
    non_atm_branch_ids = [bid for bid in branch_ids if not network.get_branch(bid).is_atmospheric]
    fan_branch_ids = [b.id for b in network.get_fan_branches()]
    damper_branch_ids = [b.id for b in network.get_damper_branches()]
    n_decision_vars = len(fan_branch_ids) + len(damper_branch_ids)

    def format_branch_label(bid):
        br = network.get_branch(bid)
        if not br:
            return f'分支 {bid}'
        label = f'分支 {bid}: {br.from_node}→{br.to_node}'
        if bid in fan_branch_ids:
            label += ' (风机)'
        if bid in damper_branch_ids:
            label += ' (风门)'
        return label

    if n_decision_vars == 0:
        st.error('❌ 网络中没有扇风机或调节风门，无法进行优化。请在网络中至少添加一个扇风机或调节风门。')
        st.info(f'当前网络: {len(fan_branch_ids)} 个扇风机, {len(damper_branch_ids)} 个调节风门')
        return

    if st.session_state.ga_params is None:
        default_workfaces = non_atm_branch_ids[-3:] if len(non_atm_branch_ids) >= 3 else non_atm_branch_ids
        st.session_state.ga_params = {
            'population_size': 50,
            'max_generations': 100,
            'crossover_prob': 0.8,
            'mutation_prob': 0.1,
            'elitism_count': 2,
            'penalty_coefficient': 10000.0,
            'min_airflow_threshold': 4.0,
            'tournament_size': 3,
            'sbx_eta': 20.0,
            'pm_eta': 20.0,
            'convergence_generations': 20,
            'convergence_improvement': 0.001,
            'fan_speed_min': 0.5,
            'fan_speed_max': 1.2,
            'damper_max_mult': 50.0,
            'tolerance': 0.001,
            'max_iterations': 500,
            'random_seed': 42,
        }
        st.session_state.ga_workface_ids = default_workfaces

    params = st.session_state.ga_params

    st.subheader('⚙️ 遗传算法参数设置')

    param_col1, param_col2, param_col3 = st.columns(3)

    with param_col1:
        st.markdown('##### 🧬 基本进化参数')
        population_size = st.number_input(
            '种群大小',
            min_value=10, max_value=500, value=params['population_size'], step=10,
            key='ga_pop_size',
            help='每一代的个体数量，越大搜索能力越强但速度越慢'
        )
        max_generations = st.number_input(
            '最大进化代数',
            min_value=5, max_value=1000, value=params['max_generations'], step=10,
            key='ga_max_gen',
            help='进化的最大代数，建议50-200代'
        )
        crossover_prob = st.slider(
            '交叉概率 (SBX)',
            min_value=0.5, max_value=1.0, value=params['crossover_prob'], step=0.05,
            key='ga_cx_prob',
            help='两个父代个体进行交叉的概率，通常0.6-0.9'
        )
        mutation_prob = st.slider(
            '变异概率 (PM)',
            min_value=0.01, max_value=0.5, value=params['mutation_prob'], step=0.01,
            key='ga_mut_prob',
            help='每个基因发生变异的概率，通常0.05-0.2'
        )
        elitism_count = st.number_input(
            '精英保留数',
            min_value=0, max_value=10, value=params['elitism_count'], step=1,
            key='ga_elitism',
            help='每代直接保留的最优个体数，防止最优解退化'
        )
        tournament_size = st.number_input(
            '锦标赛赛组大小',
            min_value=2, max_value=10, value=params['tournament_size'], step=1,
            key='ga_tour_size',
            help='选择操作中每组竞争的个体数，通常2-5'
        )

    with param_col2:
        st.markdown('##### 📐 算子与收敛参数')
        sbx_eta = st.number_input(
            'SBX分布指数 η',
            min_value=1.0, max_value=100.0, value=params['sbx_eta'], step=1.0,
            key='ga_sbx_eta',
            help='模拟二进制交叉的分布指数，越大子代越接近父代'
        )
        pm_eta = st.number_input(
            'PM分布指数 η',
            min_value=1.0, max_value=100.0, value=params['pm_eta'], step=1.0,
            key='ga_pm_eta',
            help='多项式变异的分布指数，越大变异幅度越小'
        )
        convergence_generations = st.number_input(
            '收敛判定代数',
            min_value=5, max_value=100, value=params['convergence_generations'], step=5,
            key='ga_conv_gen',
            help='连续多少代改善小于阈值则提前终止'
        )
        convergence_improvement = st.number_input(
            '收敛改善阈值 (%)',
            min_value=0.001, max_value=5.0, value=params['convergence_improvement'] * 100, step=0.01,
            key='ga_conv_imp',
            format='%.3f',
            help='连续代数内最优适应度改善小于此百分比则判定收敛'
        ) / 100.0

    with param_col3:
        st.markdown('##### 🚧 约束与惩罚参数')
        penalty_coefficient = st.number_input(
            '约束惩罚系数',
            min_value=1.0, max_value=100000.0, value=params['penalty_coefficient'], step=100.0,
            key='ga_penalty',
            help='违反约束时的惩罚放大系数，越大越倾向满足约束'
        )
        min_airflow_threshold = st.number_input(
            '最低通风量阈值 (m³/s)',
            min_value=0.5, max_value=50.0, value=params['min_airflow_threshold'], step=0.5,
            key='ga_min_q',
            help='每个工作面分支必须满足的最低风量要求'
        )
        st.markdown('##### 🔧 决策变量范围')
        fan_speed_min = st.slider(
            '扇风机转速系数下限',
            min_value=0.1, max_value=1.0, value=params['fan_speed_min'], step=0.05,
            key='ga_fan_min',
            help='扇风机最低转速与额定转速的比值'
        )
        fan_speed_max = st.slider(
            '扇风机转速系数上限',
            min_value=1.0, max_value=2.0, value=params['fan_speed_max'], step=0.05,
            key='ga_fan_max',
            help='扇风机最高转速与额定转速的比值'
        )
        damper_max_mult = st.number_input(
            '风门最大阻力倍数',
            min_value=5.0, max_value=200.0, value=params['damper_max_mult'], step=5.0,
            key='ga_damper_mult',
            help='风门全关时，附加阻力为原阻力的多少倍'
        )
        tolerance = st.number_input(
            '求解收敛阈值',
            min_value=1e-7, max_value=0.1, value=params['tolerance'], step=1e-5,
            key='ga_tol', format='%.7f',
            help='Hardy-Cross求解器的收敛阈值'
        )
        max_iterations = st.number_input(
            '求解最大迭代数',
            min_value=50, max_value=5000, value=params['max_iterations'], step=50,
            key='ga_max_iter'
        )
        random_seed = st.number_input(
            '随机数种子',
            min_value=0, max_value=999999, value=params['random_seed'], step=1,
            key='ga_seed',
            help='固定种子可复现优化结果'
        )

    st.divider()
    st.subheader('🏭 工作面分支标记')
    st.info('请从下方列表中选择需要检查最低通风量的工作面分支（显示格式：分支编号: 起节点→止节点）')

    default_wf = st.session_state.ga_workface_ids if st.session_state.ga_workface_ids else []
    valid_defaults = [wf for wf in default_wf if wf in non_atm_branch_ids]
    workface_branch_ids = st.multiselect(
        '选择工作面分支（可多选，至少选1个）',
        options=non_atm_branch_ids,
        default=valid_defaults if valid_defaults else non_atm_branch_ids[:3],
        format_func=format_branch_label,
        key='ga_workfaces',
        help='选择后，系统会检查这些分支的风量是否满足最低通风量要求'
    )
    st.session_state.ga_workface_ids = workface_branch_ids

    if workface_branch_ids:
        st.write(f'✅ 已选择 {len(workface_branch_ids)} 个工作面分支:')
        wf_df = pd.DataFrame([{
            '分支编号': bid,
            '起节点': network.get_branch(bid).from_node,
            '止节点': network.get_branch(bid).to_node,
            '长度 (m)': network.get_branch(bid).length,
            '断面积 (m²)': network.get_branch(bid).area,
            '扇风机': '是' if bid in fan_branch_ids else '否',
            '调节风门': '是' if bid in damper_branch_ids else '否'
        } for bid in workface_branch_ids])
        st.dataframe(wf_df, use_container_width=True, hide_index=True)

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric('扇风机决策变量', f'{len(fan_branch_ids)} 个',
                  delta=f"转速范围: {fan_speed_min:.2f}~{fan_speed_max:.2f}x",
                  delta_color='off')
    with col_info2:
        st.metric('风门决策变量', f'{len(damper_branch_ids)} 个',
                  delta=f"开度范围: 0.0~1.0 (阻力{damper_max_mult:.0f}x)",
                  delta_color='off')
    with col_info3:
        estimated_evals = population_size * max_generations
        st.metric('预计评估次数', f'{estimated_evals:,} 次',
                  delta=f'{population_size}×{max_generations}',
                  delta_color='off')

    st.divider()

    new_params = {
        'population_size': int(population_size),
        'max_generations': int(max_generations),
        'crossover_prob': float(crossover_prob),
        'mutation_prob': float(mutation_prob),
        'elitism_count': int(elitism_count),
        'penalty_coefficient': float(penalty_coefficient),
        'min_airflow_threshold': float(min_airflow_threshold),
        'tournament_size': int(tournament_size),
        'sbx_eta': float(sbx_eta),
        'pm_eta': float(pm_eta),
        'convergence_generations': int(convergence_generations),
        'convergence_improvement': float(convergence_improvement),
        'fan_speed_min': float(fan_speed_min),
        'fan_speed_max': float(fan_speed_max),
        'damper_max_mult': float(damper_max_mult),
        'tolerance': float(tolerance),
        'max_iterations': int(max_iterations),
        'random_seed': int(random_seed) if random_seed > 0 else None,
    }
    st.session_state.ga_params = new_params

    run_col1, run_col2 = st.columns([3, 1])
    with run_col1:
        run_optimization = st.button(
            '🚀 开始遗传算法优化',
            type='primary',
            use_container_width=True,
            disabled=len(workface_branch_ids) == 0,
            help='搜索扇风机转速和风门开度的最优组合，最小化总能耗'
        )
    with run_col2:
        if st.button('🔄 清除优化结果', use_container_width=True):
            st.session_state.ga_result = None
            st.rerun()

    if len(workface_branch_ids) == 0:
        st.warning('⚠️ 请至少选择一个工作面分支')

    if run_optimization:
        st.session_state.ga_result = None

        ga_params_obj = GAParameters(
            population_size=int(population_size),
            max_generations=int(max_generations),
            crossover_prob=float(crossover_prob),
            mutation_prob=float(mutation_prob),
            elitism_count=int(elitism_count),
            tournament_size=int(tournament_size),
            sbx_distribution_index=float(sbx_eta),
            pm_distribution_index=float(pm_eta),
            penalty_coefficient=float(penalty_coefficient),
            min_airflow_threshold=float(min_airflow_threshold),
            convergence_generations=int(convergence_generations),
            convergence_improvement=float(convergence_improvement),
            fan_speed_min=float(fan_speed_min),
            fan_speed_max=float(fan_speed_max),
            damper_max_resistance_multiplier=float(damper_max_mult),
            tolerance=float(tolerance),
            max_iterations=int(max_iterations),
            random_seed=int(random_seed) if random_seed > 0 else None,
        )

        status_box = st.status('🔄 正在执行遗传算法优化...', expanded=True, state='running')
        start_time = time.time()

        try:
            progress_bar = status_box.progress(0.0, text='初始化中...')
            fitness_placeholder = status_box.empty()

            def progress_cb(gen, total_gen, best_fit, avg_fit, worst_fit):
                frac = gen / total_gen
                progress_bar.progress(
                    frac,
                    text=f'代数: {gen}/{total_gen} | 最优适应度: {best_fit:.1f} W | 平均: {avg_fit:.1f} W'
                )
                fitness_placeholder.info(
                    f'📊 当前进度 {gen}/{total_gen} ({frac*100:.1f}%) | '
                    f'**最优**: {best_fit:.2f} W | **平均**: {avg_fit:.2f} W | **最差**: {worst_fit:.2f} W'
                )

            result = run_genetic_optimization(
                network=network,
                workface_branch_ids=workface_branch_ids,
                params=ga_params_obj,
                progress_callback=progress_cb,
            )
            st.session_state.ga_result = result

            total_time = time.time() - start_time
            if result.success:
                status_box.update(
                    state='complete',
                    label=f'✅ 优化完成! 总耗时: {total_time:.2f} 秒 | 节能: {result.energy_saving_percent:.1f}%'
                )
            else:
                status_box.update(state='error', label=f'❌ 优化失败: {result.message}')

        except Exception as e:
            status_box.update(state='error', label=f'❌ 优化异常: {str(e)}')
            import traceback
            status_box.code(traceback.format_exc())

    if st.session_state.ga_result is not None:
        result: GAOptimizationResult = st.session_state.ga_result

        if not result.success:
            st.error(f'优化失败: {result.message}')
            return

        st.divider()
        st.subheader('📊 优化结果概览')

        all_constraints_ok = all(result.constraint_satisfied.values())
        n_satisfied = sum(1 for v in result.constraint_satisfied.values() if v)
        n_total = len(result.constraint_satisfied)

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            saving_delta = f'{result.energy_saving_percent:.1f}% 节能' if result.energy_saving_percent > 0 else '无节能'
            saving_color = 'normal' if result.energy_saving_percent > 0 else 'off'
            st.metric(
                '系统总轴功率',
                f'{result.best_power:.2f} W',
                delta=saving_delta,
                delta_color=saving_color
            )
        with res_col2:
            st.metric(
                '初始方案功率',
                f'{result.initial_power:.2f} W',
                delta=f'vs 初始 {result.initial_power - result.best_power:+.2f} W',
                delta_color='inverse'
            )
        with res_col3:
            constraint_status = '✅ 全部满足' if all_constraints_ok else f'⚠️ {n_satisfied}/{n_total}'
            constraint_color = 'normal' if all_constraints_ok else 'inverse'
            st.metric(
                '约束满足情况',
                constraint_status,
                delta=f'{n_satisfied}/{n_total} 工作面达标',
                delta_color=constraint_color
            )
        with res_col4:
            conv_status = '✅ 已收敛' if result.converged else '⏹️ 达到最大代数'
            st.metric(
                '进化收敛状态',
                conv_status,
                delta=f'{result.generations_run} 代 / {result.parameters.max_generations}',
                delta_color='off'
            )

        res_col5, res_col6, res_col7, res_col8 = st.columns(4)
        with res_col5:
            st.metric('总耗时', f'{result.total_time:.2f} 秒')
        with res_col6:
            total_evals = result.generations_run * result.parameters.population_size
            avg_eval_time = result.total_time / total_evals * 1000 if total_evals > 0 else 0
            st.metric('总评估次数', f'{total_evals:,} 次',
                      delta=f'{avg_eval_time:.1f} ms/次', delta_color='off')
        with res_col7:
            st.metric('最优适应度', f'{result.best_fitness:.1f} W',
                      delta=f'含惩罚 {result.best_fitness - result.best_power:.1f}',
                      delta_color='off')
        with res_col8:
            total_violation_w = result.parameters.penalty_coefficient * result.total_violation
            st.metric('约束惩罚值', f'{total_violation_w:.1f} W',
                      delta=f'违反量 {result.total_violation:.4f}',
                      delta_color='inverse' if result.total_violation > 0 else 'off')

        result_tabs = st.tabs([
            '⚙️ 最优决策方案',
            '📈 收敛曲线图',
            '🎯 决策变量雷达图',
            '🌬️ 工作面风量对比',
            '⚡ 能耗对比图',
            '📋 全网风量分配',
            '📥 结果导出'
        ])

        with result_tabs[0]:
            st.subheader('🎯 最优扇风机转速与调节风门开度方案')

            dv_fig = plot_ga_decision_variables_bar(result)
            st.pyplot(dv_fig, use_container_width=True)

            dv_col1, dv_col2 = st.columns(2)
            with dv_col1:
                if fan_branch_ids:
                    st.markdown('##### 🔧 扇风机转速方案')
                    fan_data = []
                    for fid in sorted(result.fan_speeds.keys()):
                        speed = result.fan_speeds[fid]
                        change_pct = (speed - 1.0) * 100
                        fan_data.append({
                            '分支编号': fid,
                            '转速系数': f'{speed:.4f}',
                            '相对额定转速': f'{change_pct:+.2f}%',
                            '转速状态': '升速 ⬆️' if speed > 1.01 else ('降速 ⬇️' if speed < 0.99 else '额定 ➡️'),
                        })
                    st.dataframe(pd.DataFrame(fan_data), use_container_width=True, hide_index=True)
                else:
                    st.info('网络中无扇风机')

            with dv_col2:
                if damper_branch_ids:
                    st.markdown('##### 🚧 调节风门开度方案')
                    damper_data = []
                    for did in sorted(result.damper_openings.keys()):
                        opening = result.damper_openings[did]
                        opening_pct = opening * 100
                        branch = network.get_branch(did)
                        base_r = 0
                        if branch:
                            from core.resistance import calculate_friction_resistance, calculate_local_resistance
                            base_r = (calculate_friction_resistance(
                                branch.friction_coeff, branch.length, branch.perimeter, branch.area
                            ) + calculate_local_resistance(branch.local_coeff, branch.area))
                        added_r = base_r * result.parameters.damper_max_resistance_multiplier * (1.0 - opening)
                        damper_data.append({
                            '分支编号': did,
                            '开度系数': f'{opening:.4f}',
                            '开度百分比': f'{opening_pct:.2f}%',
                            '附加阻力 (Ns²/m⁸)': f'{added_r:.6f}',
                            '开度状态': '全开' if opening >= 0.99 else ('全关' if opening <= 0.01 else '调节'),
                        })
                    st.dataframe(pd.DataFrame(damper_data), use_container_width=True, hide_index=True)
                else:
                    st.info('网络中无调节风门')

        with result_tabs[1]:
            st.subheader('📈 遗传算法收敛曲线')
            if result.history:
                conv_fig = plot_ga_convergence_curve(result.history)
                st.pyplot(conv_fig, use_container_width=True)

                with st.expander('📋 各代适应度详情', expanded=False):
                    hist_data = []
                    for h in result.history:
                        hist_data.append({
                            '代数': h.generation,
                            '最优适应度 (W)': f'{h.best_fitness:.2f}',
                            '平均适应度 (W)': f'{h.avg_fitness:.2f}',
                            '最差适应度 (W)': f'{h.worst_fitness:.2f}',
                        })
                    st.dataframe(pd.DataFrame(hist_data), use_container_width=True, hide_index=True)
            else:
                st.info('无进化历史数据')

        with result_tabs[2]:
            st.subheader('🎯 决策变量雷达图（归一化显示）')
            radar_fig = plot_ga_radar_chart(result)
            st.pyplot(radar_fig, use_container_width=True)
            st.caption('雷达图将所有决策变量归一化到0-1范围。红线表示基准方案（风机额定转速1.0x，风门全开1.0）。'
                      '蓝色区域越靠近外圈表示值越大（风机转速越高，风门开度越大）。')

        with result_tabs[3]:
            st.subheader('🌬️ 优化前后工作面风量对比')
            wf_fig = plot_ga_airflow_comparison(result, network, workface_branch_ids)
            st.pyplot(wf_fig, use_container_width=True)

            with st.expander('📋 工作面风量详细检查', expanded=True):
                wf_detail_data = []
                for wf_id in sorted(workface_branch_ids):
                    branch = network.get_branch(wf_id)
                    initial_q = abs(branch.airflow) if branch else 0.0
                    opt_q = result.workface_airflows.get(wf_id, 0.0)
                    threshold = result.parameters.min_airflow_threshold
                    satisfied = result.constraint_satisfied.get(wf_id, False)
                    deficit = max(0.0, threshold - opt_q)
                    change_pct = ((opt_q - initial_q) / initial_q * 100) if initial_q > 0 else 0
                    wf_detail_data.append({
                        '分支编号': wf_id,
                        '起→止节点': f'{branch.from_node}→{branch.to_node}' if branch else 'N/A',
                        '优化前风量 (m³/s)': f'{initial_q:.3f}',
                        '优化后风量 (m³/s)': f'{opt_q:.3f}',
                        '变化百分比': f'{change_pct:+.2f}%',
                        '最低阈值 (m³/s)': f'{threshold:.2f}',
                        '不足量 (m³/s)': f'{deficit:.3f}' if deficit > 0 else '0.000',
                        '是否满足约束': '✅ 满足' if satisfied else '❌ 不满足',
                    })
                st.dataframe(pd.DataFrame(wf_detail_data), use_container_width=True, hide_index=True)

        with result_tabs[4]:
            st.subheader('⚡ 优化前后系统能耗对比')
            power_fig = plot_ga_power_comparison(result)
            st.pyplot(power_fig, use_container_width=True)

            with st.expander('💰 能耗节省估算（按年运行8000小时）', expanded=False):
                saving_w = max(0.0, result.initial_power - result.best_power)
                saving_kwh_per_year = saving_w * 8000 / 1000
                saving_per_year_rmb = saving_kwh_per_year * 0.6
                st.markdown(f"""
                | 指标 | 数值 |
                |------|------|
                | **初始功率** | {result.initial_power:.2f} W = {result.initial_power/1000:.3f} kW |
                | **优化后功率** | {result.best_power:.2f} W = {result.best_power/1000:.3f} kW |
                | **节能功率** | {saving_w:.2f} W = {saving_w/1000:.3f} kW |
                | **节能率** | {result.energy_saving_percent:.2f}% |
                | **年节电量（8000h）** | {saving_kwh_per_year:.2f} kWh |
                | **年节省电费（0.6元/kWh）** | ¥ {saving_per_year_rmb:,.2f} |
                """)

        with result_tabs[5]:
            st.subheader('📋 最优方案全网风量分配')

            optimized_network = copy.deepcopy(network)
            optimized_network.update_solution(result.all_airflows, result.all_pressures)
            from core.resistance import calculate_all_branch_resistances, calculate_network_natural_pressures, update_branch_pressure_drops
            calculate_all_branch_resistances(optimized_network)
            nat_p = calculate_network_natural_pressures(optimized_network)
            update_branch_pressure_drops(optimized_network, nat_p)

            all_data = []
            for bid in sorted(optimized_network.branches.keys()):
                br = optimized_network.get_branch(bid)
                init_br = network.get_branch(bid)
                initial_q = abs(init_br.airflow) if init_br else 0.0
                opt_q = abs(br.airflow)
                change_pct = ((opt_q - initial_q) / initial_q * 100) if initial_q > 0 else 0
                is_workface = '✅ 工作面' if bid in workface_branch_ids else ''
                meets_constraint = ''
                if bid in workface_branch_ids:
                    meets_constraint = '✅ 满足' if result.constraint_satisfied.get(bid, False) else '❌ 不满足'
                all_data.append({
                    '分支编号': bid,
                    '起→止': f'{br.from_node}→{br.to_node}',
                    '优化前 (m³/s)': f'{initial_q:.3f}',
                    '优化后 (m³/s)': f'{opt_q:.3f}',
                    '变化': f'{change_pct:+.2f}%',
                    '阻力 (Ns²/m⁸)': f'{br.resistance:.6f}',
                    '风压降 (Pa)': f'{br.pressure_drop:.2f}',
                    '工作面标记': is_workface,
                    '约束状态': meets_constraint,
                })
            st.dataframe(pd.DataFrame(all_data), use_container_width=True, hide_index=True)

            col_pv1, col_pv2, col_pv3 = st.columns(3)
            with col_pv1:
                show_q_pv = st.checkbox('显示风量', value=True, key='ga_pv_q')
            with col_pv2:
                show_p_pv = st.checkbox('显示风压', value=True, key='ga_pv_p')
            with col_pv3:
                show_fan_pv = st.checkbox('显示扇风机', value=True, key='ga_pv_fan')

            with st.spinner('渲染最优方案拓扑图...'):
                from visualization.network_plot import plot_network
                topo_fig = plot_network(
                    optimized_network,
                    show_airflow=show_q_pv,
                    show_pressure=show_p_pv,
                    show_fan_icon=show_fan_pv,
                    figsize=(14, 10)
                )
                st.pyplot(topo_fig, use_container_width=True)

        with result_tabs[6]:
            st.subheader('📥 导出优化结果')

            json_str = export_ga_result_to_json(result)

            exp_col1, exp_col2 = st.columns([3, 1])
            with exp_col1:
                st.download_button(
                    label='⬇️ 下载JSON格式优化结果',
                    data=json_str,
                    file_name=f'ga_optimization_result_{int(time.time())}.json',
                    mime='application/json',
                    use_container_width=True,
                    type='primary'
                )

            with st.expander('📄 预览JSON结果内容', expanded=False):
                st.code(json_str, language='json')

            with st.expander('📋 优化参数配置摘要', expanded=True):
                summary_text = f"""
**遗传算法参数配置:**
- 种群大小: {result.parameters.population_size}
- 最大代数: {result.parameters.max_generations}
- 交叉概率 (SBX): {result.parameters.crossover_prob:.2f}, 分布指数: {result.parameters.sbx_distribution_index}
- 变异概率 (PM): {result.parameters.mutation_prob:.3f}, 分布指数: {result.parameters.pm_distribution_index}
- 精英保留: {result.parameters.elitism_count} 个, 锦标赛大小: {result.parameters.tournament_size}
- 惩罚系数: {result.parameters.penalty_coefficient:.1f}
- 最低通风量阈值: {result.parameters.min_airflow_threshold:.2f} m³/s

**决策变量范围:**
- 扇风机转速系数: {result.parameters.fan_speed_min:.2f} ~ {result.parameters.fan_speed_max:.2f} (额定1.0x)
- 风门开度: {result.parameters.damper_open_min:.2f} ~ {result.parameters.damper_open_max:.2f}
- 风门全关时最大阻力倍数: {result.parameters.damper_max_resistance_multiplier:.0f}x

**运行参数:**
- 收敛判定: 连续 {result.parameters.convergence_generations} 代改善 < {result.parameters.convergence_improvement*100:.2f}%
- Hardy-Cross求解: 容差 {result.parameters.tolerance:.2e}, 最大迭代 {result.parameters.max_iterations}

**实际运行结果:**
- 实际运行代数: {result.generations_run}
- 总评估次数: {result.generations_run * result.parameters.population_size:,}
- 总耗时: {result.total_time:.2f} 秒
- 是否提前收敛: {'是' if result.converged else '否'}
- 最终最优适应度: {result.best_fitness:.2f} W
"""
                st.text(summary_text)


def _build_default_presets(network: VentilationNetwork) -> List[Dict]:
    branch_ids = sorted(network.branches.keys())
    fan_ids = [b.id for b in network.get_fan_branches()]
    presets = []

    non_atm_branches = [bid for bid in branch_ids if not network.get_branch(bid).is_atmospheric]
    if len(non_atm_branches) >= 2:
        target_bid = non_atm_branches[1]
        presets.append({
            "name": "示例1: 巷道积水(分支{bid}阻力线性增大2倍)".format(bid=target_bid),
            "rule": ChangeRule(
                id=f"preset_water_{target_bid}",
                branch_id=target_bid,
                parameter_type=ParameterType.RESISTANCE,
                mode=ChangeMode.LINEAR,
                base_value=1.0,
                target_value=2.0,
                start_time=6.0,
                end_time=18.0,
            )
        })

    if fan_ids:
        fan_bid = fan_ids[0]
        presets.append({
            "name": "示例2: 风机故障(分支{bid} 12h时转速降为70%)".format(bid=fan_bid),
            "rule": ChangeRule(
                id=f"preset_fan_{fan_bid}",
                branch_id=fan_bid,
                parameter_type=ParameterType.FAN_SPEED,
                mode=ChangeMode.STEP,
                base_value=1.0,
                target_value=0.7,
                start_time=12.0,
            )
        })

    if len(non_atm_branches) >= 1:
        bid = non_atm_branches[0]
        presets.append({
            "name": "示例3: 昼夜通风波动(分支{bid}正弦±10%)".format(bid=bid),
            "rule": ChangeRule(
                id=f"preset_sine_{bid}",
                branch_id=bid,
                parameter_type=ParameterType.RESISTANCE,
                mode=ChangeMode.SINE,
                base_value=1.0,
                amplitude=0.1,
                period=24.0,
                phase=0.0,
            )
        })

    return presets


def scene_editor_section(network: VentilationNetwork):
    st.subheader("🎬 场景编辑器 - 定义时间变化规则")

    branch_ids = sorted(network.branches.keys())
    fan_branch_ids = [b.id for b in network.get_fan_branches()]

    col_preset, col_clear = st.columns([3, 1])
    with col_preset:
        presets = _build_default_presets(network)
        preset_options = ["选择预设场景快速加载..."] + [p["name"] for p in presets]
        preset_choice = st.selectbox("⚡ 预设场景", preset_options, index=0, key="ts_preset_select")
        if preset_choice != preset_options[0]:
            for p in presets:
                if p["name"] == preset_choice:
                    existing_ids = {r.id for r in st.session_state.ts_rules}
                    if p["rule"].id not in existing_ids:
                        st.session_state.ts_rules.append(p["rule"])
                        st.success(f"已加载预设规则: {p['name']}")
                    else:
                        st.info("该预设规则已存在")
                    break
    with col_clear:
        if st.button("🗑️ 清空所有规则", use_container_width=True):
            st.session_state.ts_rules = []
            st.session_state.ts_rule_counter = 0
            st.rerun()

    st.markdown("---")
    st.markdown("#### ➕ 添加新规则")

    col_ar1, col_ar2, col_ar3, col_ar4 = st.columns([2, 2, 2, 1])
    with col_ar1:
        new_branch = st.selectbox("作用分支", branch_ids, key="new_rule_branch",
                                  format_func=lambda x: f"分支 {x}" + (" (风机)" if x in fan_branch_ids else ""))
    with col_ar2:
        param_opts = [("阻力系数", ParameterType.RESISTANCE)]
        if new_branch in fan_branch_ids:
            param_opts.append(("风机转速", ParameterType.FAN_SPEED))
        param_display = [p[0] for p in param_opts]
        param_idx = st.selectbox("作用参数", range(len(param_display)),
                                 format_func=lambda i: param_display[i], key="new_rule_param_idx")
        new_param_type = param_opts[param_idx][1]
    with col_ar3:
        mode_opts = [
            ("阶跃变化 (某时刻突然改变)", ChangeMode.STEP),
            ("线性变化 (时间段内匀速增减)", ChangeMode.LINEAR),
            ("正弦波动 (周期性变化)", ChangeMode.SINE),
        ]
        mode_display = [m[0] for m in mode_opts]
        mode_idx = st.selectbox("变化模式", range(len(mode_display)),
                                format_func=lambda i: mode_display[i], key="new_rule_mode_idx")
        new_mode = mode_opts[mode_idx][1]
    with col_ar4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        add_clicked = st.button("➕ 添加规则", type="primary", use_container_width=True)

    if new_mode == ChangeMode.STEP:
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            base_val = st.number_input("变化前基准值 (倍率)", value=1.0, step=0.1, key="new_step_base", format="%.3f")
        with col_s2:
            target_val = st.number_input("变化后目标值 (倍率)", value=0.8 if new_param_type == ParameterType.FAN_SPEED else 1.5,
                                         step=0.1, key="new_step_target", format="%.3f")
        with col_s3:
            step_t = st.number_input("变化时刻 (小时, 0-24)", value=12.0, min_value=0.0, max_value=24.0, step=0.5, key="new_step_t")
    elif new_mode == ChangeMode.LINEAR:
        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        with col_l1:
            base_val = st.number_input("起始值 (倍率)", value=1.0, step=0.1, key="new_lin_base", format="%.3f")
        with col_l2:
            target_val = st.number_input("终值 (倍率)", value=2.0, step=0.1, key="new_lin_target", format="%.3f")
        with col_l3:
            start_t = st.number_input("开始时刻 (h)", value=6.0, min_value=0.0, max_value=24.0, step=0.5, key="new_lin_start")
        with col_l4:
            end_t = st.number_input("结束时刻 (h)", value=18.0, min_value=0.0, max_value=24.0, step=0.5, key="new_lin_end")
    else:
        col_sn1, col_sn2, col_sn3, col_sn4 = st.columns(4)
        with col_sn1:
            base_val = st.number_input("基准值 (倍率)", value=1.0, step=0.1, key="new_sine_base", format="%.3f")
        with col_sn2:
            amplitude = st.number_input("振幅 (±倍率)", value=0.1, min_value=0.0, step=0.05, key="new_sine_amp", format="%.3f")
        with col_sn3:
            period = st.number_input("周期 (小时)", value=24.0, min_value=0.5, step=0.5, key="new_sine_period")
        with col_sn4:
            phase = st.number_input("相位偏移 (小时)", value=0.0, min_value=0.0, step=0.5, key="new_sine_phase")

    if add_clicked:
        rule_id = f"rule_{st.session_state.ts_rule_counter}_{int(time.time() * 1000) % 1000}"
        st.session_state.ts_rule_counter += 1

        new_rule = ChangeRule(
            id=rule_id,
            branch_id=new_branch,
            parameter_type=new_param_type,
            mode=new_mode,
            base_value=base_val if new_mode != ChangeMode.SINE else base_val,
            target_value=target_val if new_mode != ChangeMode.SINE else base_val,
            start_time=step_t if new_mode == ChangeMode.STEP else (start_t if new_mode == ChangeMode.LINEAR else 0.0),
            end_time=step_t if new_mode == ChangeMode.STEP else (end_t if new_mode == ChangeMode.LINEAR else 24.0),
            period=period if new_mode == ChangeMode.SINE else 24.0,
            amplitude=amplitude if new_mode == ChangeMode.SINE else 0.0,
            phase=phase if new_mode == ChangeMode.SINE else 0.0,
        )
        if new_mode == ChangeMode.SINE:
            new_rule.target_value = amplitude

        st.session_state.ts_rules.append(new_rule)
        st.success(f"✅ 已添加规则: 分支{new_branch} - {new_mode.value}")
        st.rerun()

    st.markdown("---")
    rules = st.session_state.ts_rules
    if not rules:
        st.info("📋 暂无规则，请添加或加载预设场景规则")
    else:
        st.markdown(f"#### 📋 已定义规则 ({len(rules)} 条)")
        mode_cn = {ChangeMode.STEP: "阶跃", ChangeMode.LINEAR: "线性", ChangeMode.SINE: "正弦"}
        param_cn = {ParameterType.RESISTANCE: "阻力", ParameterType.FAN_SPEED: "风机转速"}

        for idx, rule in enumerate(rules):
            with st.expander(
                f"[{idx+1}] 分支{rule.branch_id} | {param_cn.get(rule.parameter_type, '?')} | "
                f"{mode_cn.get(rule.mode, '?')}模式",
                expanded=(idx == len(rules) - 1)
            ):
                col_info, col_preview, col_del = st.columns([2, 3, 1])
                with col_info:
                    st.markdown(f"**规则ID**: `{rule.id[:12]}...`")
                    st.markdown(f"**作用分支**: 分支 {rule.branch_id}")
                    st.markdown(f"**参数类型**: {param_cn.get(rule.parameter_type, '?')}")
                    st.markdown(f"**变化模式**: {mode_cn.get(rule.mode, '?')}")

                    if rule.mode == ChangeMode.STEP:
                        st.markdown(f"**基准值**: {rule.base_value:.3f} 倍")
                        st.markdown(f"**目标值**: {rule.target_value:.3f} 倍")
                        st.markdown(f"**阶跃时刻**: {rule.start_time:.1f} h")
                    elif rule.mode == ChangeMode.LINEAR:
                        st.markdown(f"**起始值**: {rule.base_value:.3f} 倍")
                        st.markdown(f"**终值**: {rule.target_value:.3f} 倍")
                        st.markdown(f"**时间区间**: {rule.start_time:.1f}h ~ {rule.end_time:.1f}h")
                    else:
                        st.markdown(f"**基准值**: {rule.base_value:.3f} 倍")
                        st.markdown(f"**振幅**: ±{rule.amplitude:.3f}")
                        st.markdown(f"**周期**: {rule.period:.1f} h")
                        st.markdown(f"**相位**: {rule.phase:.1f} h")

                with col_preview:
                    try:
                        preview_fig = plot_rule_preview(rule, total_hours=24.0, figsize=(8, 4))
                        st.pyplot(preview_fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"预览图生成失败: {e}")

                with col_del:
                    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                    if st.button(f"🗑️ 删除", key=f"del_rule_{idx}", use_container_width=True):
                        st.session_state.ts_rules.pop(idx)
                        st.rerun()


def time_series_simulation_tab():
    st.header("⏱️ 时间序列模拟")

    if st.session_state.network is None:
        st.warning('请先在"网络定义"模块中定义或加载通风网络')
        return

    network = st.session_state.network
    is_valid, errors = network.validate()
    if not is_valid:
        st.error('网络存在拓扑问题，无法进行模拟：')
        for error in errors:
            st.write(f'  - {error}')
        return

    branch_ids = sorted(network.branches.keys())
    node_ids = sorted(network.nodes.keys())

    ts_tab1, ts_tab2, ts_tab3, ts_tab4 = st.tabs([
        "🎬 场景编辑器", "⚙️ 模拟设置与运行", "📈 时间轴拓扑图", "📊 趋势对比图"
    ])

    with ts_tab1:
        scene_editor_section(network)

    with ts_tab2:
        st.subheader("⚙️ 模拟参数设置")

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            total_hours = st.number_input("模拟总时长 (小时)", value=24.0, min_value=1.0, max_value=720.0, step=1.0)
        with col_cfg2:
            time_step_min = st.number_input("时间步长 (分钟)", value=15.0, min_value=1.0, max_value=360.0, step=1.0)
        with col_cfg3:
            ts_tolerance = st.number_input(
                "收敛阈值", min_value=1e-6, max_value=0.1, value=0.001, format="%.6f", step=1e-4, key="ts_tol"
            )

        col_cfg4, col_cfg5 = st.columns(2)
        with col_cfg4:
            ts_max_iter = st.number_input("最大迭代次数", min_value=10, max_value=10000, value=500, step=50, key="ts_max_iter")
        with col_cfg5:
            use_warm = st.checkbox("使用上一步解作为初值 (热启动，推荐)", value=True)

        n_expected_steps = int(round(total_hours / (time_step_min / 60.0))) + 1
        st.info(f"预计将进行 **{n_expected_steps}** 个时间步求解")

        if not st.session_state.ts_rules:
            st.warning("⚠️ 尚未定义任何变化规则，将运行稳态基准模拟（所有时间步结果相同）")

        run_sim = st.button("🚀 开始时间序列模拟", type="primary", use_container_width=True)

        if run_sim:
            st.session_state.ts_result = None
            st.session_state.ts_perf_stats = None
            st.session_state.ts_selected_time = 0.0

            progress_bar = st.progress(0.0, text="准备模拟...")
            status_placeholder = st.empty()

            def progress_cb(cur, total, cur_t):
                pct = cur / total
                progress_bar.progress(pct, text=f"求解中: {cur}/{total} 步 (当前 t={cur_t:.2f}h)")

            try:
                with st.spinner("正在进行时间序列模拟..."):
                    result, perf = run_time_series_simulation(
                        network,
                        rules=st.session_state.ts_rules,
                        total_hours=total_hours,
                        time_step_minutes=time_step_min,
                        tolerance=ts_tolerance,
                        max_iterations=ts_max_iter,
                        use_warm_start=use_warm,
                        progress_callback=progress_cb,
                    )

                progress_bar.progress(1.0, text="✅ 模拟完成!")
                st.session_state.ts_result = result
                st.session_state.ts_perf_stats = perf

                if not st.session_state.ts_compare_branches:
                    non_atm = [bid for bid in branch_ids if not network.get_branch(bid).is_atmospheric]
                    st.session_state.ts_compare_branches = non_atm[:min(5, len(non_atm))]
                if not st.session_state.ts_compare_nodes:
                    st.session_state.ts_compare_nodes = node_ids[:min(5, len(node_ids))]

            except Exception as e:
                st.error(f"模拟失败: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                progress_bar.empty()

        if st.session_state.ts_result is not None and st.session_state.ts_perf_stats is not None:
            result: TimeSeriesResult = st.session_state.ts_result
            perf: Dict = st.session_state.ts_perf_stats

            st.success("✅ 模拟完成")

            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1:
                st.metric("总模拟耗时", f'{perf["total_simulation_time_s"]:.2f} s',
                          delta=f'{"< 10s ✅" if perf["total_simulation_time_s"] < 10 else "⚠️ 超目标"}')
            with col_p2:
                st.metric("总时间步数", f'{perf["n_steps"]} 步')
            with col_p3:
                st.metric("单步平均耗时", f'{perf["avg_step_time_s"] * 1000:.1f} ms')
            with col_p4:
                converged_count = sum(1 for info in result.solver_infos if info.get("converged", False))
                st.metric("收敛步数", f'{converged_count}/{perf["n_steps"]}',
                          delta=f'{converged_count * 100 / perf["n_steps"]:.1f}%')

            with st.expander("📋 求解统计详情", expanded=False):
                iter_counts = [info.get("iterations", 0) for info in result.solver_infos]
                st.write(f"- 迭代次数范围: {min(iter_counts)} ~ {max(iter_counts)} (平均 {sum(iter_counts)/len(iter_counts):.1f})")
                st.write(f"- 单步最快: {perf['min_step_time_s']*1000:.1f} ms, 最慢: {perf['max_step_time_s']*1000:.1f} ms")

            st.markdown("---")
            st.subheader("📥 导出CSV结果")
            csv_str = result.to_csv()
            st.download_button(
                label="⬇️ 下载完整CSV数据表",
                data=csv_str,
                file_name=f"time_series_result_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )
            with st.expander("CSV数据预览 (前10行)", expanded=False):
                lines = csv_str.split("\n")[:11]
                st.code("\n".join(lines))

    with ts_tab3:
        if st.session_state.ts_result is None:
            st.info("请先在「模拟设置与运行」标签页运行时间序列模拟")
        else:
            result: TimeSeriesResult = st.session_state.ts_result
            st.subheader("🕒 时间轴网络拓扑图")

            timestamps = result.timestamps
            if timestamps:
                col_slider, col_time_display = st.columns([4, 1])
                with col_slider:
                    sel_idx = st.slider(
                        "拖动时间滑块查看网络状态",
                        min_value=0,
                        max_value=len(timestamps) - 1,
                        value=0,
                        step=1,
                        format="%d",
                        key="ts_time_slider_idx",
                    )
                    st.markdown(
                        f"<div style='text-align:center; font-size:14px; font-weight:bold; color:#1f77b4;'>"
                        f"⏰ 当前时刻: t = {timestamps[sel_idx]:.2f} h "
                        f"({int(timestamps[sel_idx]):02d}:{int((timestamps[sel_idx]%1)*60):02d})</div>",
                        unsafe_allow_html=True
                    )
                with col_time_display:
                    st.metric("时间进度", f"{(sel_idx+1)}/{len(timestamps)}",
                              delta=f"{(sel_idx+1)*100/len(timestamps):.1f}%")

                col_play1, col_play2, col_play3 = st.columns(3)
                with col_play1:
                    if st.button("⏮️ 起始", use_container_width=True, key="btn_t_start"):
                        st.session_state.ts_time_slider_idx = 0
                        st.rerun()
                with col_play2:
                    step_adj = st.number_input("跳转步长 (步)", min_value=1, max_value=max(1, len(timestamps)//4),
                                               value=max(1, len(timestamps)//8), key="btn_adj_step")
                with col_play3:
                    if st.button("⏭️ 结束", use_container_width=True, key="btn_t_end"):
                        st.session_state.ts_time_slider_idx = len(timestamps) - 1
                        st.rerun()

                col_nav1, col_nav2, _, _, _, _ = st.columns(6)
                with col_nav1:
                    if st.button("◀ 后退", use_container_width=True, key="btn_t_prev"):
                        new_idx = max(0, sel_idx - step_adj)
                        st.session_state.ts_time_slider_idx = new_idx
                        st.rerun()
                with col_nav2:
                    if st.button("前进 ▶", use_container_width=True, key="btn_t_next"):
                        new_idx = min(len(timestamps) - 1, sel_idx + step_adj)
                        st.session_state.ts_time_slider_idx = new_idx
                        st.rerun()

                st.markdown("---")
                _, airflows, pressures, resistances = get_solution_at_time(result, timestamps[sel_idx])

                viz_network = copy.deepcopy(network)
                for bid in branch_ids:
                    br = viz_network.get_branch(bid)
                    if br:
                        br.airflow = airflows.get(bid, 0.0)
                        br.resistance = resistances.get(bid, br.resistance)
                for nid in node_ids:
                    nd = viz_network.get_node(nid)
                    if nd:
                        nd.pressure = pressures.get(nid, 0.0)

                from core.resistance import calculate_network_natural_pressures, update_branch_pressure_drops
                nat_p = calculate_network_natural_pressures(viz_network)
                update_branch_pressure_drops(viz_network, nat_p)

                col_tv1, col_tv2, col_tv3 = st.columns(3)
                with col_tv1:
                    show_q = st.checkbox("显示风量", value=True, key="ts_show_q")
                with col_tv2:
                    show_p = st.checkbox("显示风压", value=True, key="ts_show_p")
                with col_tv3:
                    show_fan = st.checkbox("显示扇风机图标", value=True, key="ts_show_fan")

                with st.spinner("渲染拓扑图..."):
                    fig = plot_network(
                        viz_network,
                        show_airflow=show_q,
                        show_pressure=show_p,
                        show_fan_icon=show_fan,
                        figsize=(14, 10)
                    )
                    st.pyplot(fig, use_container_width=True)

                with st.expander(f"📋 t={timestamps[sel_idx]:.2f}h 时刻数据快照", expanded=False):
                    snap_col1, snap_col2 = st.tabs(["分支风量/阻力", "节点风压"])
                    with snap_col1:
                        snap_data = []
                        for bid in branch_ids:
                            br = viz_network.get_branch(bid)
                            snap_data.append({
                                "分支编号": bid,
                                "风量 (m³/s)": f"{airflows.get(bid, 0):.4f}",
                                "阻力 (Ns²/m⁸)": f"{resistances.get(bid, 0):.6f}",
                                "风压降 (Pa)": f"{br.pressure_drop if br else 0:.2f}",
                            })
                        st.dataframe(pd.DataFrame(snap_data), use_container_width=True, hide_index=True)
                    with snap_col2:
                        p_data = []
                        for nid in node_ids:
                            p_data.append({
                                "节点编号": nid,
                                "风压 (Pa)": f"{pressures.get(nid, 0):.3f}",
                            })
                        st.dataframe(pd.DataFrame(p_data), use_container_width=True, hide_index=True)

    with ts_tab4:
        if st.session_state.ts_result is None:
            st.info("请先在「模拟设置与运行」标签页运行时间序列模拟")
        else:
            result: TimeSeriesResult = st.session_state.ts_result
            timestamps = result.timestamps

            st.subheader("📊 多分支/节点趋势对比图")

            trend_tab1, trend_tab2 = st.tabs(["🌬️ 分支风量趋势", "💨 节点风压趋势"])
            with trend_tab1:
                col_cb1, col_cb2 = st.columns([3, 1])
                with col_cb1:
                    non_atm_branches = [bid for bid in branch_ids if not network.get_branch(bid).is_atmospheric]
                    default_sel = st.session_state.ts_compare_branches if st.session_state.ts_compare_branches else non_atm_branches[:5]
                    selected_b = st.multiselect(
                        "选择要对比的分支 (可多选)",
                        options=branch_ids,
                        default=default_sel,
                        format_func=lambda x: f"分支 {x}" + (" (风机)" if network.get_branch(x).has_fan else ""),
                        key="ts_select_branches",
                    )
                    st.session_state.ts_compare_branches = selected_b
                with col_cb2:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                    sel_all_b = st.button("全选/全不选", key="btn_all_b", use_container_width=True)
                    if sel_all_b:
                        if selected_b:
                            st.session_state.ts_compare_branches = []
                        else:
                            st.session_state.ts_compare_branches = branch_ids
                        st.rerun()

                if selected_b:
                    sel_idx = st.session_state.get("ts_time_slider_idx", 0)
                    markers = [timestamps[sel_idx]] if 0 <= sel_idx < len(timestamps) else None
                    fig_a = plot_time_series_airflows(
                        timestamps, result.branch_airflows,
                        selected_branches=selected_b,
                        figsize=(14, 8),
                        time_markers=markers,
                    )
                    st.pyplot(fig_a, use_container_width=True)
                else:
                    st.info("请至少选择一个分支进行对比")

            with trend_tab2:
                col_cn1, col_cn2 = st.columns([3, 1])
                with col_cn1:
                    default_sel_n = st.session_state.ts_compare_nodes if st.session_state.ts_compare_nodes else node_ids[:min(5, len(node_ids))]
                    selected_n = st.multiselect(
                        "选择要对比的节点 (可多选)",
                        options=node_ids,
                        default=default_sel_n,
                        format_func=lambda x: f"节点 {x}",
                        key="ts_select_nodes",
                    )
                    st.session_state.ts_compare_nodes = selected_n
                with col_cn2:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                    sel_all_n = st.button("全选/全不选", key="btn_all_n", use_container_width=True)
                    if sel_all_n:
                        if selected_n:
                            st.session_state.ts_compare_nodes = []
                        else:
                            st.session_state.ts_compare_nodes = node_ids
                        st.rerun()

                if selected_n:
                    fig_p = plot_time_series_pressures(
                        timestamps, result.node_pressures,
                        selected_nodes=selected_n,
                        figsize=(14, 8),
                    )
                    st.pyplot(fig_p, use_container_width=True)
                else:
                    st.info("请至少选择一个节点进行对比")


def main():
    st.title('⛏️ 矿井通风网络阻力计算与风流分配模拟系统')
    st.markdown('---')

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        '📐 网络定义',
        '🔬 求解计算',
        '📊 可视化分析',
        '⏱️ 时间序列',
        '🔒 可靠性分析',
        '🧬 遗传优化',
        '📑 报告导出'
    ])

    with tab1:
        network_definition_tab()

    with tab2:
        solver_tab()

    with tab3:
        visualization_tab()

    with tab4:
        time_series_simulation_tab()

    with tab5:
        reliability_analysis_tab()

    with tab6:
        genetic_optimization_tab()

    with tab7:
        report_tab()

    st.markdown('---')
    st.caption('矿井通风网络模拟系统 v4.0 | 含时间序列模拟、可靠性分析与遗传算法优化模块')


if __name__ == '__main__':
    main()
