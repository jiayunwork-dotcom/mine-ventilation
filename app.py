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
            '模拟次数',
            min_value=100,
            max_value=10000,
            value=params['n_simulations'],
            step=100,
            key='rel_n_sim'
        )
        min_airflow_threshold = st.number_input(
            '最低通风量阈值 (m³/s)',
            min_value=0.5,
            max_value=20.0,
            value=params['min_airflow_threshold'],
            step=0.5,
            key='rel_min_q'
        )
        random_seed = st.number_input(
            '随机数种子',
            min_value=0,
            max_value=99999,
            value=params['random_seed'],
            step=1,
            key='rel_seed'
        )

    with col_p2:
        branch_failure_prob = st.slider(
            '分支故障概率',
            min_value=0.01,
            max_value=0.5,
            value=params['branch_failure_prob'],
            step=0.01,
            format='%.2f',
            key='rel_branch_prob'
        )
        fan_failure_prob = st.slider(
            '风机故障概率',
            min_value=0.01,
            max_value=0.3,
            value=params['fan_failure_prob'],
            step=0.01,
            format='%.2f',
            key='rel_fan_prob'
        )
        resistance_multiplier = st.number_input(
            '故障阻力倍增系数',
            min_value=2.0,
            max_value=50.0,
            value=params['resistance_multiplier'],
            step=1.0,
            key='rel_res_mult'
        )

    with col_p3:
        use_parallel = st.checkbox(
            '使用多进程并行计算',
            value=params['use_parallel'],
            key='rel_parallel'
        )
        generate_heatmap = st.checkbox(
            '生成可靠度热力图',
            value=params['generate_heatmap'],
            key='rel_heatmap'
        )
        identify_critical = st.checkbox(
            '识别关键分支',
            value=params['identify_critical'],
            key='rel_critical'
        )

    st.subheader('🏭 工作面分支标记')
    st.info('请选择需要检查最低通风量的工作面分支')

    workface_branch_ids = st.multiselect(
        '选择工作面分支',
        options=non_atm_branch_ids,
        default=params['workface_branch_ids'],
        format_func=lambda x: f'分支 {x}' + (' (风机)' if x in fan_branch_ids else ''),
        key='rel_workfaces'
    )

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

        progress_bar = st.progress(0.0, text='准备模拟...')
        status_placeholder = st.empty()

        start_time = time.time()

        def progress_cb(cur, total):
            pct = cur / total
            progress_bar.progress(pct, text=f'蒙特卡洛模拟: {cur}/{total} 次')

        try:
            with st.spinner('正在进行蒙特卡洛模拟...'):
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

            if generate_heatmap:
                def heatmap_progress_cb(cur, total):
                    pct = cur / total
                    progress_bar.progress(pct, text=f'生成热力图: {cur}/{total} 点')

                with st.spinner('正在生成可靠度热力图...'):
                    heatmap_data = generate_reliability_heatmap(
                        network=network,
                        workface_branch_ids=workface_branch_ids,
                        min_airflow_threshold=min_airflow_threshold,
                        resistance_multiplier=resistance_multiplier,
                        random_seed=random_seed,
                        n_simulations_per_point=300,
                        use_parallel=use_parallel,
                        progress_callback=heatmap_progress_cb
                    )
                result.heatmap_data = heatmap_data
                st.session_state.reliability_heatmap = heatmap_data

            if identify_critical:
                def critical_progress_cb(cur, total):
                    pct = cur / total
                    progress_bar.progress(pct, text=f'识别关键分支: {cur}/{total} 分支')

                with st.spinner('正在识别关键分支...'):
                    critical_branches = identify_critical_branches(
                        network=network,
                        workface_branch_ids=workface_branch_ids,
                        min_airflow_threshold=min_airflow_threshold,
                        base_branch_failure_prob=branch_failure_prob,
                        fan_failure_prob=fan_failure_prob,
                        resistance_multiplier=resistance_multiplier,
                        random_seed=random_seed,
                        n_simulations_per_branch=500,
                        top_k=3,
                        use_parallel=use_parallel,
                        progress_callback=critical_progress_cb
                    )
                result.critical_branches = critical_branches
                st.session_state.reliability_critical = critical_branches

            total_time = time.time() - start_time
            progress_bar.progress(1.0, text=f'✅ 分析完成! 总耗时: {total_time:.2f} 秒')
            status_placeholder.success(f'分析完成！总耗时: {total_time:.2f} 秒')

        except Exception as e:
            st.error(f'分析失败: {str(e)}')
            import traceback
            st.code(traceback.format_exc())
            progress_bar.empty()
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
            '📋 详细数据'
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        '📐 网络定义',
        '🔬 求解计算',
        '📊 可视化分析',
        '⏱️ 时间序列',
        '� 可靠性分析',
        '�📑 报告导出'
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
        report_tab()

    st.markdown('---')
    st.caption('矿井通风网络模拟系统 v3.0 | 含时间序列模拟与可靠性分析模块')


if __name__ == '__main__':
    main()
