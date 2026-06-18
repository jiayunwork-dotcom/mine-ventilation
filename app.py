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
    plot_airflow_distribution
)
from visualization.fan_plot import (
    plot_fan_operating_point,
    plot_multiple_fan_curves,
    plot_system_curve_comparison
)
from report.pdf_generator import generate_pdf_report

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
            damper_resistance=float(row.get('damper_resistance', 0.0))
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
                    from_n = st.number_input(f'起节点', min_value=1, value=1, key=f'from_{i}')
                with col_b3:
                    to_n = st.number_input(f'止节点', min_value=1, value=2, key=f'to_{i}')
                with col_b4:
                    length = st.number_input(f'长度 (m)', value=500.0, key=f'len_{i}')
                with col_b5:
                    area = st.number_input(f'断面积 (m²)', value=10.0, key=f'area_{i}')
                with col_b6:
                    perimeter = st.number_input(f'周长 (m)', value=13.0, key=f'peri_{i}')

                col_b7, col_b8, col_b9, col_b10 = st.columns([1, 1, 1, 1])
                with col_b7:
                    fric = st.number_input(f'摩擦阻力系数', value=0.012, key=f'fric_{i}', format='%.4f')
                with col_b8:
                    local = st.number_input(f'局部阻力系数', value=0.5, key=f'local_{i}')
                with col_b9:
                    has_fan = st.checkbox(f'有扇风机', key=f'fan_{i}')
                with col_b10:
                    has_damper = st.checkbox(f'有调节风门', key=f'damper_{i}')

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
                    '调节风门': '是' if branch.has_damper else '否'
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


def main():
    st.title('⛏️ 矿井通风网络阻力计算与风流分配模拟系统')
    st.markdown('---')

    tab1, tab2, tab3, tab4 = st.tabs([
        '📐 网络定义',
        '🔬 求解计算',
        '📊 可视化分析',
        '📑 报告导出'
    ])

    with tab1:
        network_definition_tab()

    with tab2:
        solver_tab()

    with tab3:
        visualization_tab()

    with tab4:
        report_tab()

    st.markdown('---')
    st.caption('矿井通风网络模拟系统 v1.0 | 基于Hardy-Cross和Newton-Raphson方法')


if __name__ == '__main__':
    main()
