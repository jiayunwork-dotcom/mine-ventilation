from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import io
import os
from datetime import datetime
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

from core.network import VentilationNetwork
from core.fan_operation import calculate_all_fan_operating_points, calculate_total_power_consumption
from core.optimization import generate_optimization_suggestions
from visualization.network_plot import plot_network
from visualization.fan_plot import plot_fan_operating_point


def add_chinese_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    font_paths = [
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/msyh.ttc',
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont('ChineseFont', path))
                return 'ChineseFont'
            except:
                continue
    
    return 'Helvetica'


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 10)
    page_num = canvas.getPageNumber()
    canvas.drawRightString(200 * mm, 20 * mm, f'第 {page_num} 页')
    canvas.restoreState()


def create_styles(font_name: str):
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='ChineseTitle',
        parent=styles['Title'],
        fontName=font_name,
        fontSize=20,
        leading=24,
        alignment=1,
        spaceAfter=20
    ))

    styles.add(ParagraphStyle(
        name='ChineseHeading1',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1a5276'),
        spaceBefore=15,
        spaceAfter=10
    ))

    styles.add(ParagraphStyle(
        name='ChineseHeading2',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2874a6'),
        spaceBefore=10,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name='ChineseBody',
        parent=styles['BodyText'],
        fontName=font_name,
        fontSize=10,
        leading=14,
        alignment=0
    ))

    styles.add(ParagraphStyle(
        name='ChineseBold',
        parent=styles['BodyText'],
        fontName=font_name,
        fontSize=10,
        leading=14,
        alignment=0,
        textColor=colors.HexColor('#2c3e50')
    ))

    styles.add(ParagraphStyle(
        name='ChineseWarning',
        parent=styles['BodyText'],
        fontName=font_name,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#c0392b')
    ))

    styles.add(ParagraphStyle(
        name='ChineseSuccess',
        parent=styles['BodyText'],
        fontName=font_name,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#27ae60')
    ))

    return styles


def create_table_style():
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
                            [colors.HexColor('#f8f9fa'), colors.HexColor('#e9ecef')]),
    ])


def fig_to_image(fig, width: int = 500) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image(buf, width=width, kind='proportional')
    return img


def generate_network_parameters_table(network: VentilationNetwork, styles) -> Table:
    data = [['节点编号', '标高 (m)', '温度 (°C)', '风压 (Pa)']]
    for node_id in sorted(network.nodes.keys()):
        node = network.get_node(node_id)
        data.append([
            str(node_id),
            f'{node.elevation:.2f}',
            f'{node.temperature:.1f}',
            f'{node.pressure:.2f}'
        ])
    
    return Table(data, colWidths=[3*cm, 3*cm, 3*cm, 3*cm])


def generate_branches_table(network: VentilationNetwork, styles) -> Table:
    data = [['分支编号', '起节点', '止节点', '长度 (m)', '断面积 (m²)', '周长 (m)', '阻力系数', '局部阻力']]
    for branch_id in sorted(network.branches.keys()):
        branch = network.get_branch(branch_id)
        data.append([
            str(branch_id),
            str(branch.from_node),
            str(branch.to_node),
            f'{branch.length:.2f}',
            f'{branch.area:.2f}',
            f'{branch.perimeter:.2f}',
            f'{branch.friction_coeff:.4f}',
            f'{branch.local_coeff:.4f}'
        ])
    
    return Table(data, colWidths=[2*cm, 1.5*cm, 1.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm])


def generate_results_table(network: VentilationNetwork, styles) -> Table:
    data = [['分支编号', '风量 (m³/s)', '风压降 (Pa)', '阻力 (Ns²/m⁸)', '风速 (m/s)', '扇风机']]
    for branch_id in sorted(network.branches.keys()):
        branch = network.get_branch(branch_id)
        velocity = network.get_air_velocity(branch_id)
        has_fan = '是' if branch.has_fan else '否'
        data.append([
            str(branch_id),
            f'{branch.airflow:.3f}',
            f'{branch.pressure_drop:.2f}',
            f'{branch.resistance:.6f}',
            f'{velocity:.2f}',
            has_fan
        ])
    
    return Table(data, colWidths=[2*cm, 2.5*cm, 2.5*cm, 3*cm, 2*cm, 1.5*cm])


def generate_pdf_report(
    network: VentilationNetwork,
    solver_info: Optional[Dict],
    output_path: str,
    include_network_plot: bool = True,
    include_fan_plots: bool = True,
    title: str = '矿井通风系统分析报告'
) -> str:
    font_name = add_chinese_font()
    styles = create_styles(font_name)
    table_style = create_table_style()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    story = []

    story.append(Paragraph(title, styles['ChineseTitle']))
    story.append(Paragraph(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', styles['ChineseBody']))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph('一、网络基本信息', styles['ChineseHeading1']))
    story.append(Paragraph(f'网络节点数: {len(network.nodes)}', styles['ChineseBody']))
    story.append(Paragraph(f'网络分支数: {len(network.branches)}', styles['ChineseBody']))
    story.append(Paragraph(f'独立回路数: {network.get_independent_loops_count()}', styles['ChineseBody']))
    
    is_valid, errors = network.validate()
    if is_valid:
        story.append(Paragraph('网络拓扑: 连通', styles['ChineseSuccess']))
    else:
        story.append(Paragraph('网络拓扑: 不连通', styles['ChineseWarning']))
        for error in errors:
            story.append(Paragraph(f'  - {error}', styles['ChineseWarning']))
    
    fan_branches = network.get_fan_branches()
    story.append(Paragraph(f'扇风机数量: {len(fan_branches)}', styles['ChineseBody']))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('二、节点参数表', styles['ChineseHeading1']))
    node_table = generate_network_parameters_table(network, styles)
    node_table.setStyle(table_style)
    story.append(node_table)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph('三、分支参数表', styles['ChineseHeading1']))
    branch_table = generate_branches_table(network, styles)
    branch_table.setStyle(table_style)
    story.append(branch_table)
    story.append(PageBreak())

    story.append(Paragraph('四、求解结果表', styles['ChineseHeading1']))
    if solver_info:
        method = solver_info.get('method', '未知方法')
        iterations = solver_info.get('iterations', 0)
        converged = solver_info.get('converged', False)
        final_residual = solver_info.get('final_residual', 0)
        
        story.append(Paragraph(f'求解方法: {method}', styles['ChineseBody']))
        story.append(Paragraph(f'迭代次数: {iterations}', styles['ChineseBody']))
        status = '收敛' if converged else '未收敛'
        story.append(Paragraph(f'收敛状态: {status}', styles['ChineseSuccess'] if converged else styles['ChineseWarning']))
        story.append(Paragraph(f'最终残差: {final_residual:.6e}', styles['ChineseBody']))
        story.append(Spacer(1, 0.3*cm))

    result_table = generate_results_table(network, styles)
    result_table.setStyle(table_style)
    story.append(result_table)
    story.append(Spacer(1, 0.5*cm))

    power_info = calculate_total_power_consumption(network)
    story.append(Paragraph('五、系统能耗分析', styles['ChineseHeading1']))
    story.append(Paragraph(f'总轴功率: {power_info["total_shaft_power"]:.2f} W', styles['ChineseBody']))
    story.append(Paragraph(f'总有效功率: {power_info["total_air_power"]:.2f} W', styles['ChineseBody']))
    story.append(Paragraph(f'系统总效率: {power_info["total_efficiency"]*100:.1f}%', styles['ChineseBody']))
    story.append(Paragraph(f'总风量: {power_info["total_airflow"]:.2f} m³/s', styles['ChineseBody']))
    story.append(Paragraph(f'单位风量能耗: {power_info["specific_power"]:.3f} W/(m³/s)', styles['ChineseBody']))
    story.append(Spacer(1, 0.5*cm))

    if include_network_plot:
        story.append(PageBreak())
        story.append(Paragraph('六、网络拓扑图', styles['ChineseHeading1']))
        try:
            from visualization.network_plot import plot_network
            fig = plot_network(network, figsize=(10, 8))
            img = fig_to_image(fig, width=17*cm)
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f'绘图失败: {str(e)}', styles['ChineseWarning']))
        story.append(Spacer(1, 0.5*cm))

    if include_fan_plots and fan_branches:
        story.append(PageBreak())
        story.append(Paragraph('七、扇风机工作点图', styles['ChineseHeading1']))
        for fan_branch in fan_branches:
            try:
                from visualization.fan_plot import plot_fan_operating_point
                fig = plot_fan_operating_point(network, fan_branch.id, figsize=(10, 8))
                img = fig_to_image(fig, width=17*cm)
                story.append(img)
                story.append(Spacer(1, 0.3*cm))
            except Exception as e:
                story.append(Paragraph(f'扇风机 {fan_branch.id} 绘图失败: {str(e)}', styles['ChineseWarning']))

    story.append(PageBreak())
    story.append(Paragraph('八、优化建议', styles['ChineseHeading1']))
    
    suggestions = generate_optimization_suggestions(network)
    
    story.append(Paragraph(f'检测到问题总数: {suggestions["total_issues"]} 个', styles['ChineseBody']))
    story.append(Paragraph(f'  严重问题: {suggestions["severity_counts"].get("high", 0)} 个', styles['ChineseWarning']))
    story.append(Paragraph(f'  中等问题: {suggestions["severity_counts"].get("medium", 0)} 个', styles['ChineseBody']))
    story.append(Paragraph(f'  轻微问题: {suggestions["severity_counts"].get("low", 0)} 个', styles['ChineseBody']))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('详细建议（按优先级排序):', styles['ChineseHeading2']))
    for i, issue in enumerate(suggestions['sorted_suggestions'][:20], 1):
        severity = issue.get('severity', 'low')
        suggestion_text = issue.get('suggestion', '无具体建议')
        style_key = 'ChineseWarning' if severity == 'high' else 'ChineseBody'
        story.append(Paragraph(f'{i}. [{severity.upper()}] {suggestion_text}', styles[style_key]))
        story.append(Spacer(1, 0.1*cm))

    if len(suggestions['sorted_suggestions']) > 20:
        story.append(Paragraph(f'... 还有 {len(suggestions["sorted_suggestions"]) - 20} 条建议未显示', styles['ChineseBody']))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

    return output_path
