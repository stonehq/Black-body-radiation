"""
黑体辐射仿真程序 (Streamlit版)
基于普朗克黑体辐射定律
作者：从MATLAB转换而来
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.constants as const
from scipy.optimize import fsolve
from scipy.integrate import trapezoid  # 添加这一行

# ==================== 物理常量 ====================
h = const.h  # 普朗克常数: 6.626e-34 J·s
c = const.c  # 光速: 2.998e8 m/s
k = const.k  # 玻尔兹曼常数: 1.38e-23 J/K
sigma_sb = const.sigma  # 斯特藩-玻尔兹曼常数: 5.670374e-8 W/(m²·K⁴)

# 辐射常数
CONST_C1 = 2 * np.pi * h * c ** 2
CONST_C2 = h * c / k
CONST_RJ = 2 * np.pi * c * k

# 温度范围
MIN_T = 200
MAX_T = 10000
DEFAULT_T = 5748  # 太阳表面温度

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="黑体辐射仿真",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS ====================
st.markdown("""
<style>
    /* 主内容区域 - 黑色背景 */
    .main {
        background-color: #000000;
    }
    .stApp {
        background-color: #000000;
    }

    /* 侧边栏 - 浅色背景，深色文字 */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    /* 侧边栏标题 */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1f1f1f !important;
    }

    /* 侧边栏标签 */
    section[data-testid="stSidebar"] label {
        color: #262730 !important;
        font-size: 32px !important;
    }

    /* 侧边栏滑块标签 */
    section[data-testid="stSidebar"] .stSlider label {
        color: #1f1f1f !important;
        font-size: 32px !important;
        font-weight: bold !important;
    }

    /* 侧边栏复选框 */
    section[data-testid="stSidebar"] .stCheckbox label {
        color: #262730 !important;
        font-size: 32px !important;
    }

    /* 侧边栏单选按钮 */
    section[data-testid="stSidebar"] .stRadio label {
        color: #262730 !important;
    }

    /* 侧边栏下拉框 */
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #262730 !important;
    }

    /* 侧边栏数字输入框 */
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #262730 !important;
    }

    /* 侧边栏信息提示框 */
    section[data-testid="stSidebar"] .stAlert {
        background-color: #e8f4f8 !important;
        color: #0e1117 !important;
    }

    /* 侧边栏按钮 */
    section[data-testid="stSidebar"] .stButton button {
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
    }

    /* 侧边栏分割线 */
    section[data-testid="stSidebar"] hr {
        border-color: #d0d0d0 !important;
    }

    /* 主内容区域文字 - 白色 */
    .main h1, .main h2, .main h3, .main p, .main label {
        color: #ffffff !important;
    }

    /* 主内容区域的指标 */
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        color: #FFD700;
    }
    div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }

    /* 温度显示框 */
        .temp-display {
        text-align: left;
        font-size: 32px;
        font-weight: bold;
        color: #FFD700;
        padding: 15px 25px;
        background-color: #1a1a1a;
        border-radius: 10px;
        border: 2px solid #FFD700;
        display: inline-block;
        margin-bottom: 20px;
    }

    /* 信息框 */
    .info-box {
        background-color: #1a1a1a;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 辅助函数 ====================
def calculate_optimal_x_range(temperature, threshold_ratio=0.05):
    """
    计算最优X轴范围
    找到辐射强度降到峰值 threshold_ratio 倍时的波长范围

    参数:
        temperature: 温度(K)
        threshold_ratio: 阈值比例（默认0.05即5%）

    返回:
        (x_min, x_max): 波长范围(μm)
    """
    # 获取峰值波长
    peak_lambda = wien_displacement_law(temperature)  # μm
    peak_lambda_m = peak_lambda * 1e-6

    # 计算峰值辐射强度
    peak_intensity = planck_law(peak_lambda_m, temperature)
    threshold_intensity = peak_intensity * threshold_ratio

    # 在峰值左侧搜索（短波长）
    lambda_left = np.linspace(0.01, peak_lambda, 500)
    B_left = planck_law(lambda_left * 1e-6, temperature)

    # 找到第一个超过阈值的点
    idx_left = np.where(B_left >= threshold_intensity)[0]
    if len(idx_left) > 0:
        x_min = lambda_left[idx_left[0]]
    else:
        x_min = 0.01

    # 在峰值右侧搜索（长波长）
    lambda_right = np.linspace(peak_lambda, peak_lambda * 20, 500)
    B_right = planck_law(lambda_right * 1e-6, temperature)

    # 找到最后一个超过阈值的点
    idx_right = np.where(B_right >= threshold_intensity)[0]
    if len(idx_right) > 0:
        x_max = lambda_right[idx_right[-1]]
    else:
        x_max = peak_lambda * 10

    # 添加一些余量（左右各扩展10%）
    margin = (x_max - x_min) * 0.1
    x_min = max(0.01, x_min - margin)
    x_max = x_max + margin

    return x_min, x_max
def kelvin_to_rgb(temp):
    """
    将开尔文温度转换为RGB颜色
    基于黑体辐射近似
    """
    temp = temp / 100.0

    if temp <= 66:
        r = 255
        g = 99.4708025861 * np.log(temp) - 161.1195681661
        if temp <= 19:
            b = 0
        else:
            b = 138.5177312231 * np.log(temp - 10) - 305.0447927307
    else:
        r = 329.698727446 * ((temp - 60) ** -0.1332047592)
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
        b = 255

    # 归一化到0-1范围
    r = np.clip(r, 0, 255) / 255.0
    g = np.clip(g, 0, 255) / 255.0
    b = np.clip(b, 0, 255) / 255.0

    return f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})'


def wavelength_to_rgb(wavelength_nm):
    """
    将波长(nm)转换为可见光颜色RGB
    """
    w = wavelength_nm

    if 380 <= w < 440:
        r = -(w - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= w < 490:
        r = 0.0
        g = (w - 440) / (490 - 440)
        b = 1.0
    elif 490 <= w < 510:
        r = 0.0
        g = 1.0
        b = -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        r = (w - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= w < 645:
        r = 1.0
        g = -(w - 645) / (645 - 580)
        b = 0.0
    elif 645 <= w <= 780:
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        r = 0.0
        g = 0.0
        b = 0.0

    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)

    return f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})'


def planck_law(wavelength_m, temperature):
    """
    普朗克黑体辐射定律
    返回单位: MW/m²/μm
    """
    with np.errstate(over='ignore', invalid='ignore'):
        exponent = CONST_C2 / (wavelength_m * temperature)
        exponent = np.clip(exponent, 0, 700)  # 防止溢出

        B = (CONST_C1 / (wavelength_m ** 5)) / (np.exp(exponent) - 1)
        # 转换为 MW/m²/μm
        B = B * 1e-12

    return B


def rayleigh_jeans_law(wavelength_m, temperature):
    """
    瑞利-金斯公式（经典近似，长波长适用）
    """
    B = (CONST_RJ * temperature) / (wavelength_m ** 4)
    B = B * 1e-12  # 转换为 MW/m²/μm
    B = np.where(B > 50000, np.nan, B)  # 防止无穷大
    return B


def wien_law(wavelength_m, temperature):
    """
    维恩公式（短波长近似）
    """
    with np.errstate(over='ignore'):
        exponent = CONST_C2 / (wavelength_m * temperature)
        exponent = np.clip(exponent, 0, 700)

        B = (CONST_C1 / (wavelength_m ** 5)) * np.exp(-exponent)
        B = B * 1e-12

    return B


def wien_displacement_law(temperature):
    """
    维恩位移定律: λ_max * T = 2.898e-3 m·K
    返回峰值波长(μm)
    """
    lambda_max_m = 2.898e-3 / temperature
    return lambda_max_m * 1e6  # 转换为μm


def stefan_boltzmann_law(temperature):
    """
    斯特藩-玻尔兹曼定律: I = σT⁴
    返回总辐射强度 (W/m²)
    """
    return sigma_sb * temperature ** 4


def format_scientific(value):
    """
    格式化科学计数法显示
    """
    if value == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(value))))
    base = value / (10 ** exponent)

    return f"{base:.2f} × 10^{exponent}"


# ==================== 绘图函数 ====================

def create_spectrum_background():
    """
    创建可见光光谱背景色带
    """
    wavelengths = np.linspace(380, 780, 500)
    colors = [wavelength_to_rgb(w) for w in wavelengths]

    # 创建渐变色带数据
    spectrum_trace = []
    for i in range(len(wavelengths) - 1):
        spectrum_trace.append(
            go.Scatter(
                x=[wavelengths[i] / 1000, wavelengths[i + 1] / 1000],
                y=[1e10, 1e10],
                mode='lines',
                line=dict(color=colors[i], width=20),
                showlegend=False,
                hoverinfo='skip'
            )
        )

    return spectrum_trace


def create_main_plot(temperature, show_rj, show_wien, show_labels,
                     show_values, show_intensity, x_min, x_max, y_max):
    """
    创建主要的辐射曲线图
    """
    # 波长范围 (μm) - 使用新的x_min和x_max
    lambda_um = np.linspace(x_min, x_max, 2000)
    lambda_m = lambda_um * 1e-6

    # 计算普朗克曲线
    B_planck = planck_law(lambda_m, temperature)

    # 创建图形
    fig = go.Figure()

    # 添加可见光彩色光谱带
    vis_min = 0.38
    vis_max = 0.78

    # 创建彩色渐变条带（从紫到红）
    num_bands = 100
    wavelengths_vis = np.linspace(vis_min, vis_max, num_bands)

    for i in range(len(wavelengths_vis) - 1):
        # 将波长（μm）转换为纳米用于颜色映射
        wl_nm = wavelengths_vis[i] * 1000
        color = wavelength_to_rgb(wl_nm)

        fig.add_shape(
            type="rect",
            x0=wavelengths_vis[i],
            x1=wavelengths_vis[i + 1],
            y0=0,
            y1=y_max,  # 填充到图表顶部
            fillcolor=color,
            opacity=0.3,  # 半透明，不遮挡曲线
            layer="below",
            line_width=0
        )

    # 如果显示强度，添加填充区域
    if show_intensity:
        fig.add_trace(go.Scatter(
            x=lambda_um,
            y=B_planck,
            fill='tozeroy',
            fillcolor='rgba(180, 180, 180, 0.3)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='强度积分区域'
        ))

    # 瑞利-金斯公式
    if show_rj:
        B_rj = rayleigh_jeans_law(lambda_m, temperature)
        fig.add_trace(go.Scatter(
            x=lambda_um,
            y=B_rj,
            mode='lines',
            name='瑞利-金斯公式',
            line=dict(color='rgb(204, 102, 255)', width=2, dash='dash'),
            hovertemplate='λ: %{x:.3f} μm<br>B: %{y:.2f}<extra></extra>'
        ))

    # 维恩公式
    if show_wien:
        B_wien = wien_law(lambda_m, temperature)
        fig.add_trace(go.Scatter(
            x=lambda_um,
            y=B_wien,
            mode='lines',
            name='维恩公式',
            line=dict(color='rgb(102, 204, 255)', width=2, dash='dash'),
            hovertemplate='λ: %{x:.3f} μm<br>B: %{y:.2f}<extra></extra>'
        ))

    # 普朗克曲线（主曲线）
    fig.add_trace(go.Scatter(
        x=lambda_um,
        y=B_planck,
        mode='lines',
        name='普朗克公式',
        line=dict(color='rgb(255, 128, 0)', width=4),
        hovertemplate='波长: %{x:.3f} μm<br>辐射强度: %{y:.2f} MW/m²/μm<extra></extra>'
    ))

    # 标记峰值
    peak_lambda = wien_displacement_law(temperature)
    peak_B = planck_law(peak_lambda * 1e-6, temperature)

    fig.add_trace(go.Scatter(
        x=[peak_lambda],
        y=[peak_B],
        mode='markers',
        name='峰值',
        marker=dict(size=12, color='white', symbol='circle'),
        hovertemplate=f'峰值波长: {peak_lambda:.3f} μm<br>峰值强度: {peak_B:.2f}<extra></extra>'
    ))

    if show_values:
        # 垂直虚线（从峰值到X轴）
        fig.add_shape(
            type="line",
            x0=peak_lambda, x1=peak_lambda,
            y0=0, y1=peak_B,
            line=dict(color="yellow", width=2, dash="dash")
        )
        # 水平虚线（从峰值到Y轴）
        fig.add_shape(
            type="line",
            x0=x_min, x1=peak_lambda,
            y0=peak_B, y1=peak_B,
            line=dict(color="yellow", width=2, dash="dash")
        )

        # X轴下方显示波长值（黄色大字）
        fig.add_annotation(
            x=peak_lambda,
            y=0,
            text=f"{peak_lambda:.3f}",
            showarrow=False,
            font=dict(size=24, color="yellow", family="Arial Black"),
            bgcolor="rgba(0,0,0,0.8)",
            borderpad=6,
            yshift=-30,  # 向下偏移
            xanchor='center',
            yanchor='top'
        )

        # Y轴左侧显示能量密度值（黄色大字）
        fig.add_annotation(
            x=x_min,
            y=peak_B,
            text=f"{peak_B:.2f}",
            showarrow=False,
            font=dict(size=24, color="yellow", family="Arial Black"),
            bgcolor="rgba(0,0,0,0.8)",
            borderpad=6,
            xshift=20,  # 向左偏移
            xanchor='right',
            yanchor='middle'
        )

    # 如果显示标签
    if show_labels:
        label_y = y_max * 1.05

        fig.add_annotation(
            x=0.19, y=label_y,
            text="紫外线",
            showarrow=False,
            font=dict(size=14, color="white"),
            yanchor="bottom"
        )

        fig.add_annotation(
            x=(vis_min + vis_max) / 2, y=label_y,
            text="可见光",
            showarrow=False,
            font=dict(size=14, color="white"),
            yanchor="bottom"
        )

        fig.add_annotation(
            x=vis_max + (x_max - vis_max) * 0.3, y=label_y,
            text="红外线",
            showarrow=False,
            font=dict(size=14, color="white"),
            yanchor="bottom"
        )

        # 分界线
        fig.add_shape(type="line", x0=vis_min, x1=vis_min,
                      y0=y_max, y1=y_max * 0.95,
                      line=dict(color="white", width=2))
        fig.add_shape(type="line", x0=vis_max, x1=vis_max,
                      y0=y_max, y1=y_max * 0.95,
                      line=dict(color="white", width=2))

    # 布局设置
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=14),
        xaxis=dict(
            title=dict(text='波长 λ (μm)', font=dict(size=18, color='rgb(230,230,230)')),
            range=[x_min, x_max],  # 使用动态范围
            gridcolor='rgba(128,128,128,0.2)',
            color='rgb(180,180,180)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text='能量密度 (MW/m²/μm)', font=dict(size=18, color='rgb(230,230,230)')),
            range=[0, y_max],
            gridcolor='rgba(128,128,128,0.2)',
            color='rgb(180,180,180)',
            showgrid=True,
            zeroline=False
        ),
        hovermode='closest',
        height=600,
        margin=dict(l=80, r=40, t=40, b=60),
        legend=dict(
            x=0.7, y=0.98,
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='white',
            borderwidth=1
        )
    )

    return fig


def create_star_visualization(temperature, total_power):
    """
    创建2D圆形可视化（大小根据辐射功率变化）
    """
    color = kelvin_to_rgb(temperature)

    # 根据辐射功率计算圆的大小
    # 归一化：以太阳表面温度的功率为基准
    reference_power = stefan_boltzmann_law(5778)  # 太阳表面温度
    power_ratio = total_power / reference_power

    # 半径范围：0.5 到 2.0（相对于基准大小）
    radius = 0.5 + 1.5 * min(power_ratio / 10, 1.0)  # 限制最大为2倍

    # 创建圆形
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    fig = go.Figure()

    # 添加填充圆
    fig.add_trace(go.Scatter(
        x=x, y=y,
        fill='toself',
        fillcolor=color,
        line=dict(color='white', width=2),
        mode='lines',
        hoverinfo='text',
        hovertext=f'温度: {temperature} K<br>功率: {total_power / 1e6:.2f} MW/m²',
        showlegend=False
    ))

    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(
            visible=False,
            range=[-2.5, 2.5]
        ),
        yaxis=dict(
            visible=False,
            range=[-2.5, 2.5],
            scaleanchor="x",
            scaleratio=1
        ),
        width=120,
        height=120,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )

    return fig


# ==================== 主界面 ====================

def main():
    # 标题
    st.markdown("""
    <h1 style='text-align: center; color: #FFD700; font-size: 48px;'>
        🌟 黑体辐射仿真程序 🌟
    </h1>
    <p style='text-align: center; color: #ffffff; font-size: 18px;'>
        基于普朗克黑体辐射定律的交互式可视化
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 侧边栏控制面板
    with st.sidebar:
        st.markdown("## 🎛️ 控制面板")

        # 温度选择方式
        temp_mode = st.radio(
            "温度选择方式",
            ["滑块调节", "预设温度", "精确输入"],
            help="选择设置温度的方式"
        )

        if temp_mode == "滑块调节":
            temperature = st.slider(
                "黑体温度 (K)",
                min_value=MIN_T,
                max_value=MAX_T,
                value=DEFAULT_T,
                step=50,
                help="拖动滑块调节温度"
            )

        elif temp_mode == "预设温度":
            preset_temps = {
                "液氮 (77K)": 77,
                "干冰 (195K)": 195,
                "冰水混合物 (273K)": 273,
                "室温 (300K)": 300,
                "沸水 (373K)": 373,
                "白炽灯 (2850K)": 2850,
                "蜡烛 (1850K)": 1850,
                "太阳表面 (5778K)": 5778,
                "蓝色恒星 (10000K)": 10000,
                "白矮星 (8000K)": 8000
            }

            selected = st.selectbox(
                "选择预设温度",
                list(preset_temps.keys()),
                index=6  # 默认太阳表面
            )
            temperature = preset_temps[selected]

            st.info(f"**当前温度**: {temperature} K")

        else:  # 精确输入
            temperature = st.number_input(
                "输入温度 (K)",
                min_value=MIN_T,
                max_value=MAX_T,
                value=DEFAULT_T,
                step=100,
                help="直接输入精确温度值"
            )

        st.markdown("---")

        # 显示选项
        st.markdown("### 📊 显示选项")

        show_rj = st.checkbox(
            "显示瑞利-金斯公式",
            value=False,
            help="经典物理近似（长波长适用）"
        )

        show_wien = st.checkbox(
            "显示维恩公式",
            value=False,
            help="短波长近似"
        )

        show_labels = st.checkbox(
            "显示光谱分区标签",
            value=True,
            help="标注紫外线、可见光、红外线区域"
        )

        show_values = st.checkbox(
            "显示峰值数值",
            value=True,
            help="显示峰值波长和辐射强度的具体数值"
        )

        show_intensity = st.checkbox(
            "显示强度积分区域",
            value=False,
            help="填充曲线下方区域（积分表示总辐射）"
        )

        st.markdown("---")

        # 坐标轴范围
        # 坐标轴范围
        st.markdown("### 📏 坐标轴范围")

        # X轴范围模式选择
        x_range_mode = st.radio(
            "X轴范围模式",
            ["自动适应", "手动设置"],
            index=0,
            help="自动模式：根据温度自动调整显示范围"
        )

        if x_range_mode == "自动适应":
            # 阈值选择
            threshold_percent = st.slider(
                "显示阈值（峰值的百分比）",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="当辐射强度降到峰值的该百分比时截断显示"
            )

            # 自动计算范围
            x_min, x_max = calculate_optimal_x_range(temperature, threshold_percent / 100)

            st.info(f"📊 自动范围: {x_min:.2f} - {x_max:.2f} μm")

        else:
            # 手动设置
            col_x1, col_x2 = st.columns(2)
            with col_x1:
                x_min = st.number_input(
                    "波长最小值 (μm)",
                    min_value=0.01,
                    max_value=10.0,
                    value=0.1,
                    step=0.1,
                    format="%.2f"
                )
            with col_x2:
                x_max = st.number_input(
                    "波长最大值 (μm)",
                    min_value=1.0,
                    max_value=40.0,
                    value=10.0,
                    step=0.5,
                    format="%.1f"
                )

        # 自动计算合适的Y轴范围
        lambda_m_temp = np.linspace(0.02e-6, x_max * 1e-6, 1000)
        B_temp = planck_law(lambda_m_temp, temperature)
        auto_y_max = np.max(B_temp) * 1.2

        y_max = st.slider(
            "辐射强度最大值 (MW/m²/μm)",
            min_value=1.0,
            max_value=max(auto_y_max * 2, 100.0),
            value=float(auto_y_max),
            step=1.0,
            help="调整Y轴显示范围"
        )

        st.markdown("---")

        # 复位按钮
        if st.button("🔄 复位所有设置", use_container_width=True):
            st.rerun()

    # ==================== 主显示区域 ====================

    # 计算关键参数（提前计算，后面多处使用）
    peak_wavelength = wien_displacement_law(temperature)
    total_power = stefan_boltzmann_law(temperature)

    if peak_wavelength < 0.38:
        region = "紫外区"
    elif 0.38 <= peak_wavelength <= 0.78:
        region = "可见光区"
    else:
        region = "红外区"

    # === 第一行：图表标题 + 黑体圆 ===
    st.markdown("---")

    col_title, col_circle = st.columns([0.45, 0.55])

    with col_title:
        st.markdown("<h2 style='color: #ffffff;'>📈 辐射强度 vs 波长</h2>", unsafe_allow_html=True)

    with col_circle:
        # 在标题右边显示黑体圆（大小随功率变化）
        star_fig = create_star_visualization(temperature, total_power)
        st.plotly_chart(star_fig, use_container_width=False, key="star_circle")

    # === 第二行：坐标图 + 关键参数并排 ===
    col_chart, col_params = st.columns([0.8, 0.2])

    with col_chart:
        main_fig = create_main_plot(
            temperature, show_rj, show_wien, show_labels,
            show_values, show_intensity, x_min, x_max, y_max
        )
        st.plotly_chart(main_fig, use_container_width=True, key="main_plot")

    with col_params:
        # 自定义CSS：放大标签并改为白色
        # 自定义CSS：强制放大标签字号
        st.markdown("""
        <style>
        /* 强制修改metric标签样式 - 使用多重选择器提高优先级 */
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] > div,
        div[data-testid="stMetricLabel"] > div > div,
        div[data-testid="stMetricLabel"] label {
            color: #ffffff !important;
            font-size: 32px !important;
            font-weight: bold !important;
        }

        /* metric数值样式 */
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] > div {
            font-size: 32px !important;
            color: #FFD700 !important;
        }

        /* metric的delta（区域标签）样式 */
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] svg {
            font-size: 32px !important;
        }

        /* 强制所有字体继承 */
        [data-testid="stMetric"] * {
            font-family: sans-serif !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<h3 style='color: #ffffff;'>📊 关键参数</h3>", unsafe_allow_html=True)

        # 1. 温度
        st.markdown(f"""
        <div style='margin-bottom: 20px;'>
            <p style='color: #ffffff; font-size: 28px; font-weight: bold; margin: 0;'>温度</p>
            <p style='color: #FFD700; font-size: 28px; margin: 5px 0 0 0;'>{temperature} K</p>
        </div>
        """, unsafe_allow_html=True)

        # 2. 峰值波长
        st.markdown(f"""
        <div style='margin-bottom: 20px;'>
            <p style='color: #ffffff; font-size: 28px; font-weight: bold; margin: 0;'>峰值波长</p>
            <p style='color: #FFD700; font-size: 28px; margin: 5px 0 0 0;'>{peak_wavelength:.3f} μm</p>
            <p style='color: #4ade80; font-size: 18px; margin: 5px 0 0 0;'>↑ {region}</p>
        </div>
        """, unsafe_allow_html=True)

        # 3. 总辐射功率
        st.markdown(f"""
        <div style='margin-bottom: 20px;'>
            <p style='color: #ffffff; font-size: 28px; font-weight: bold; margin: 0;'>总辐射功率</p>
            <p style='color: #FFD700; font-size: 28px; margin: 5px 0 0 0;'>{total_power / 1e6:.2f} MW/m²</p>
        </div>
        """, unsafe_allow_html=True)

        # 4. 可见光比例
        lambda_vis = np.linspace(0.38e-6, 0.78e-6, 500)
        B_vis = planck_law(lambda_vis, temperature)

        if hasattr(np, 'trapezoid'):
            visible_power = np.trapezoid(B_vis, lambda_vis * 1e6) * 1e6
        else:
            visible_power = np.trapz(B_vis, lambda_vis * 1e6) * 1e6

        visible_ratio = (visible_power / total_power) * 100

        st.markdown(f"""
        <div style='margin-bottom: 20px;'>
            <p style='color: #ffffff; font-size: 28px; font-weight: bold; margin: 0;'>可见光比例</p>
            <p style='color: #FFD700; font-size: 28px; margin: 5px 0 0 0;'>{visible_ratio:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # 详细信息展开区域
    with st.expander("📚 理论知识与公式", expanded=False):
        st.markdown("""
        <div style='color: #ffffff;'>
        ### 普朗克黑体辐射定律

        黑体在温度 T 下，单位面积在波长 λ 处的辐射强度为：

        $$
        B(\lambda, T) = \\frac{2\pi hc^2}{\lambda^5} \\frac{1}{e^{\\frac{hc}{\lambda k_B T}} - 1}
        $$

        其中：
        - h = 6.626 × 10⁻³⁴ J·s （普朗克常数）
        - c = 2.998 × 10⁸ m/s （光速）
        - k_B = 1.381 × 10⁻²³ J/K （玻尔兹曼常数）

        ---

        ### 维恩位移定律

        峰值波长与温度成反比：

        $$
        \lambda_{max} \cdot T = 2.898 \\times 10^{-3} \\ \\text{m·K}
        $$

        ---

        ### 斯特藩-玻尔兹曼定律

        黑体总辐射功率与温度的四次方成正比：

        $$
        I = \sigma T^4
        $$

        其中 σ = 5.670 × 10⁻⁸ W/(m²·K⁴) （斯特藩-玻尔兹曼常数）

        ---

        ### 瑞利-金斯公式（经典近似）

        在长波长极限下的近似：

        $$
        B_{RJ}(\lambda, T) = \\frac{2\pi c k_B T}{\lambda^4}
        $$

        **注意**：该公式在短波长处会趋向无穷大（紫外灾难），说明经典物理的局限性。

        ---

        ### 维恩公式（短波长近似）

        在短波长极限下的近似：

        $$
        B_W(\lambda, T) = \\frac{2\pi hc^2}{\lambda^5} e^{-\\frac{hc}{\lambda k_B T}}
        $$
        </div>
        """, unsafe_allow_html=True)

    # 应用案例
    with st.expander("🌍 实际应用案例", expanded=False):
        st.markdown("""
        <div style='color: #ffffff;'>
        ### 🌟 天文学应用

        **1. 测量恒星表面温度**
        - 通过观测恒星光谱的峰值波长，利用维恩位移定律反推温度
        - 例：太阳峰值在 502 nm（绿光），对应温度约 5778 K

        **2. 恒星分类（光谱型）**
        - O型（蓝色，>30,000 K）
        - B型（蓝白色，10,000-30,000 K）
        - A型（白色，7,500-10,000 K）
        - F型（黄白色，6,000-7,500 K）
        - G型（黄色，5,200-6,000 K）← 太阳
        - K型（橙色，3,700-5,200 K）
        - M型（红色，2,400-3,700 K）

        ---

        ### 🏭 工业应用

        **1. 温度测量（红外测温仪）**
        - 通过测量物体的红外辐射强度推算温度
        - 非接触式测量，适用于高温环境

        **2. 白炽灯设计**
        - 钨丝灯约 2850 K，大部分能量在红外区（低效）
        - LED灯通过半导体发光，效率更高

        **3. 陶瓷烧制**
        - 通过观察陶瓷颜色判断窑内温度
        - 暗红→鲜红→橙→黄→白（温度递增）

        ---

        ### 🔬 科学研究

        **1. 宇宙微波背景辐射（CMB）**
        - 温度约 2.7 K
        - 峰值波长在微波波段（约 1.9 mm）
        - 证明大爆炸理论的重要证据

        **2. 量子力学的诞生**
        - 普朗克为解决"紫外灾难"提出能量量子化假设
        - 标志着量子力学的开端（1900年）
        </div>
        """, unsafe_allow_html=True)

    # 温度对比表
    with st.expander("🌡️ 常见物体的黑体温度参考", expanded=False):
        st.markdown("""
        <div style='color: #ffffff;'>
        | 物体 | 温度 (K) | 峰值波长 | 颜色 |
        |------|----------|----------|------|
        | 宇宙微波背景 | 2.7 | 1.07 mm | 不可见（微波） |
        | 液氮 | 77 | 37.6 μm | 不可见（远红外） |
        | 干冰（固态CO₂） | 195 | 14.9 μm | 不可见（红外） |
        | 冰点（0°C） | 273 | 10.6 μm | 不可见（红外） |
        | 人体 | 310 | 9.3 μm | 不可见（红外） |
        | 沸水（100°C） | 373 | 7.8 μm | 不可见（红外） |
        | 蜡烛火焰 | 1,850 | 1.57 μm | 暗红 |
        | 白炽灯钨丝 | 2,850 | 1.02 μm | 橙黄 |
        | 太阳表面 | 5,778 | 502 nm | 白色（微黄） |
        | 蓝色恒星 | 10,000 | 290 nm | 蓝白 |
        | 天狼星A | 9,940 | 292 nm | 蓝白 |
        | 参宿七 | 11,000 | 264 nm | 蓝色 |
        </div>
        """, unsafe_allow_html=True)

    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #ffffff;'>
        <p>📚 基于普朗克黑体辐射定律 | 🔬 物理常数来自 scipy.constants</p>
        <p>💡 交互式物理教学演示工具 | ⚡ Powered by Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)


# ==================== 程序入口 ====================
if __name__ == "__main__":
    main()