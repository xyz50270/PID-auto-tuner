import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import pickle
import datetime
from io import BytesIO

# 将项目根目录添加到 sys.path，以便能够导入 src 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import load_and_validate_csv, IngestionError
from src.diagnosis import analyze_loop_health, HealthStatus
from src.modeling import fit_fopdt, FOPDTModel
from src.tuning import calculate_imc_pid, suggest_parameters, PIDParams, TuningSuggestion
from src.simulation import simulate_closed_loop
from src.evaluation import calculate_metrics, PerformanceMetrics
from src.analysis import analyze_controller_characteristics, check_data_sufficiency, ControllerStats

st.set_page_config(page_title="PID 迭代整定与智能诊断系统", layout="wide")

# --- CSS 注入：强制移除垃圾桶按钮的背景衬底 ---
st.markdown(r"""
<style>
    /* 针对侧边栏历史记录中的垃圾桶按钮，强力移除背景、边框和阴影 */
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stColumn"] button {
        border: none !important;
        background-color: transparent !important;
        padding: 0 !important;
        box-shadow: none !important;
        color: #888 !important;
        font-size: 1.1rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 2rem !important;
        width: 100% !important;
        min-height: unset !important;
    }
    
    /* 悬停效果：仅改变颜色，不显示背景 */
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stColumn"] button:hover {
        color: #ff4b4b !important;
        background-color: rgba(255, 75, 75, 0.1) !important;
    }

    /* 让左侧的文字与垃圾桶图标在视觉上对齐 */
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stColumn"] .stText {
        display: flex;
        align-items: center;
        height: 2rem;
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

# --- 辅助函数：绘制过程数据趋势图 ---
def plot_time_series(df, title="实时过程数据趋势图"):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=df['Time'], y=df['SP'], name='设定值 (SP/SetPoint)', line=dict(color='green', dash='dash')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['PV'], name='过程变量 (PV/ProcessVar)', line=dict(color='blue')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['OP'], name='控制器输出 (OP/Output)', line=dict(color='red'), opacity=0.4), secondary_y=True)
    
    fig.update_layout(
        title=title, 
        hovermode="x unified", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="PV / SP (工程单位)", secondary_y=False)
    fig.update_yaxes(title_text="输出值 OP (%)", secondary_y=True)
    return fig

# --- 辅助函数：渲染整定建议详情卡片 ---
def render_tuning_suggestion(suggestion: TuningSuggestion):
    st.markdown("### 🔍 详细整定建议面板")
    
    is_pb = st.session_state.get('pid_mode') == "PB"
    mode_str = st.session_state.get('pid_mode', 'Kp')
    p_label = "比例度 PB (%)" if is_pb else "比例增益 Kp"
    
    def get_p_val(pid):
        return pid.PB if is_pb else pid.Kp

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("###### 1. 该阶段当前参数 (Current)")
        st.metric(p_label, f"{get_p_val(suggestion.current_pid):.4f}", help="该段数据采集时实际生效的比例参数。")
        st.metric("积分时间 Ti (s)", f"{suggestion.current_pid.Ti:.2f}", help="该段数据采集时实际生效的积分时间（单位：秒）。")
        st.metric("微分时间 Td (s)", f"{suggestion.current_pid.Td:.2f}")

    with col2:
        st.success("###### 2. 基于此阶段建议调整 (Next Step)")
        st.metric(f"建议 {p_label.split()[-1]}", f"{get_p_val(suggestion.next_step_pid):.4f}", help=suggestion.get_delta_desc('Kp', mode=mode_str))
        st.metric("建议 Ti (s)", f"{suggestion.next_step_pid.Ti:.2f}", help=suggestion.get_delta_desc('Ti', mode=mode_str))
        st.metric("建议 Td (s)", f"{suggestion.next_step_pid.Td:.2f}", help=suggestion.get_delta_desc('Td', mode=mode_str))
        
    with col3:
        st.warning("###### 3. 最终理论目标值 (Target)")
        st.metric(f"理论目标 {p_label.split()[-1]}", f"{get_p_val(suggestion.target_pid):.4f}", help="根据辨识出的物理模型计算出的理论最优比例参数值。")
        st.metric("理论目标 Ti (s)", f"{suggestion.target_pid.Ti:.2f}", help="理论最优积分时间。")
        st.metric("理论目标 Td (s)", f"{suggestion.target_pid.Td:.2f}")

    if suggestion.warnings:
        st.markdown("#### ⚠️ 调整步长限制说明")
        for w in suggestion.warnings:
            st.caption(f"· {w}")
    else:
        st.caption("✅ 建议调整值已完美匹配理论目标，无需进一步分步。" )

# --- 帮助与技术文档页面渲染 ---
def render_help_page():
    st.markdown(r"""
    # 📖 PID 迭代整定系统 - 帮助与文档中心
    
    ## 1. PID 控制基础原理
    PID 控制器通过对误差进行比例 (P)、积分 (I) 和微分 (D) 运算来生成输出 (OP)：
    
    *   **P (比例参数)**: 
        *   **比例增益 Kp**: 决定对当前误差的调节力度。Kp 越大，响应越快，但过大会引起系统震荡。
        *   **比例度 PB (%)**: 输出变化 100% 时对应的输入偏差占量程的百分比。**PB 越小，控制作用越强**。
        *   **关系**: $PB = 100 / Kp$。
    *   **I (积分时间 Ti)**: 用于消除稳态误差。Ti 越小，积分项作用越强，消除残余偏差的速度越快。
    *   **D (微分时间 Td)**: 预测未来趋势，起到“提前刹车”的作用，有助于抑制超调并改善动态稳定性。
        
    ## 2. 核心术语定义 (Terminology) 
    
    | 缩写 | 中文全称 | 物理描述与作用 |
    | :--- | :--- | :--- |
    | **SP** | **设定值** | 您的控制目标点（如目标温度、目标压力）。 |
    | **PV** | **过程变量** | 传感器实时反馈的测量值。 |
    | **OP** | **控制器输出** | 控制器发出的指令，通常表现为阀门开度、变频器频率等。 |
    | **IAE** | **绝对误差积分** | 衡量控制精度的金标准。IAE 越小，代表过程越贴近设定值。 |
    | **TV** | **总变差 (Total Variation)** | 反映 OP 的动作频繁程度。TV 越高，执行器的物理磨损风险越大。 |

    ## 3. 过程辨识模型 (FOPDT)
    系统通过您的数据自动拟合“一阶加纯滞后”物理模型：
    $$ G(s) = \frac{K e^{-\theta s}}{\tau s + 1} $$
    
    *   **增益 K (Gain)**: 灵敏度。表示 OP 改变 1% 最终会引起 PV 改变多少。
    *   **时间常数 τ (Tau)**: 惯性。反映系统响应扰动并达到最终稳定值 63.2% 所需的时间。
    *   **滞后时间 θ (Theta)**: 纯死区。反映从发出指令到 PV 产生反应之间的物理延迟。
    
    ## 4. 如何进行迭代优化
    1.  **上传基准数据**: 上传一段在当前 PID 参数下运行的 CSV 数据。
    2.  **模型辨识**: 在工作台中点击“辨识模型”。系统将确定该阶段下被控对象的特性。
    3.  **获取建议**: 系统根据安全步长计算建议的 PID。
    4.  **循环优化**: 应用新参数后再次上传数据，系统会根据新响应**自适应**地修正后续建议。
    
    ---
    *注：工业现场安全第一。本工具提供的所有参数建议均作为辅助工程参考。*
    """)

# --- 主程序逻辑入口 ---
def main():
    st.title("🏭 PID 迭代整定与智能诊断系统")
    
    # 初始化 Session 状态
    if 'datasets' not in st.session_state:
        st.session_state['datasets'] = []
    if 'confirm_reset' not in st.session_state:
        st.session_state['confirm_reset'] = False
    if 'pending_delete_idx' not in st.session_state:
        st.session_state['pending_delete_idx'] = None
    if 'pid_mode_toggle' not in st.session_state:
        st.session_state['pid_mode_toggle'] = False
    if 'last_pid_mode' not in st.session_state:
        st.session_state['last_pid_mode'] = "Kp"
        
    # --- 侧边栏：全局配置 ---
    st.sidebar.header("⚙️ 全局配置")
    st.sidebar.toggle(
        "使用比例度 (PB) 模式", 
        key='pid_mode_toggle',
        help="开启后，所有比例参数将以比例度 (%) 形式显示和录入。关系：PB = 100 / Kp"
    )
    st.session_state['pid_mode'] = "PB" if st.session_state['pid_mode_toggle'] else "Kp"
    
    # 模式切换时的实时换算逻辑 (针对当前录入框)
    if st.session_state['pid_mode'] != st.session_state['last_pid_mode']:
        curr_p_key = f"p_v8_{len(st.session_state['datasets'])}"
        if curr_p_key in st.session_state:
            old_val = st.session_state[curr_p_key]
            # 换算公式: 新值 = 100 / 旧值 (Kp 和 PB 互为倒数关系 * 100)
            st.session_state[curr_p_key] = 100.0 / old_val if abs(old_val) > 1e-9 else 0.0
        st.session_state['last_pid_mode'] = st.session_state['pid_mode']
    
    # --- 侧边栏：会话管理与持久化 ---
    with st.sidebar.expander("💾 会话与进度管理", expanded=False):
        if st.session_state['datasets']:
            # 自动添加时间戳后缀以防止文件名冲突
            ts_label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                buffer = BytesIO()
                pickle.dump(st.session_state['datasets'], buffer)
                buffer.seek(0)
                st.download_button(
                    label="导出当前整定进度",
                    data=buffer,
                    file_name=f"pid_session_{ts_label}.pkl",
                    mime="application/octet-stream",
                    help="将当前所有历史数据、模型和 PID 轨迹打包下载到本地。文件名已自动增加时间戳。",
                    width='stretch'
                )
            except Exception as e:
                st.error(f"进度打包失败: {e}")
        
        st.markdown("---")
        # 会话恢复
        upl_sess = st.file_uploader("从本地加载进度文件", type=["pkl"], key="sess_v8_final")
        if upl_sess:
            if st.button("确认恢复会话数据", width='stretch', key="btn_res_v8_final"):
                try:
                    st.session_state['datasets'] = pickle.load(upl_sess)
                    st.rerun()
                except Exception as e:
                    st.error(f"文件加载失败: {e}")
        
        st.markdown("---")
        # 会话重置
        if not st.session_state['confirm_reset']:
            if st.button("🔴 重置当前任务", width='stretch'):
                st.session_state['confirm_reset'] = True
                st.rerun()
        else:
            st.warning("⚠️ 确定要清空所有数据吗？操作前建议先通过上方按钮保存进度文件。" )
            cr1, cr2 = st.columns(2)
            if cr1.button("确认清空", type="primary", width='stretch'):
                st.session_state['datasets'] = []
                st.session_state['confirm_reset'] = False
                st.rerun()
            if cr2.button("取消", width='stretch'):
                st.session_state['confirm_reset'] = False
                st.rerun()

    # --- 侧边栏：历史记录与数据删除 ---
    st.sidebar.divider()
    st.sidebar.header("⏱️ 整定历史记录")
    
    n_ds = len(st.session_state['datasets'])
    if n_ds == 0:
        st.sidebar.info("👋 欢迎！请在下方上传**初始状态 (Baseline) 数据**开始整定。" )
    else:
        st.sidebar.success(f"✅ 系统已记录 {n_ds} 轮分析数据。" )
        with st.sidebar.expander("📜 历史阶段管理", expanded=True):
            for i in range(n_ds - 1, -1, -1):
                c_name, c_del = st.columns([0.8, 0.2])
                ds_item = st.session_state['datasets'][i]
                c_name.text(f"#{i+1}: {ds_item['name']}")
                if i > 0: # Baseline 节点禁止删除
                    # 纯图标按钮，通过顶部注入的 CSS 去除背景
                    if c_del.button("🗑️", key=f"btn_del_v8_{i}", help="点击开启删除确认"):
                        st.session_state['pending_delete_idx'] = i
                        st.rerun()
        
        # 删除确认弹窗
        if st.session_state['pending_delete_idx'] is not None:
            d_idx = st.session_state['pending_delete_idx']
            if 0 <= d_idx < len(st.session_state['datasets']):
                d_name = st.session_state['datasets'][d_idx]['name']
                st.sidebar.warning(f"确定删除阶段: '{d_name}' 吗？")
                dc1, dc2 = st.sidebar.columns(2)
                if dc1.button("确认删除", type="primary", key="cfm_del_v8_f"):
                    st.session_state['datasets'].pop(d_idx)
                    st.session_state['pending_delete_idx'] = None
                    st.rerun()
                if dc2.button("取消操作", key="can_del_v8_f"):
                    st.session_state['pending_delete_idx'] = None
                    st.rerun()

    # --- 侧边栏：数据录入表单 ---
    st.sidebar.divider()
    step_lbl = "基准状态" if n_ds == 0 else f"调整 #{n_ds} 后响应数据"
    st.sidebar.markdown(f"### 📥 录入阶段数据 ({step_lbl})")
    upl_name = st.sidebar.text_input("给此阶段起个名字", value=f"Adjustment_{n_ds}" if n_ds > 0 else "Baseline")
    
    is_pb = st.session_state['pid_mode'] == "PB"
    p_label = "比例度 PB (%)" if is_pb else "比例 Kp"
    
    st.sidebar.markdown("#### ⚙️ 该阶段运行时的 PID 参数")
    c1, c2, c3 = st.sidebar.columns(3)
    
    # 定义 Key
    p_key = f"p_v8_{n_ds}"
    i_key = f"i_v8_{n_ds}"
    d_key = f"d_v8_{n_ds}"
    
    # 如果是首次进入该阶段，初始化 Session State 中的值
    if p_key not in st.session_state:
        pk, pi, pdv = 1.0, 10.0, 0.0
        if n_ds > 0:
            lp = st.session_state['datasets'][-1]['pid']
            pk, pi, pdv = (lp.PB if is_pb else lp.Kp), lp.Ti, lp.Td
        elif is_pb:
            pk = 100.0
        
        st.session_state[p_key] = float(pk)
        st.session_state[i_key] = float(pi)
        st.session_state[d_key] = float(pdv)
        
    # 使用 key 绑定，不再传入 value 参数以避免冲突
    p_in = c1.number_input(p_label, key=p_key)
    ti_in = c2.number_input("积分 Ti", key=i_key)
    td_in = c3.number_input("微分 Td", key=d_key)
    
    upl_file = st.sidebar.file_uploader("上传 CSV 响应数据", type=["csv"], key=f"upl_v8_{n_ds}")
    st.sidebar.caption("数据需包含列: Time(时间), SP(设定值), PV(过程变量), OP(输出)。")
    
    if upl_file:
        try:
            # ... (previous code for CSV mapping)
            df_preview = pd.read_csv(upl_file)
            cols = df_preview.columns.tolist()
            upl_file.seek(0)
            cmap = {}
            for c in cols:
                cl = c.lower()
                if 'time' in cl or 'date' in cl: cmap[c] = 'Time'
                elif 'sp' in cl or 'set' in cl: cmap[c] = 'SP'
                elif 'pv' in cl or 'process' in cl: cmap[c] = 'PV'
                elif 'op' in cl or 'out' in cl: cmap[c] = 'OP'
            
            df = load_and_validate_csv(upl_file, cmap)
            if st.sidebar.button("确认添加此轮数据并分析", width='stretch', key=f"btn_add_v8_{n_ds}"):
                final_pid = PIDParams.from_pb(p_in, ti_in, td_in) if is_pb else PIDParams(p_in, ti_in, td_in)
                new_e = {
                    'name': upl_name, 'df': df, 'pid': final_pid,
                    'metrics': calculate_metrics(df), 'ctrl_stats': analyze_controller_characteristics(df), 'model': None
                }
                st.session_state['datasets'].append(new_e)
                st.rerun()
        except Exception as e: 
            st.sidebar.error(f"数据解析失败: {e}")

    # --- 主界面布局 ---
    mt1, mt2 = st.tabs(["🛠️ 自动整定工作台", "📖 帮助与技术文档"])
    with mt2: render_help_page()
    with mt1:
        if not st.session_state['datasets']:
            st.info("👈 请先在左侧侧边栏上传初始数据。" )
            return
            
        t1, t2, t3 = st.tabs(["📈 进化趋势看板", "🩺 诊断与模型辨识", "🔍 原始响应分析"])
        
        # --- Tab 1: 进化看板 ---
        with t1:
            st.subheader("整定效果迭代演变看板")
            h_list = []
            for ds in st.session_state['datasets']:
                p_val = ds['pid'].PB if is_pb else ds['pid'].Kp
                row = {
                    "阶段名称": ds['name'], p_label: p_val, "积分 Ti (s)": ds['pid'].Ti, "微分 Td (s)": ds['pid'].Td,
                    "IAE 误差": ds['metrics'].iae, "超调量 (%)": ds['metrics'].overshoot
                }
                if 'ctrl_stats' in ds:
                    row["输出动作变差(TV)"] = ds['ctrl_stats'].total_variation
                    row["控制攻击性"] = ds['ctrl_stats'].aggressiveness
                h_list.append(row)
            df_h = pd.DataFrame(h_list)
            st.dataframe(df_h, width='stretch')
            
            cg1, cg2 = st.columns(2)
            with cg1:
                fi = go.Figure()
                fi.add_trace(go.Scatter(x=df_h['阶段名称'], y=df_h['IAE 误差'], mode='lines+markers', name='IAE 趋势'))
                fi.update_layout(title="控制精度 (IAE) 下降趋势 (越低越好)", yaxis_title="IAE 指标值")
                st.plotly_chart(fi, width='stretch')
            with cg2:
                fp = go.Figure()
                fp.add_trace(go.Scatter(x=df_h['阶段名称'], y=df_h[p_label], mode='lines+markers', name=p_label))
                fp.add_trace(go.Scatter(x=df_h['阶段名称'], y=df_h['积分 Ti (s)'], mode='lines+markers', name='积分 Ti', yaxis='y2'))
                fp.add_trace(go.Scatter(x=df_h['阶段名称'], y=df_h['微分 Td (s)'], mode='lines+markers', name='微分 Td', yaxis='y2'))
                fp.update_layout(
                    title="PID 参数演变路径图", 
                    yaxis=dict(title=p_label), 
                    yaxis2=dict(title="时间参数 Ti/Td (s)", overlaying='y', side='right')
                )
                st.plotly_chart(fp, width='stretch')

        # --- Tab 2: 诊断与辨识 (支持回溯) ---
        with t2:
            ds_names_back = [f"#{i+1}: {d['name']}" for i, d in enumerate(st.session_state['datasets'])]
            sel_idx_back = st.selectbox("选择要回溯或分析的历史阶段", range(len(ds_names_back)), 
                                        index=len(ds_names_back)-1, format_func=lambda x: ds_names_back[x])
            cur_ds = st.session_state['datasets'][sel_idx_back]
            
            st.markdown(f"### 📍 当前正在查看分析: {cur_ds['name']}")
            
            with st.expander("🩺 回路健康诊断报告与统计", expanded=True):
                res_diag = analyze_loop_health(cur_ds['df'])
                if res_diag.issues:
                    for iss in res_diag.issues: st.warning(f"⚠️ {iss}")
                else:
                    st.success("✅ 该阶段回路状态健康，未检测到震荡或执行器饱和。" )
                    
                if 'ctrl_stats' in cur_ds:
                    st.markdown("#### 🎮 控制器执行特性评价")
                    cs1, cs2, cs3 = st.columns(3)
                    cs1.metric("OP 总变差 (TV)", f"{cur_ds['ctrl_stats'].total_variation:.1f}", help="反映执行器的物理动作强度与磨损风险。" )
                    cs2.metric("控制攻击性", f"{cur_ds['ctrl_stats'].aggressiveness:.2f}", help="控制器对误差的反应速度。过高可能放大噪音。" )
                    cs3.metric("采样质量评分", f"{cur_ds['ctrl_stats'].data_quality_score:.0f}/100")

            st.divider()
            st.subheader("🚀 过程物理模型辨识")
            ct1, ct2 = st.columns([1, 1])
            with ct1:
                st.markdown("根据当前阶段的测量响应，自动辨识对象的 FOPDT 模型。建议将基于此模型**自适应**更新。" )
                if st.button("辨识此阶段物理模型", key=f"btn_fit_v8_{sel_idx_back}"):
                    with st.spinner("正在计算非线性回归拟合模型..."):
                        try:
                            m_result = fit_fopdt(cur_ds['df'])
                            cur_ds['model'] = m_result
                            st.success("模型辨识成功！")
                            check_s = check_data_sufficiency(cur_ds['df'], m_result)
                            if not check_s.is_sufficient:
                                st.warning(f"⚠️ {check_s.message}")
                                for su in check_s.suggestions: st.markdown(f"- {su}")
                        except Exception as e: st.error(f"辨识失败: {e}")
            with ct2:
                if cur_ds['model']:
                    m_val = cur_ds['model']
                    st.info(f"**模型参数**: 增益 K={m_val.K:.4f}, 时间常数 τ={m_val.tau:.2f}s, 滞后 θ={m_val.theta:.2f}s")
                    tf_v = (cur_ds['df']['Time'] - cur_ds['df']['Time'].iloc[0]).dt.total_seconds().values
                    pp_v = m_val.predict(cur_ds['df']['OP'].values, tf_v)
                    ff_v = go.Figure()
                    ff_v.add_trace(go.Scatter(x=cur_ds['df']['Time'], y=cur_ds['df']['PV'], name='实际测量 PV'))
                    ff_v.add_trace(go.Scatter(x=cur_ds['df']['Time'], y=pp_v, name='模型拟合 PV', line=dict(dash='dash')))
                    ff_v.update_layout(title="拟合质量验证 (拟合度越高建议越可靠)", height=250, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(ff_v, width='stretch')

            st.divider()
            if cur_ds['model']:
                st.subheader("💡 针对此阶段的 PID 调整建议")
                mod_pref = st.radio("整定偏好策略", ["保守 (Conservative)", "适中 (Moderate)", "激进 (Aggressive)"], 
                                    index=1, horizontal=True, key=f"rad_v8_{sel_idx_back}")
                mm_map = {"保守 (Conservative)": "conservative", "适中 (Moderate)": "moderate", "激进 (Aggressive)": "aggressive"}
                
                with st.expander("ℹ️ 三种策略的具体区别说明", expanded=False):
                    st.markdown(r"""
                    **SIMC 整定标准说明**:
                    *   **保守**: 设定闭环时间常数 $\tau_c = 10\theta$。系统极其稳定，无超调，响应慢。
                    *   **适中**: 设定 $\tau_c = 3\theta$。工业平衡标准，兼顾速度与稳定性。
                    *   **激进**: 设定 $\tau_c = \theta$。响应极快，旨在迅速抵消干扰，但会有一定超调。
                    """
                    )
                
                tp_target = calculate_imc_pid(cur_ds['model'], aggressiveness=mm_map[mod_pref])
                sug_step = suggest_parameters(cur_ds['pid'], tp_target, max_change_percent=20.0)
                render_tuning_suggestion(sug_step)
                
                with st.expander("🔮 闭环响应仿真对比 (当前 vs 建议)", expanded=False):
                    sd_val = st.slider("仿真时长 (秒)", 100, 3600, int(cur_ds['model'].tau * 10), key=f"sli_v8_{sel_idx_back}")
                    ts_ax = np.linspace(0, sd_val, 500); stm_pt = sd_val * 0.05
                    def ssp_func(t): return 10.0 if t > stm_pt else 0.0
                    rc_res = simulate_closed_loop(cur_ds['model'], sug_step.current_pid, ssp_func, ts_ax)
                    rn_res = simulate_closed_loop(cur_ds['model'], sug_step.next_step_pid, ssp_func, ts_ax)
                    rt_res = simulate_closed_loop(cur_ds['model'], sug_step.target_pid, ssp_func, ts_ax)
                    fs_fig = go.Figure()
                    fs_fig.add_trace(go.Scatter(x=ts_ax, y=rc_res['SP'], name='设定值 SP 阶跃', line=dict(color='green', dash='dash')))
                    fs_fig.add_trace(go.Scatter(x=ts_ax, y=rc_res['PV'], name='当前参数响应(灰色)', line=dict(color='gray')))
                    fs_fig.add_trace(go.Scatter(x=ts_ax, y=rn_res['PV'], name='本次建议响应(蓝色)', line=dict(color='blue')))
                    fs_fig.add_trace(go.Scatter(x=ts_ax, y=rt_res['PV'], name='理论最终目标响应(橙色)', line=dict(color='orange', dash='dot')))
                    fs_fig.update_layout(title=f"预测曲线 (基于模型辨识死区滞后: {cur_ds['model'].theta:.2f}s)", xaxis_title="时间 (s)", yaxis_title="过程变量 PV")
                    st.plotly_chart(fs_fig, width='stretch')

        # --- Tab 3: 原始数据趋势分析 ---
        with t3:
            s_name_raw = st.selectbox("选择要查看的原始响应阶段", [d['name'] for d in st.session_state['datasets']], key="sel_t3_v8")
            s_data_raw = next(d for d in st.session_state['datasets'] if d['name'] == s_name_raw)
            st.plotly_chart(plot_time_series(s_data_raw['df'], title=f"{s_name_raw} - 原始数据响应详情"), width='stretch')
            st.markdown("### 📊 该阶段性能核心指标 (KPI)")
            met_vals = s_data_raw['metrics']
            mk1, mk2, mk3, mk4 = st.columns(4)
            mk1.metric("IAE (绝对误差积分)", f"{met_vals.iae:.2f}", help="绝对误差随时间的累积。反映整体控制精度。" )
            mk2.metric("ISE (平方误差积分)", f"{met_vals.ise:.2f}", help="对大幅波动的惩罚更重。反映系统的抗扰稳定性。" )
            mk3.metric("观察到的最大超调", f"{met_vals.overshoot:.1f}%", help="该阶段中 PV 超过设定值 SP 的比例。" )
            mk4.metric("调节时间 (s)", f"{met_vals.settling_time:.1f}", help="系统进入并保持在 ±5% 误差带内所需的时间。" )
            st.caption("注：通过对比不同阶段的 IAE 趋势，可以量化参数调整带来的实际闭环改进。" )

if __name__ == "__main__":
    main()
