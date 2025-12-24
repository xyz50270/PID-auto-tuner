import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from src.modeling import FOPDTModel

@dataclass
class ControllerStats:
    total_variation: float  # Total movement of OP (Valve wear proxy)
    aggressiveness: float   # Ratio of OP change to PV error
    avg_sampling_time: float # Mean dt
    max_sampling_time: float # Max dt
    duration: float
    data_quality_score: float # 0-100

@dataclass
class SufficiencyCheck:
    is_sufficient: bool
    required_duration: float
    current_duration: float
    message: str
    suggestions: List[str]

def analyze_controller_characteristics(df: pd.DataFrame) -> ControllerStats:
    """
    Quantify controller behavior using time-weighted metrics.
    Handles uneven timestamps robustly.
    """
    df = df.sort_values('Time')
    t_sec = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds().values
    dt = np.diff(t_sec)
    
    # Avoid div by zero
    dt = np.where(dt < 1e-6, 1e-6, dt)
    
    op = df['OP'].values
    pv = df['PV'].values
    sp = df['SP'].values
    error = sp - pv
    
    # 1. Total Variation (TV) - Measure of Control Effort / Valve Wear
    # TV = Sum( |OP_i - OP_{i-1}| )
    # Normalized by time duration to get "Movement per Minute" maybe?
    # Let's keep it absolute sum first, but imply rate.
    op_diff = np.diff(op)
    total_variation = np.sum(np.abs(op_diff))
    
    # Normalize TV per minute for readability
    duration_min = t_sec[-1] / 60.0 if t_sec[-1] > 0 else 1.0
    tv_per_min = total_variation / duration_min
    
    # 2. Aggressiveness
    # StdDev(Delta OP) / StdDev(Error)
    # If error is small but OP moves a lot -> Aggressive / Noise Amplification
    std_d_op = np.std(op_diff)
    std_error = np.std(error)
    
    if std_error < 1e-6:
        aggressiveness = 0.0 # Perfect control or dead sensor
    else:
        aggressiveness = std_d_op / std_error

    # 3. Sampling Stats
    avg_dt = np.mean(dt)
    max_dt = np.max(dt)
    
    # Quality Score (Simple heuristic)
    # Penalize high max_dt variance
    score = 100.0
    if max_dt > 5 * avg_dt:
        score -= 20 # Missing data potential
    if avg_dt > 10.0: # Very slow sampling
        score -= 10
        
    return ControllerStats(
        total_variation=tv_per_min,
        aggressiveness=aggressiveness,
        avg_sampling_time=avg_dt,
        max_sampling_time=max_dt,
        duration=t_sec[-1],
        data_quality_score=max(0, score)
    )

def check_data_sufficiency(df: pd.DataFrame, model: Optional[FOPDTModel] = None) -> SufficiencyCheck:
    """
    Check if data covers enough dynamic response time.
    """
    t_sec = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds().values
    duration = t_sec[-1]
    
    suggestions = []
    
    # 1. Check Density
    n_points = len(df)
    if n_points < 50:
        return SufficiencyCheck(False, 0, duration, "数据点过少", ["请收集至少 50 个数据点以进行有效分析"])
        
    # 2. Check Duration vs Model
    if model:
        # Settling time approx 4 * Tau + Theta
        # We want to see at least one full settling cycle, ideally more.
        process_time_scale = 4 * model.tau + model.theta
        required = process_time_scale * 1.2 # Safety margin
        
        if duration < required:
            shortage = required - duration
            msg = f"数据覆盖不足。过程预计响应周期约为 {process_time_scale:.1f}秒，当前仅 {duration:.1f}秒。"
            suggestions.append(f"建议延长采集时间：至少还需 {shortage:.1f} 秒")
            suggestions.append(f"当前数据可能处于瞬态，无法反映最终稳态，导致整定建议不准。")
            
            return SufficiencyCheck(False, required, duration, msg, suggestions)
    else:
        # No model yet. Heuristic check based on dominant dynamics?
        # Or simply warn if < 60s (Arbitrary default)
        if duration < 60:
             suggestions.append("数据过短，可能无法捕捉完整动态。")
             return SufficiencyCheck(True, 60, duration, "数据较短 (无模型参考)", suggestions) # Soft warning
             
    return SufficiencyCheck(True, 0, duration, "数据覆盖度良好", [])

