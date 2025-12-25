import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

class HealthStatus(Enum):
    HEALTHY = "Healthy"
    WARNING = "Warning"
    CRITICAL = "Critical"

@dataclass
class DiagnosisResult:
    status: HealthStatus
    issues: List[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    # New: Masks for plotting
    saturation_mask: Optional[np.ndarray] = None
    oscillation_mask: Optional[np.ndarray] = None
    stiction_mask: Optional[np.ndarray] = None

@dataclass
class ValveHealthReport:
    linearity_score: float # 0-100, 100 is perfectly linear
    avg_gain: float
    gain_by_range: dict # OP_Bin -> Gain
    stiction_zones: list # List of (min_op, max_op) with high stiction
    erosion_risk: bool
    suggestions: list[str] = field(default_factory=list)
    # New: Mask for stiction points
    raw_stiction_mask: Optional[np.ndarray] = None

def analyze_advanced_valve_health(df: pd.DataFrame) -> ValveHealthReport:
    """
    Perform deep analysis of valve mechanical and static characteristics.
    """
    op = df['OP'].values
    pv = df['PV'].values
    
    # 1. Gain Linearity Analysis (Binning OP)
    bins = np.arange(0, 101, 10)
    gain_map = {}
    
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        mask = (op >= low) & (op < high)
        if mask.sum() > 10: # Require minimum data points
            # Simple local gain estimate: Delta PV / Delta OP
            d_op = op[mask].max() - op[mask].min()
            d_pv = pv[mask].max() - pv[mask].min()
            if d_op > 0.5: # Sufficient movement
                gain_map[f"{low}-{high}%"] = d_pv / d_op
    
    gains = list(gain_map.values())
    avg_gain = np.mean(gains) if gains else 1.0
    cv = (np.std(gains) / avg_gain) if gains and avg_gain != 0 else 0
    linearity_score = max(0, 100 * (1 - cv))
    
    # 2. Erosion Detection (Focus on low opening < 15%)
    erosion_risk = False
    low_op_mask = (op < 15) & (op > 1)
    if low_op_mask.sum() > 5:
        # Check if local gain at low opening is significantly higher
        low_gains = [v for k, v in gain_map.items() if float(k.split('-')[0]) < 15]
        if low_gains and np.mean(low_gains) > 1.5 * avg_gain:
            erosion_risk = True

    # 3. Stiction Mapping
    # Identify samples where OP moves but PV stays still
    window = 5
    op_std = df['OP'].rolling(window=window).std()
    pv_std = df['PV'].rolling(window=window).std()
    
    # Heuristic: OP moves > 0.2% but PV moves < noise floor
    stiction_mask = (op_std > 0.2) & (pv_std < 0.05)
    stiction_mask = stiction_mask.fillna(False).values
    stiction_ops = op[stiction_mask]
    
    stiction_zones = []
    if len(stiction_ops) > 0:
        counts, edges = np.histogram(stiction_ops, bins=bins)
        # Zones where more than 10% of total stiction points occur
        for i in range(len(counts)):
            if counts[i] > 0.1 * len(stiction_ops) and counts[i] > 5:
                stiction_zones.append((edges[i], edges[high_idx if 'high_idx' in locals() else i+1])) # Fix logic later if needed
                stiction_zones[-1] = (edges[i], edges[i+1]) # Corrected

    # 4. Generate Suggestions
    suggestions = []
    if linearity_score < 70:
        suggestions.append("阀门线性度较差：建议检查凸轮机构或定位器线性化配置。")
    if erosion_risk:
        suggestions.append("检测到冲刷腐蚀风险：阀门在小开度下增益异常，建议检查阀座与阀芯。")
    if stiction_zones:
        zone_str = ", ".join([f"{int(z[0])}-{int(z[1])}%" for z in stiction_zones])
        suggestions.append(f"特定区间粘滞：在 {zone_str} 存在明显摩擦力不均，建议检查阀杆润滑或填料压盖。")
    if not suggestions:
        suggestions.append("阀门机械特性良好：可优先通过调节 PID 参数优化性能。")

    return ValveHealthReport(
        linearity_score=linearity_score,
        avg_gain=avg_gain,
        gain_by_range=gain_map,
        stiction_zones=stiction_zones,
        erosion_risk=erosion_risk,
        suggestions=suggestions,
        raw_stiction_mask=stiction_mask
    )

def analyze_loop_health(df: pd.DataFrame) -> DiagnosisResult:
    """
    Analyze PID loop health indicators: Saturation, Noise, Oscillation.
    """
    issues = []
    status = HealthStatus.HEALTHY
    details = {}

    # Common parameters
    sp_mean = df['SP'].mean() if len(df) > 0 else 100.0
    err_threshold = max(0.01 * abs(sp_mean), 0.5)

    # 1. Saturation Check
    op_max = df['OP'].max()
    op_min = df['OP'].min()
    
    op_range = op_max - op_min
    if op_range < 1e-6:
        op_range = 1.0
        
    tol = 0.01 * op_range
    if tol < 0.1: tol = 0.1 
    
    error = df['SP'] - df['PV']
    avg_error = error.mean()
    
    is_at_max = (df['OP'] >= op_max - tol)
    is_at_min = (df['OP'] <= op_min + tol)
    
    high_sat_points = is_at_max & (error > err_threshold)
    low_sat_points = is_at_min & (error < -err_threshold)
    sat_mask = (high_sat_points | low_sat_points).values
    
    total_points = len(df)
    
    if total_points > 0:
        if high_sat_points.sum() / total_points > 0.1:
            issues.append("执行器饱和 (高限) - 无法达到设定值")
            status = HealthStatus.WARNING
        elif is_at_max.sum() / total_points > 0.9 and avg_error > err_threshold:
            issues.append("执行器饱和 (高限) - 输出持续维持最大值")
            status = HealthStatus.WARNING
            
        if low_sat_points.sum() / total_points > 0.1:
            issues.append("执行器饱和 (低限) - 无法达到设定值")
            status = HealthStatus.WARNING
        elif is_at_min.sum() / total_points > 0.9 and avg_error < -err_threshold:
            issues.append("执行器饱和 (低限) - 输出持续维持最小值")
            status = HealthStatus.WARNING

    # 2. Noise Check
    pv = df['PV']
    sp = df['SP']
    
    smoothed_pv = pv.rolling(window=5, center=True).mean().fillna(pv)
    noise_signal = pv - smoothed_pv
    noise_std = noise_signal.std()
    
    sp_range_val = sp.max() - sp.min()
    if sp_range_val == 0:
        sp_range_val = sp.mean() * 0.1
        if sp_range_val == 0: sp_range_val = 1.0

    if (3 * noise_std) > (0.05 * sp_range_val):
        issues.append("高噪音水平检测")
        if status != HealthStatus.CRITICAL:
            status = HealthStatus.WARNING
            
    details['noise_std'] = noise_std

    # 3. Oscillation & Divergence
    # Analyze Error
    zero_crossings = np.where(np.diff(np.signbit(error)))[0]
    
    if len(zero_crossings) > 4:
        peaks = []
        for i in range(len(zero_crossings)-1):
            start = zero_crossings[i]
            end = zero_crossings[i+1]
            segment = error.iloc[start:end].abs()
            if len(segment) > 0:
                peaks.append(segment.max())
        
        if len(peaks) >= 3:
            x_idx = np.arange(len(peaks))
            slope, _ = np.polyfit(x_idx, peaks, 1)
            
            avg_peak = np.mean(peaks)
            
            if avg_peak > (0.02 * abs(sp_mean)) and slope > 0.05 * avg_peak:
                 issues.append("检测到发散震荡")
                 status = HealthStatus.CRITICAL
            elif avg_peak > (0.05 * abs(sp_mean)):
                 issues.append("检测到持续震荡")
                 if status != HealthStatus.CRITICAL:
                     status = HealthStatus.WARNING

    # 4. Steady State Error (Offset)
    # Check last 20% of data
    last_window_size = int(len(df) * 0.2)
    if last_window_size > 5:
        segment = df.iloc[-last_window_size:]
        # Ensure SP is relatively constant in this window
        if segment['SP'].std() < (0.01 * abs(sp_mean)):
            # Robust mean of error
            avg_segment_error = (segment['SP'] - segment['PV']).mean()
            # Threshold: 2% of SP
            if abs(avg_segment_error) > max(0.02 * abs(sp_mean), 1.0):
                 issues.append(f"存在稳态误差 (Offset): {avg_segment_error:.2f}")
                 if status != HealthStatus.CRITICAL:
                     status = HealthStatus.WARNING

    # 5. Severe Overshoot
    # Detect SP step changes
    sp_diff = df['SP'].diff().abs()
    # Step > 5% of SP Mean
    step_thresh = 0.05 * abs(sp_mean)
    step_indices = sp_diff[sp_diff > step_thresh].index
    
    for idx in step_indices:
        # Define window after step: e.g. 50 samples
        end_idx = min(idx + 50, len(df))
        if end_idx > idx + 5:
            window_sp = df['SP'].iloc[idx:end_idx]
            window_pv = df['PV'].iloc[idx:end_idx]
            target_sp = window_sp.iloc[-1] # Assuming step to a new level
            
            # Step size
            step_size = target_sp - df['SP'].iloc[idx-1]
            if abs(step_size) < 1e-3: continue

            # Max deviation from target
            if step_size > 0:
                max_pv = window_pv.max()
                overshoot = max_pv - target_sp
                if overshoot > 0.2 * step_size:
                    issues.append(f"检测到严重超调 (>{overshoot/step_size*100:.1f}%)")
                    if status != HealthStatus.CRITICAL:
                        status = HealthStatus.WARNING
                    break
            else:
                min_pv = window_pv.min()
                overshoot = target_sp - min_pv
                if overshoot > 0.2 * abs(step_size):
                    issues.append(f"检测到严重超调 (>{overshoot/abs(step_size)*100:.1f}%)")
                    if status != HealthStatus.CRITICAL:
                         status = HealthStatus.WARNING
                    break

    # 6. Valve Stiction / Stick-Slip
    window_size = 5
    if len(df) > window_size * 2:
        op_std = df['OP'].rolling(window=window_size).std()
        pv_std = df['PV'].rolling(window=window_size).std()
        op_moving = op_std > (0.005 * op_range) 
        pv_stuck = pv_std < (0.001 * sp_range_val)
        stiction_candidates = op_moving & pv_stuck
        if stiction_candidates.sum() > (0.05 * len(df)):
             issues.append("疑似阀门粘滞 (Stiction): 输出变化但PV响应迟滞")
             if status != HealthStatus.CRITICAL:
                 status = HealthStatus.WARNING

        return DiagnosisResult(

            status=status, 

            issues=issues, 

            details=details,

            saturation_mask=sat_mask,

            stiction_mask=stiction_candidates.fillna(False).values

        )

    