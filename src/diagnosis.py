import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List

class HealthStatus(Enum):
    HEALTHY = "Healthy"
    WARNING = "Warning"
    CRITICAL = "Critical"

@dataclass
class DiagnosisResult:
    status: HealthStatus
    issues: List[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

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
            # Overshoot is when PV goes past Target in direction of step
            if step_size > 0:
                max_pv = window_pv.max()
                overshoot = max_pv - target_sp
                # Check if > 20% of step size
                if overshoot > 0.2 * step_size:
                    issues.append(f"检测到严重超调 (>{overshoot/step_size*100:.1f}%)")
                    if status != HealthStatus.CRITICAL:
                        status = HealthStatus.WARNING
                    break # Report once
            else:
                min_pv = window_pv.min()
                overshoot = target_sp - min_pv
                if overshoot > 0.2 * abs(step_size):
                    issues.append(f"检测到严重超调 (>{overshoot/abs(step_size)*100:.1f}%)")
                    if status != HealthStatus.CRITICAL:
                         status = HealthStatus.WARNING
                    break

    # 6. Valve Stiction / Stick-Slip
    # Heuristic: OP Variance is high, PV Variance is low (locally) -> then PV jumps?
    # Simpler: Look for segments where OP changes monotonically but PV is flat
    
    # We use rolling windows
    window_size = 5
    if len(df) > window_size * 2:
        op_std = df['OP'].rolling(window=window_size).std()
        pv_std = df['PV'].rolling(window=window_size).std()
        
        # Normalize criteria
        # OP is moving: std > 0.1% of range
        op_moving = op_std > (0.005 * op_range) 
        # PV is stuck: std < noise floor (0.1% of range)
        pv_stuck = pv_std < (0.001 * sp_range_val)
        
        stiction_candidates = op_moving & pv_stuck
        
        if stiction_candidates.sum() > (0.05 * len(df)): # If > 5% of time implies sticking
             issues.append("疑似阀门粘滞 (Stiction): 输出变化但PV响应迟滞")
             if status != HealthStatus.CRITICAL:
                 status = HealthStatus.WARNING

    return DiagnosisResult(status=status, issues=issues, details=details)
