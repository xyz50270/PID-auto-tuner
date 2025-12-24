import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    iae: float  # Integral of Absolute Error
    ise: float  # Integral of Squared Error
    overshoot: float # Percentage
    settling_time: float # Seconds (approx)

def calculate_metrics(df: pd.DataFrame) -> PerformanceMetrics:
    """
    Calculate control loop performance metrics.
    """
    # Ensure sorted by time
    df = df.sort_values('Time')
    
    # Calculate dt in seconds
    t_sec = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds().values
    dt = np.diff(t_sec)
    # Append last dt to match length
    dt = np.append(dt, dt[-1] if len(dt) > 0 else 1.0)
    
    error = df['SP'] - df['PV']
    abs_error = np.abs(error)
    squared_error = error ** 2
    
    # IAE = Sum(|e| * dt)
    iae = np.sum(abs_error * dt)
    
    # ISE = Sum(e^2 * dt)
    ise = np.sum(squared_error * dt)
    
    # Overshoot & Settling Time Analysis
    # This is complex for general data (might have multiple steps).
    # We analyze the largest step change found in SP.
    
    sp_diff = df['SP'].diff().abs()
    if sp_diff.max() > 0:
        step_idx = sp_diff.idxmax()
        step_time = df['Time'].iloc[step_idx]
        step_size = df['SP'].iloc[step_idx] - df['SP'].iloc[step_idx-1]
        target_sp = df['SP'].iloc[step_idx]
        
        # Analyze data AFTER the step
        post_step = df.iloc[step_idx:]
        if len(post_step) > 5:
            # Overshoot
            if step_size > 0:
                max_pv = post_step['PV'].max()
                overshoot_val = max(0, max_pv - target_sp)
            else:
                min_pv = post_step['PV'].min()
                overshoot_val = max(0, target_sp - min_pv)
            
            overshoot_pct = (overshoot_val / abs(step_size)) * 100.0 if abs(step_size) > 1e-6 else 0.0
            
            # Settling Time (Time to stay within 5% of target)
            band = 0.05 * abs(step_size)
            # Find last time PV was OUTSIDE the band
            outside_band = post_step[np.abs(post_step['PV'] - target_sp) > band]
            
            if not outside_band.empty:
                last_outside_time = outside_band['Time'].iloc[-1]
                settling_time = (last_outside_time - step_time).total_seconds()
            else:
                # Never went outside? Already settled?
                settling_time = 0.0
        else:
            overshoot_pct = 0.0
            settling_time = 0.0
    else:
        # No step detected, maybe steady state
        overshoot_pct = 0.0
        settling_time = 0.0
        
    return PerformanceMetrics(
        iae=iae,
        ise=ise,
        overshoot=overshoot_pct,
        settling_time=settling_time
    )
