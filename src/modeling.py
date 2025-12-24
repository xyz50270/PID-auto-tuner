import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, Optional
import pandas as pd

@dataclass
class FOPDTModel:
    K: float    # Process Gain
    tau: float  # Time Constant
    theta: float # Dead Time
    y0: float   # Bias

    def predict(self, op_array: np.ndarray, t_array: np.ndarray) -> np.ndarray:
        """
        Simulate FOPDT response for a given OP sequence with numerical stability.
        """
        dt = t_array[1] - t_array[0]
        n = len(t_array)
        pv_pred = np.zeros(n)
        pv_pred[0] = self.y0
        
        # Delay in steps
        delay_steps = int(max(0, self.theta) / dt)
        op_base = op_array[0]
        
        for k in range(1, n):
            op_idx = k - 1 - delay_steps
            op_val = op_array[op_idx] if op_idx >= 0 else op_base
            
            # Prediction using Euler method
            # tau * dy/dt = K * (u - u0) - (y - y0)
            
            # Ensure tau is not too small to prevent division by near-zero (infinite speed)
            safe_tau = max(self.tau, 0.1)
            
            driving_force = self.K * (op_val - op_base) - (pv_pred[k-1] - self.y0)
            
            # Check for non-finite values before calculation
            if not np.isfinite(driving_force):
                driving_force = 0.0
                
            change = (driving_force / safe_tau) * dt
            
            # Robustness: Clip change to prevent numerical explosion during optimization iterations
            change = np.clip(change, -1e5, 1e5)
            
            new_val = pv_pred[k-1] + change
            
            # Final sanity check for NaN/Inf
            pv_pred[k] = new_val if np.isfinite(new_val) else pv_pred[k-1]
            
        return pv_pred

def fit_fopdt(df: pd.DataFrame) -> FOPDTModel:
    """
    Fit FOPDT model to Time/OP/PV data with improved stability.
    """
    t = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds().values
    op = df['OP'].values
    pv = df['PV'].values
    
    dt = t[1] - t[0]
    if dt <= 0:
        raise ValueError("时间必须严格递增")
        
    # Initial guesses with sanity bounds
    delta_pv = pv[-1] - pv[0]
    delta_op = op[-1] - op[0]
    k_guess = delta_pv / delta_op if abs(delta_op) > 1e-3 else 1.0
    k_guess = np.clip(k_guess, -1e4, 1e4)
    
    y0_guess = pv[0]
    
    duration = t[-1] - t[0]
    tau_guess = max(duration / 5.0, 1.0)
    theta_guess = 1.0
    
    x0 = [k_guess, tau_guess, y0_guess]
    
    # K: unbounded, tau: min 0.1s, y0: unbounded
    bounds_no_theta = [(None, None), (0.1, None), (None, None)]
    
    best_mse = float('inf')
    best_params = [k_guess, tau_guess, theta_guess, y0_guess]
    
    # Grid scan for Dead Time (Theta) to handle non-convexity
    theta_candidates = np.linspace(0, duration * 0.4, 15)
    
    for theta_test in theta_candidates:
        def objective_fixed_theta(x):
            K_i, tau_i, y0_i = x
            model = FOPDTModel(K_i, tau_i, theta_test, y0_i)
            pv_pred = model.predict(op, t)
            
            error = pv - pv_pred
            # Clip error to avoid square overflow (max ~1e308 for float64)
            error = np.clip(error, -1e10, 1e10)
            mse = np.mean(error**2)
            return mse if np.isfinite(mse) else 1e30
        
        res = minimize(objective_fixed_theta, x0, bounds=bounds_no_theta, method='L-BFGS-B')
        
        if res.fun < best_mse:
            best_mse = res.fun
            best_params = [res.x[0], res.x[1], theta_test, res.x[2]]
            
    return FOPDTModel(*best_params)