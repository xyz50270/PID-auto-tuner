import numpy as np
from src.modeling import FOPDTModel
from src.tuning import PIDParams

def simulate_closed_loop(
    model: FOPDTModel,
    pid: PIDParams,
    setpoint_func: callable,
    t_span: np.ndarray,
    op_limits: tuple = (0, 100)
) -> dict:
    """
    Simulate closed-loop PID control.
    Returns dictionary with time, sp, pv, op arrays.
    """
    dt = t_span[1] - t_span[0]
    n = len(t_span)
    
    pv = np.zeros(n)
    op = np.zeros(n)
    sp = np.zeros(n)
    
    pv[0] = model.y0
    op[0] = 0 # Assume starting at 0 output or need bias? 
    # Usually steady state: OP = (y0 - y0) / K = 0 deviation.
    # Let's assume deviation variables.
    
    integral = 0.0
    prev_error = 0.0
    
    # Delay buffer for model
    delay_steps = int(max(0, model.theta) / dt)
    op_buffer = np.zeros(n + delay_steps) # Buffer to store past OPs
    
    for k in range(n):
        t = t_span[k]
        sp_val = setpoint_func(t)
        sp[k] = sp_val
        
        # 1. Read PV (from Process Model)
        # PV[k] based on PAST OP
        if k > 0:
            # Model dynamics:
            # PV[k] = PV[k-1] + (dt/tau) * ( K * OP[k-1-d] - (PV[k-1]-y0) )
            
            # Retrieve delayed OP
            delayed_idx = k - 1 - delay_steps
            if delayed_idx < 0:
                op_delayed = 0 # Initial condition (deviation)
            else:
                op_delayed = op[delayed_idx]
            
            driving_force = model.K * op_delayed - (pv[k-1] - model.y0)
            pv[k] = pv[k-1] + (driving_force / model.tau) * dt
        else:
            pv[k] = model.y0

        # 2. Calculate PID Output
        error = sp[k] - pv[k]
        
        # P
        P = pid.Kp * error
        
        # I
        if pid.Ti > 0:
            integral += (pid.Kp * dt / pid.Ti) * error
        
        # D (Simple backward difference)
        D = 0
        if pid.Td > 0 and k > 0:
            D = (pid.Kp * pid.Td / dt) * (error - prev_error)
            
        raw_op = P + integral + D
        
        # Clamp OP
        clamped_op = max(op_limits[0], min(op_limits[1], raw_op))
        
        # Anti-windup (Clamping)
        # If clamped, stop integrating in that direction?
        # Simple back-calculation or conditional integration is better, but here just clamping integral implies:
        if pid.Ti > 0 and (clamped_op != raw_op):
            # Back-calculate integral to be consistent with clamped OP
            # clamped_op = P + I_new + D
            integral = clamped_op - P - D
            
        op[k] = clamped_op
        prev_error = error
        
    return {'Time': t_span, 'SP': sp, 'PV': pv, 'OP': op}
