import pytest
import numpy as np
from src.modeling import FOPDTModel
from src.tuning import PIDParams
from src.simulation import simulate_closed_loop

def test_simulation_step_response():
    # Model: K=1, Tau=10, Theta=0 (No delay for easy check)
    model = FOPDTModel(K=1.0, tau=10.0, theta=0.0, y0=0.0)
    
    # PID: Kp=1, Ti=10 (Matches process Tau)
    # Loop Gain K*Kp = 1. Ti = Tau.
    # This is a classic setup.
    pid = PIDParams(Kp=1.0, Ti=10.0, Td=0.0)
    
    t_span = np.linspace(0, 100, 101)
    
    def step_sp(t):
        return 10.0 if t >= 10 else 0.0
        
    res = simulate_closed_loop(model, pid, step_sp, t_span, op_limits=(-100, 100))
    
    pv = res['PV']
    sp = res['SP']
    
    # Check if PV settles to SP (Integrator ensures zero error)
    assert abs(pv[-1] - 10.0) < 0.1
    
    # Check dynamics: at t=10 (step), pv starts rising
    idx_step = 10
    assert pv[idx_step] < pv[idx_step+1]

def test_simulation_limit():
    # Check clamping
    model = FOPDTModel(K=10.0, tau=1.0, theta=0.0, y0=0.0)
    pid = PIDParams(Kp=100.0, Ti=0.1, Td=0.0) # Very aggressive
    
    t_span = np.linspace(0, 10, 11)
    res = simulate_closed_loop(model, pid, lambda t: 100, t_span, op_limits=(0, 50))
    
    assert np.all(res['OP'] <= 50.0)
