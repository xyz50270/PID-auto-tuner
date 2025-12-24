import pytest
import numpy as np
import pandas as pd
from src.modeling import FOPDTModel, fit_fopdt

def test_fopdt_prediction():
    # Test step response
    # K=2, tau=10, theta=2, y0=0
    # Step OP 0->1 at t=0
    t = np.linspace(0, 50, 51) # dt=1
    op = np.ones(51) 
    op[0] = 0 # Step from 0 to 1 happens essentially at start
    # Wait, my simulation logic uses op[0] as base.
    # If op jumps at k=1, then op_base=op[0].
    
    model = FOPDTModel(K=2.0, tau=10.0, theta=2.0, y0=0.0)
    
    # Create a step in OP
    op = np.zeros(51)
    op[5:] = 1.0 # Step at t=5
    
    pv = model.predict(op, t)
    
    # Check basic properties
    # 1. Delay: PV should stay 0 until t=5+2=7
    assert np.allclose(pv[:7], 0.0, atol=1e-3)
    
    # 2. Steady state: should approach K*(1-0) + y0 = 2
    assert abs(pv[-1] - 2.0) < 0.1 # at t=50 (4.5 tau after step), should be close
    
def test_fit_fopdt():
    # Generate clean data
    t = np.linspace(0, 100, 101)
    op = np.zeros(101)
    op[10:] = 5.0 # Step size 5
    
    real_model = FOPDTModel(K=1.5, tau=20.0, theta=5.0, y0=10.0)
    pv = real_model.predict(op, t)
    
    # Add timestamps
    time_index = pd.to_datetime('2023-01-01') + pd.to_timedelta(t, unit='s')
    df = pd.DataFrame({'Time': time_index, 'OP': op, 'PV': pv})
    
    # Fit
    fitted = fit_fopdt(df)
    
    # Tolerances
    assert abs(fitted.K - 1.5) < 0.1
    assert abs(fitted.tau - 20.0) < 2.0
    assert abs(fitted.theta - 5.0) < 1.0
    assert abs(fitted.y0 - 10.0) < 0.5
