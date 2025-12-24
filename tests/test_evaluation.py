import pytest
import pandas as pd
import numpy as np
from src.evaluation import calculate_metrics

def test_metrics_perfect_step():
    # Step at t=10 from 0 to 10. PV follows perfectly instantly (impossible but good for math)
    t = pd.date_range(start='2023-01-01', periods=20, freq='s')
    sp = np.zeros(20)
    sp[10:] = 10.0
    pv = sp.copy()
    
    df = pd.DataFrame({'Time': t, 'SP': sp, 'PV': pv})
    metrics = calculate_metrics(df)
    
    assert metrics.iae == 0.0
    assert metrics.overshoot == 0.0

def test_metrics_overshoot():
    # Step at t=0, SP=10. PV goes to 12 then 10.
    t = pd.date_range(start='2023-01-01', periods=10, freq='s')
    sp = np.ones(10) * 10.0
    # Create a step manually by diff logic requires a change in SP column
    # Let's make index 0 be 0, index 1 be 10
    sp[0] = 0.0 
    
    pv = np.array([0, 5, 8, 12, 11, 10, 10, 10, 10, 10])
    
    df = pd.DataFrame({'Time': t, 'SP': sp, 'PV': pv})
    metrics = calculate_metrics(df)
    
    # Step size 10. Max PV 12. Overshoot 2. % = 20%
    assert abs(metrics.overshoot - 20.0) < 1.0
    assert metrics.iae > 0
