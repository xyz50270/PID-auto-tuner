import pytest
import pandas as pd
import numpy as np
from src.diagnosis import analyze_loop_health, HealthStatus

def test_steady_state_error():
    # SP=50, PV settles at 45. Constant offset.
    t = pd.date_range(start='2023-01-01', periods=100, freq='s')
    df = pd.DataFrame({
        'Time': t,
        'SP': 50.0,
        'PV': 45.0, # Offset of 5
        'OP': 50.0
    })
    result = analyze_loop_health(df)
    # Should detect offset
    assert any("稳态误差" in s for s in result.issues)

def test_severe_overshoot():
    # Step change SP 50->60. PV goes to 80 then settles at 60.
    t = pd.date_range(start='2023-01-01', periods=100, freq='s')
    sp = np.zeros(100) + 50
    sp[10:] = 60
    
    pv = np.zeros(100) + 50
    # Overshoot logic: reach 60, then go up to 80, then back
    pv[10:20] = np.linspace(50, 80, 10)
    pv[20:40] = np.linspace(80, 60, 20)
    pv[40:] = 60
    
    df = pd.DataFrame({'Time': t, 'SP': sp, 'PV': pv, 'OP': 50})
    result = analyze_loop_health(df)
    assert any("严重超调" in s for s in result.issues)

def test_stiction_behavior():
    # OP integrating (ramping) while PV stuck, then PV jump
    t = pd.date_range(start='2023-01-01', periods=100, freq='s')
    
    op = np.linspace(50, 60, 100) # Ramping OP
    pv = np.zeros(100) + 50
    pv[50:] = 55 # PV stuck at 50 then jumps to 55
    
    df = pd.DataFrame({'Time': t, 'SP': 55, 'PV': pv, 'OP': op})
    
    # This simulates a sticky valve: Controller pushes OP, PV doesn't move, then snaps.
    result = analyze_loop_health(df)
    assert any("粘滞" in s for s in result.issues)
