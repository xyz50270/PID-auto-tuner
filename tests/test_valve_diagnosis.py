import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.diagnosis import analyze_advanced_valve_health

def test_linear_valve():
    """Verify perfectly linear valve gets high score."""
    t = pd.date_range("2025-01-01", periods=100, freq="1s")
    op = np.linspace(10, 90, 100)
    pv = op * 2.0 # Perfect gain K=2
    df = pd.DataFrame({'Time': t, 'OP': op, 'PV': pv, 'SP': op})
    
    report = analyze_advanced_valve_health(df)
    assert report.linearity_score > 90
    assert not report.erosion_risk

def test_erosion_detection():
    """Verify high gain at low opening is detected as erosion."""
    t = pd.date_range("2025-01-01", periods=100, freq="1s")
    # Low opening range
    op_low = np.linspace(2, 10, 50)
    pv_low = op_low * 10.0 # Extreme gain at low opening (erosion symptom)
    # Mid opening range
    op_mid = np.linspace(40, 50, 50)
    pv_mid = op_mid * 2.0  # Normal gain
    
    df = pd.DataFrame({
        'Time': t,
        'OP': np.concatenate([op_low, op_mid]),
        'PV': np.concatenate([pv_low, pv_mid]),
        'SP': 50.0
    })
    
    report = analyze_advanced_valve_health(df)
    assert report.erosion_risk == True
    assert any("冲刷腐蚀" in s for s in report.suggestions)

def test_stiction_zone_mapping():
    """Verify specific OP range stiction is located."""
    t = pd.date_range("2025-01-01", periods=200, freq="1s")
    op = np.linspace(0, 100, 200)
    pv = op * 1.0
    
    # Simulate stiction at 40-50% range
    # In this range, PV stays flat while OP moves
    mask = (op >= 40) & (op < 50)
    pv[mask] = 40.0
    
    df = pd.DataFrame({'Time': t, 'OP': op, 'PV': pv, 'SP': 50.0})
    
    report = analyze_advanced_valve_health(df)
    assert len(report.stiction_zones) > 0
    # The zone should overlap with 40-50%
    assert any(z[0] >= 30 and z[1] <= 60 for z in report.stiction_zones)
    assert any("特定区间粘滞" in s for s in report.suggestions)
