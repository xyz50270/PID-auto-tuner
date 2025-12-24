import pytest
import pandas as pd
import numpy as np
from src.diagnosis import analyze_loop_health, DiagnosisResult, HealthStatus

@pytest.fixture
def clean_data():
    t = pd.date_range(start='2023-01-01', periods=100, freq='s')
    return pd.DataFrame({
        'Time': t,
        'SP': 50.0,
        'PV': 50.0 + np.sin(np.linspace(0, 10, 100)) * 0.1, # Stable
        'OP': 25.0
    })

def test_healthy_loop(clean_data):
    result = analyze_loop_health(clean_data)
    assert result.status == HealthStatus.HEALTHY
    assert len(result.issues) == 0

def test_saturation_high():
    t = pd.date_range(start='2023-01-01', periods=100, freq='s')
    df = pd.DataFrame({
        'Time': t,
        'SP': 50.0,
        'PV': 40.0,
        'OP': 100.0 # Saturated High
    })
    result = analyze_loop_health(df)
    assert result.status == HealthStatus.WARNING
    assert any("执行器饱和 (高限)" in s for s in result.issues)

def test_severe_noise():
    t = pd.date_range(start='2023-01-01', periods=100, freq='s')
    # High noise
    noise = np.random.normal(0, 5, 100)
    df = pd.DataFrame({
        'Time': t,
        'SP': 50.0,
        'PV': 50.0 + noise,
        'OP': 25.0
    })
    result = analyze_loop_health(df)
    assert "高噪音水平检测" in result.issues

def test_oscillation_divergence():
    t = pd.date_range(start='2023-01-01', periods=200, freq='s')
    # Diverging oscillation: amplitude grows with time
    oscillation = np.sin(np.linspace(0, 20, 200)) * np.linspace(1, 10, 200)
    df = pd.DataFrame({
        'Time': t,
        'SP': 50.0,
        'PV': 50.0 + oscillation,
        'OP': 25.0
    })
    result = analyze_loop_health(df)
    assert result.status == HealthStatus.CRITICAL
    assert "检测到发散震荡" in result.issues