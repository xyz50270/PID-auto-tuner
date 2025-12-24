import pytest
import pandas as pd
import numpy as np
from src.analysis import analyze_controller_characteristics, check_data_sufficiency
from src.modeling import FOPDTModel

def test_controller_stats():
    # 1 min data, 1s interval
    t = pd.date_range(start='2023-01-01', periods=61, freq='s') 
    
    # Scenario: High OP movement (Chattering)
    # OP oscillates between 50 and 52 every step
    op = np.array([50 if i%2==0 else 52 for i in range(61)])
    pv = np.zeros(61) # Stable PV
    sp = np.zeros(61)
    
    df = pd.DataFrame({'Time': t, 'SP': sp, 'PV': pv, 'OP': op})
    
    stats = analyze_controller_characteristics(df)
    
    # Total Variation: Sum of abs diff. |52-50| + |50-52| ... = 2 * 60 = 120
    # Duration 1 min. So TV/min = 120.
    assert abs(stats.total_variation - 120.0) < 1.0
    
    # Sampling
    assert abs(stats.avg_sampling_time - 1.0) < 0.1

def test_sufficiency_check_with_model():
    # Process Tau=100s, Theta=10s. Settling ~ 410s.
    model = FOPDTModel(K=1, tau=100, theta=10, y0=0)
    
    # Data is only 200s
    t = pd.date_range(start='2023-01-01', periods=200, freq='s') 
    df = pd.DataFrame({
        'Time': t, 
        'SP': np.zeros(200), 
        'PV': np.zeros(200), 
        'OP': np.zeros(200)
    })
    
    check = check_data_sufficiency(df, model)
    assert check.is_sufficient is False
    assert "建议延长采集时间" in check.suggestions[0]
    
def test_sufficiency_check_uneven_time():
    # Data with gaps
    t1 = pd.date_range(start='2023-01-01 10:00:00', periods=10, freq='s')
    t2 = pd.date_range(start='2023-01-01 10:05:00', periods=10, freq='s') # 5 min gap
    t = t1.union(t2)
    
    df = pd.DataFrame({
        'Time': t, 
        'SP': np.zeros(20), 
        'PV': np.zeros(20), 
        'OP': np.zeros(20)
    })
    
    stats = analyze_controller_characteristics(df)
    # Max sampling time should be large (~300s)
    assert stats.max_sampling_time > 290.0
    assert stats.data_quality_score < 90 # Should be penalized
