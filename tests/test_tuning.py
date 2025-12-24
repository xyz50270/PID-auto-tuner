import pytest
from src.modeling import FOPDTModel
from src.tuning import calculate_imc_pid, suggest_parameters, PIDParams, TuningSuggestion

def test_imc_calculation():
    model = FOPDTModel(K=2.0, tau=10.0, theta=1.0, y0=0)
    pid = calculate_imc_pid(model, aggressiveness='moderate')
    assert abs(pid.Kp - 0.833) < 0.01
    assert abs(pid.Ti - 10.0) < 0.1

def test_progressive_tuning_structure():
    current = PIDParams(Kp=1.0, Ti=10.0, Td=0.0)
    target = PIDParams(Kp=2.0, Ti=5.0, Td=0.0) # Huge change
    
    # Max change 20%
    suggestion = suggest_parameters(current, target, max_change_percent=20.0)
    
    assert isinstance(suggestion, TuningSuggestion)
    assert abs(suggestion.next_step_pid.Kp - 1.2) < 0.001
    assert abs(suggestion.next_step_pid.Ti - 8.0) < 0.001 # 10 - 20% = 8
    
    # Check descriptions
    assert "限制步长" in suggestion.delta['Kp_desc']
    assert len(suggestion.warnings) >= 2 # Kp and Ti warnings