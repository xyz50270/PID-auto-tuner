import pandas as pd
import numpy as np
from src.modeling import FOPDTModel
from src.tuning import calculate_imc_pid, suggest_parameters, PIDParams
from src.evaluation import calculate_metrics

def test_iteration_workflow():
    # 1. Generate Baseline Data (Poor Control)
    # Process: K=2, Tau=10, Theta=1
    true_process = FOPDTModel(2, 10, 1, 0)
    
    # User uses Kp=0.1, Ti=100 (Very slow)
    initial_pid = PIDParams(0.1, 100, 0)
    
    # (Skip simulation generation, just assume we have metrics)
    # 2. Fit Model (Assume perfect fit for test)
    fitted_model = true_process
    
    # 3. Calculate Target
    target_pid = calculate_imc_pid(fitted_model, 'moderate')
    # Expected: Kp ~ 0.8, Ti ~ 10
    
    # 4. First Suggestion
    sugg1 = suggest_parameters(initial_pid, target_pid, max_change_percent=20.0)
    
    # Safe step: 0.1 -> 0.12 (20% increase)
    assert abs(sugg1.next_step_pid.Kp - 0.12) < 0.001
    
    # 5. User applies 0.12, runs process, gets Data 2.
    # Suppose process changed slightly? (Adaptive)
    # K=2.2 now.
    new_process = FOPDTModel(2.2, 10, 1, 0)
    new_target = calculate_imc_pid(new_process, 'moderate')
    # New Target Kp ~ 0.8 / 1.1 = 0.75
    
    # 6. Second Suggestion
    # Current PID is now 0.12
    sugg2 = suggest_parameters(sugg1.next_step_pid, new_target, max_change_percent=20.0)
    
    # Safe step: 0.12 -> 0.144
    assert abs(sugg2.next_step_pid.Kp - 0.144) < 0.001
    
    # Target should be updated
    assert abs(sugg2.target_pid.Kp - new_target.Kp) < 0.001
