from dataclasses import dataclass
from typing import Dict, Any
from src.modeling import FOPDTModel

@dataclass
class PIDParams:
    Kp: float
    Ti: float
    Td: float = 0.0

@dataclass
class TuningSuggestion:
    current_pid: PIDParams
    target_pid: PIDParams     # The theoretical best
    next_step_pid: PIDParams  # The safe step
    delta: Dict[str, Any]     # Description of changes
    warnings: list[str] = None

def calculate_imc_pid(model: FOPDTModel, aggressiveness: str = 'moderate') -> PIDParams:
    """
    Calculate PID parameters using SIMC rules.
    Aggressiveness: 'aggressive', 'moderate', 'conservative'
    """
    # SIMC Rules
    if aggressiveness == 'aggressive':
        lambda_c = max(0.1 * model.tau, model.theta)
    elif aggressiveness == 'moderate':
        lambda_c = max(0.5 * model.tau, 3 * model.theta)
    else: # conservative
        lambda_c = max(1.0 * model.tau, 10 * model.theta)
        
    if abs(model.K) < 1e-6:
        return PIDParams(0, 0, 0) # Cannot control
        
    Kc = (1.0 / model.K) * (model.tau / (lambda_c + model.theta))
    Ti = min(model.tau, 4 * (lambda_c + model.theta))
    
    return PIDParams(Kp=Kc, Ti=Ti, Td=0.0)

def suggest_parameters(
    current_pid: PIDParams, 
    recommended_pid: PIDParams, 
    max_change_percent: float = 20.0
) -> TuningSuggestion:
    """
    Progressively move current parameters towards recommended.
    Returns a detailed TuningSuggestion object.
    """
    warnings = []
    
    def step_value(curr, target, label):
        if curr == 0: 
            return target, f"0 -> {target:.4f} (初始设定)"
        
        diff = target - curr
        pct_change = (diff / curr) * 100.0
        max_step = abs(curr) * (max_change_percent / 100.0)
        
        if abs(diff) <= max_step:
            val = target
            desc = f"{curr:.4f} -> {val:.4f} (达到目标, {pct_change:+.1f}%)"
        else:
            step_sign = 1 if diff > 0 else -1
            val = curr + max_step * step_sign
            desc = f"{curr:.4f} -> {val:.4f} (限制步长, 目标 {target:.4f})"
            warnings.append(f"{label} 调整幅度受限 (理论需 {pct_change:+.1f}%)")
            
        return val, desc

    new_Kp, desc_Kp = step_value(current_pid.Kp, recommended_pid.Kp, "Kp")
    new_Ti, desc_Ti = step_value(current_pid.Ti, recommended_pid.Ti, "Ti")
    new_Td, desc_Td = step_value(current_pid.Td, recommended_pid.Td, "Td")
    
    next_pid = PIDParams(new_Kp, new_Ti, new_Td)
    
    delta_info = {
        "Kp_desc": desc_Kp,
        "Ti_desc": desc_Ti,
        "Td_desc": desc_Td
    }
    
    return TuningSuggestion(
        current_pid=current_pid,
        target_pid=recommended_pid,
        next_step_pid=next_pid,
        delta=delta_info,
        warnings=warnings
    )