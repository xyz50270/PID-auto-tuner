from dataclasses import dataclass
from typing import Dict, Any
from src.modeling import FOPDTModel

@dataclass
class PIDParams:
    Kp: float
    Ti: float
    Td: float = 0.0

    @property
    def PB(self) -> float:
        """Calculate Proportional Band (PB). PB = 100 / Kp."""
        return 100.0 / self.Kp if abs(self.Kp) > 1e-9 else 9999.9

    @staticmethod
    def from_pb(pb: float, ti: float, td: float = 0.0) -> 'PIDParams':
        """Create PIDParams from Proportional Band."""
        kp = 100.0 / pb if abs(pb) > 1e-9 else 0.0
        return PIDParams(Kp=kp, Ti=ti, Td=td)

@dataclass
class TuningSuggestion:
    current_pid: PIDParams
    target_pid: PIDParams     # The theoretical best
    next_step_pid: PIDParams  # The safe step
    warnings: list[str] = None
    
    def get_delta_desc(self, label: str, mode: str = "Kp") -> str:
        """Dynamically generate description based on current mode."""
        is_pb = (mode == "PB" and label == "Kp")
        
        c_val = self.current_pid.PB if is_pb else getattr(self.current_pid, label)
        n_val = self.next_step_pid.PB if is_pb else getattr(self.next_step_pid, label)
        t_val = self.target_pid.PB if is_pb else getattr(self.target_pid, label)
        
        if c_val == 0 or (is_pb and c_val >= 9999.9):
            return f"0 -> {n_val:.4f} (初始设定)"
            
        diff = n_val - c_val
        pct_change = (diff / c_val) * 100.0 if abs(c_val) > 1e-9 else 0.0
        
        if abs(n_val - t_val) < 1e-6:
            return f"{c_val:.4f} -> {n_val:.4f} (达到目标, {pct_change:+.1f}%)"
        else:
            return f"{c_val:.4f} -> {n_val:.4f} (限制步长, 目标 {t_val:.4f})"

def calculate_imc_pid(model: FOPDTModel, aggressiveness: str = 'moderate') -> PIDParams:
    # ... (existing calculate_imc_pid code remains same)
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

    

    def calculate_next_value(curr, target, label):

        if curr == 0: 

            return target

        

        diff = target - curr

        max_step = abs(curr) * (max_change_percent / 100.0)

        

        if abs(diff) <= max_step:

            return target

        else:

            step_sign = 1 if diff > 0 else -1

            warnings.append(f"{label} 调整幅度受限 (理论需 {(diff/curr)*100.0:+.1f}%)")

            return curr + max_step * step_sign



    new_Kp = calculate_next_value(current_pid.Kp, recommended_pid.Kp, "Kp")

    new_Ti = calculate_next_value(current_pid.Ti, recommended_pid.Ti, "Ti")

    new_Td = calculate_next_value(current_pid.Td, recommended_pid.Td, "Td")

    

    return TuningSuggestion(

        current_pid=current_pid,

        target_pid=recommended_pid,

        next_step_pid=PIDParams(new_Kp, new_Ti, new_Td),

        warnings=warnings

    )
