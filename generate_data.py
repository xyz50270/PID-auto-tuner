import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.modeling import FOPDTModel
from src.tuning import PIDParams
from src.simulation import simulate_closed_loop

def generate_files():
    # 1. Define Process (The "Real World" Object)
    # K=2.0 (Gain), Tau=50s (Lag), Theta=10s (Delay)
    process_model = FOPDTModel(K=2.0, tau=50.0, theta=10.0, y0=50.0)

    # 2. Define Iteration Steps (PID Evolution)
    # Target (SIMC): Kp ~ 0.6, Ti ~ 50
    scenarios = [
        {"name": "0_initial", "pid": PIDParams(Kp=2.5, Ti=15.0, Td=0.0), "desc": "初始: 严重震荡 (过激)"},
        {"name": "1_iter",    "pid": PIDParams(Kp=2.0, Ti=20.0, Td=0.0), "desc": "调整1: 震荡减弱"},
        {"name": "2_iter",    "pid": PIDParams(Kp=1.5, Ti=30.0, Td=0.0), "desc": "调整2: 仍有超调"},
        {"name": "3_iter",    "pid": PIDParams(Kp=1.2, Ti=40.0, Td=0.0), "desc": "调整3: 轻微超调"},
        {"name": "4_iter",    "pid": PIDParams(Kp=0.9, Ti=45.0, Td=0.0), "desc": "调整4: 接近完美"},
        {"name": "5_iter",    "pid": PIDParams(Kp=0.65, Ti=50.0, Td=0.0), "desc": "调整5: 最优状态"},
    ]

    # 3. Simulation Settings
    duration = 600 # 10 minutes
    t_sim = np.linspace(0, duration, duration+1) # 1s resolution
    
    # Step SP from 50 to 60 at t=50
    def sp_func(t):
        return 60.0 if t >= 50 else 50.0

    output_dir = "test_data_suite"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating 6 datasets in '{output_dir}/'...")

    for sc in scenarios:
        # Simulate
        res = simulate_closed_loop(process_model, sc['pid'], sp_func, t_sim)
        
        # Add some noise to make it realistic
        noise_level = 0.1
        pv_noisy = res['PV'] + np.random.normal(0, noise_level, len(res['PV']))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Time': pd.to_datetime('2024-01-01 10:00:00') + pd.to_timedelta(res['Time'], unit='s'),
            'SP': res['SP'],
            'PV': pv_noisy,
            'OP': res['OP']
        })
        
        # Save
        filename = f"{output_dir}/{sc['name']}.csv"
        df.to_csv(filename, index=False)
        print(f"  - Created {filename}: {sc['desc']} (Kp={sc['pid'].Kp}, Ti={sc['pid'].Ti})")

if __name__ == "__main__":
    generate_files()
