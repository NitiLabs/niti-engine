import time
import os
import sys
import numpy as np

# Ensure we import local niti_core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from niti_core import run_simulation, VariableRequest, TraceConfig

def build_simple_situation(run_axes: bool = False):
    year = "2025"
    
    # Simple situation with regular income and dividend income
    people = {
        "you": {
            "age": { year: 45 },
            "employment_income": { year: 80000},
            "dividend_income": { year: 5000},
        },
        "spouse": {
            "age": { year: 45 },
            "employment_income": { year: 0 },
        }
    }
    
    members = ["you", "spouse"]
    
    tax_units = {
        "tax_unit": {
            "members": members,
            "filing_status": { year: "JOINT" }
        }
    }
    
    families = { "family": { "members": members } }
    
    households = {
        "household": {
            "members": members,
            "state_code": { year: "CA" }
        }
    }
    
    situation = {
        "people": people,
        "tax_units": tax_units,
        "families": families,
        "households": households
    }
    
    if run_axes:
        del situation["people"]["you"]["employment_income"]
        axis = { "name": "employment_income", "count": 171, "min": 30000, "max": 200000, "period": 2025 }
        situation["axes"] = [[axis]]
        
    return situation

def run_benchmark(mode="axes"):
    run_axes = (mode == "axes")
    situation_dict = build_simple_situation(run_axes=run_axes)
    
    start_time = time.time()
    
    variables = [
        VariableRequest("income_tax"),
        VariableRequest("ca_income_tax"),
    ]

    sim_result = run_simulation(
        situation_dict,
        variables,
        2025,
        trace_config=TraceConfig(enabled=False)
    )
    
    # Evaluate variables
    fed_tax = sim_result.arrays["income_tax"]
    ca_tax = sim_result.arrays["ca_income_tax"]
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"[{mode}] it took: {duration:.4f} seconds")
    
    if run_axes:
        print(f"Generated {len(fed_tax)} points on the axes.")

if __name__ == "__main__":
    # Perform a quick, untimed warmup to ensure lazy-loaded compilation is resolved before benchmarking
    try:
        from niti_core import Simulation
        _warmup_sim = Simulation(situation={"people": {"p": {"age": {"2025": 30}}}})
        _warmup_sim.calculate("income_tax", 2025)
    except Exception:
        pass

    print("--- Benchmarking Simple Standalone PolicyEngine (3 runs) ---")
    run_benchmark("taxes")
    run_benchmark("taxes")
    run_benchmark("taxes")
    
    print("\n--- Benchmarking Simple Standalone PolicyEngine Axes (3 runs) ---")
    run_benchmark("axes")
    run_benchmark("axes")
    run_benchmark("axes")
