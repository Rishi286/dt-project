"""
Experiment runner.
Runs 45 trials (5 seeds x 3 regimes) for each of 3 allocators.
Saves results to C:\dt-project\results\
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dt_controller'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines'))

import pandas as pd
import numpy as np
from simulation import Simulator, SimConfig
from dt_controller import DigitalTwinController
from baselines import ThresholdAllocator, GreedyBestFitAllocator

SEEDS   = [42, 123, 456, 789, 1011]
REGIMES = ["steady", "diurnal", "bursty"]

CLOUD_CPU = 100.0;  CLOUD_MEM = 204800.0
EDGE_CPU  = 20.0;   EDGE_MEM  = 20480.0

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_allocator(name):
    if name == "dt":
        return DigitalTwinController(CLOUD_CPU, CLOUD_MEM, EDGE_CPU, EDGE_MEM)
    elif name == "threshold":
        return ThresholdAllocator(CLOUD_CPU, CLOUD_MEM, EDGE_CPU, EDGE_MEM)
    elif name == "greedy":
        return GreedyBestFitAllocator(CLOUD_CPU, CLOUD_MEM, EDGE_CPU, EDGE_MEM)
    raise ValueError(name)


def summarise(df: pd.DataFrame, allocator: str, regime: str, seed: int) -> dict:
    completed = df[~df["rejected"]]
    total     = len(df)
    n_done    = len(completed)
    n_viol    = completed["sla_violated"].sum()

    return {
        "allocator":          allocator,
        "regime":             regime,
        "seed":               seed,
        "total_tasks":        total,
        "completed":          n_done,
        "rejected":           df["rejected"].sum(),
        "completion_rate":    n_done / total if total > 0 else 0,
        "sla_violations":     int(n_viol),
        "sla_violation_rate": n_viol / n_done if n_done > 0 else 0,
        "avg_response_time":  completed["response_time"].mean() if n_done > 0 else np.nan,
        "p95_response_time":  completed["response_time"].quantile(0.95) if n_done > 0 else np.nan,
        "cloud_tasks":        (completed["decision"] == "cloud").sum(),
        "edge_tasks":         (completed["decision"] == "edge").sum(),
    }


all_summaries = []
all_raw       = []

total_runs = len(["dt","threshold","greedy"]) * len(SEEDS) * len(REGIMES)
run_num    = 0

for allocator_name in ["dt", "threshold", "greedy"]:
    for regime in REGIMES:
        for seed in SEEDS:
            run_num += 1
            print(f"[{run_num}/{total_runs}] allocator={allocator_name}  regime={regime}  seed={seed}")

            cfg       = SimConfig(regime=regime, seed=seed,
                                  cloud_cpu=CLOUD_CPU, cloud_mem=CLOUD_MEM,
                                  edge_cpu=EDGE_CPU,   edge_mem=EDGE_MEM)
            allocator = make_allocator(allocator_name)
            sim       = Simulator(allocator, cfg)
            df        = sim.run()

            df["allocator"] = allocator_name
            df["regime"]    = regime
            df["seed"]      = seed
            all_raw.append(df)

            summary = summarise(df, allocator_name, regime, seed)
            all_summaries.append(summary)
            print(f"    tasks={summary['total_tasks']}  "
                  f"completion={summary['completion_rate']:.2%}  "
                  f"sla_viol={summary['sla_violation_rate']:.2%}  "
                  f"avg_rt={summary['avg_response_time']:.2f}s")

# Save
summary_df = pd.DataFrame(all_summaries)
raw_df     = pd.concat(all_raw, ignore_index=True)

summary_path = os.path.join(RESULTS_DIR, "summary.csv")
raw_path     = os.path.join(RESULTS_DIR, "raw_tasks.csv")

summary_df.to_csv(summary_path, index=False)
raw_df.to_csv(raw_path, index=False)

print(f"\nDone. Results saved to {RESULTS_DIR}")
print(summary_df.groupby("allocator")[["completion_rate","sla_violation_rate","avg_response_time"]].mean().round(4))
