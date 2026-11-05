"""
run_experiments_trace.py
Runs 45 trials (5 seeds × 3 regimes × 3 allocators) using GCT2019-derived trace.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv, time
import numpy as np
import simpy
from simulation.trace_generator import generate_trace
from dt_controller.dt_controller import DigitalTwinController, Task
from baselines.baselines import ThresholdAllocator, GreedyBestFitAllocator

CLOUD_CPU, CLOUD_MEM = 100.0, 204800.0
EDGE_CPU,  EDGE_MEM  =  20.0,  20480.0
SIM_DURATION = 3600.0
SEEDS   = [42, 7, 13, 99, 2025]
REGIMES = ["steady", "diurnal", "bursty"]

os.makedirs("results", exist_ok=True)

def make_allocator(name):
    if name == "DT":
        return DigitalTwinController(CLOUD_CPU, CLOUD_MEM, EDGE_CPU, EDGE_MEM)
    elif name == "Threshold":
        return ThresholdAllocator(CLOUD_CPU, CLOUD_MEM, EDGE_CPU, EDGE_MEM)
    else:
        return GreedyBestFitAllocator(CLOUD_CPU, CLOUD_MEM, EDGE_CPU, EDGE_MEM)

def run_trial(allocator_name, regime, seed):
    alloc = make_allocator(allocator_name)
    trace = generate_trace(regime, duration_sim=SIM_DURATION, seed=seed)
    env   = simpy.Environment()

    completed, rejected, sla_violations = [], [], [0]
    total_rt = [0.0]

    def process_task(tt):
        task = Task(
            task_id=tt.task_id,
            cpu_req=tt.cpu_req,
            mem_req=tt.mem_req,
            duration=tt.duration,
            deadline=tt.deadline,
            priority=tt.priority,
            arrival_time=tt.arrival_time,
        )
        decision = alloc.allocate(task, env.now)
        if decision == "reject":
            rejected.append(task.task_id)
            return

        # Edge is faster (0.6×), cloud is 1.0×
        factor = 0.6 if decision == "edge" else 1.0
        yield env.timeout(task.duration * factor)

        rt = env.now - task.arrival_time
        violated = env.now > task.deadline
        if violated:
            sla_violations[0] += 1
        total_rt[0] += rt
        completed.append(rt)
        alloc.release(decision, task, rt, violated)

    def arrival_loop():
        for tt in trace:
            yield env.timeout(max(0, tt.arrival_time - env.now))
            env.process(process_task(tt))

    env.process(arrival_loop())
    env.run(until=SIM_DURATION)

    n_total = len(trace)
    n_comp  = len(completed)
    n_rej   = len(rejected)
    avg_rt  = float(np.mean(completed)) if completed else 0.0
    comp_rate = n_comp / n_total if n_total > 0 else 0.0
    sla_rate  = sla_violations[0] / n_comp if n_comp > 0 else 0.0

    return {
        "allocator": allocator_name,
        "regime": regime,
        "seed": seed,
        "n_tasks": n_total,
        "n_completed": n_comp,
        "n_rejected": n_rej,
        "avg_response_time": round(avg_rt, 4),
        "completion_rate": round(comp_rate, 4),
        "sla_violation_rate": round(sla_rate, 4),
    }

rows = []
total = len(SEEDS) * len(REGIMES) * 3
done  = 0
t0    = time.time()

for seed in SEEDS:
    for regime in REGIMES:
        for alloc_name in ["DT", "Threshold", "Greedy"]:
            row = run_trial(alloc_name, regime, seed)
            rows.append(row)
            done += 1
            print(f"[{done:2d}/{total}] {alloc_name:10s} | {regime:8s} | seed={seed} | "
                  f"rt={row['avg_response_time']:7.3f}s  comp={row['completion_rate']:.3f}  "
                  f"sla_viol={row['sla_violation_rate']:.3f}")

with open("results/summary_trace.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone in {time.time()-t0:.1f}s — results/summary_trace.csv")
