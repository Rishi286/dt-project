import simpy
import numpy as np
import csv
import os
import sys
sys.path.insert(0, ".")

from dt_controller.dt_controller import DigitalTwinController, Task
from dt_controller.adaptive_dt_controller import AdaptiveDTController
from baselines.baselines import ThresholdAllocator, GreedyBestFitAllocator
from alibaba_loader import load_alibaba_tasks

SEEDS        = [42, 7, 13, 99, 2025]
REGIMES      = ["steady", "diurnal", "bursty"]
ALLOCATORS   = ["DT", "AdaptiveDT", "Threshold", "Greedy"]
SIM_DURATION = 3600.0
CLOUD_CORES  = 100
EDGE_CORES   = 20
N_TASKS      = 5000

TRACE_PATH   = "data/alibaba/batch_task.csv"


def run_simulation(alloc_name, regime, seed):
    env = simpy.Environment()
    cloud_res = simpy.Resource(env, capacity=CLOUD_CORES)
    edge_res  = simpy.Resource(env, capacity=EDGE_CORES)

    if alloc_name == "DT":
        alloc = DigitalTwinController(
            cloud_cpu=CLOUD_CORES, cloud_mem=CLOUD_CORES * 4,
            edge_cpu=EDGE_CORES,   edge_mem=EDGE_CORES  * 4,
        )
    elif alloc_name == "AdaptiveDT":
        alloc = AdaptiveDTController(
            cloud_cpu=CLOUD_CORES, cloud_mem=CLOUD_CORES * 4,
            edge_cpu=EDGE_CORES,   edge_mem=EDGE_CORES  * 4,
        )
    elif alloc_name == "Threshold":
        alloc = ThresholdAllocator(
            cloud_cpu=CLOUD_CORES, cloud_mem=CLOUD_CORES * 4,
            edge_cpu=EDGE_CORES,   edge_mem=EDGE_CORES  * 4,
        )
    else:
        alloc = GreedyBestFitAllocator(
            cloud_cpu=CLOUD_CORES, cloud_mem=CLOUD_CORES * 4,
            edge_cpu=EDGE_CORES,   edge_mem=EDGE_CORES  * 4,
        )

    # Load tasks from Alibaba trace (falls back to synthetic if not found)
    task_dicts = load_alibaba_tasks(
        seed=seed, regime=regime, n=N_TASKS,
        sim_duration=SIM_DURATION, trace_path=TRACE_PATH,
    )

    completed = []
    rejected  = []

    # pending[tier] = number of tasks accepted but not yet finished
    pending = {"cloud": 0, "edge": 0}
    if hasattr(alloc, "set_pending"):
        alloc.set_pending(pending)

    def process_task(td):
        task = Task(
            task_id=td["task_id"], cpu_req=td["cpu_req"], mem_req=td["mem_req"],
            duration=td["duration"], deadline=td["deadline"],
            priority=td["priority"], arrival_time=td["arrival_time"],
        )
        decision = alloc.allocate(task, env.now)
        if decision == "reject":
            rejected.append(task)
            return

        res = cloud_res if decision == "cloud" else edge_res
        pending[decision] += 1          # accepted: backlog grows
        with res.request() as req:
            yield req                   # wait for a SimPy slot
            start = env.now
            yield env.timeout(task.duration)
            finish = env.now
            pending[decision] -= 1      # completed: backlog shrinks (ONLY HERE)
            rt = finish - task.arrival_time
            violated = finish > task.deadline
            if hasattr(alloc, "release"):
                alloc.release(decision, task, rt, violated)
            completed.append({
                "task_id": task.task_id, "decision": decision,
                "response_time": rt, "violated": violated,
            })

    def arrival_loop():
        for idx, td in enumerate(task_dicts):
            env.process(process_task(td))
            if idx + 1 < len(task_dicts):
                gap = task_dicts[idx + 1]["arrival_time"] - td["arrival_time"]
                if gap > 0:
                    yield env.timeout(gap)

    env.process(arrival_loop())
    env.run(until=SIM_DURATION + 300)

    n_comp   = len(completed)
    n_viol   = sum(1 for c in completed if c["violated"])
    avg_rt   = np.mean([c["response_time"] for c in completed]) if completed else 0
    sla_rate = n_viol / n_comp if n_comp > 0 else 0
    n_adapt  = len(alloc.weight_history) if hasattr(alloc, "weight_history") else 0

    return {
        "allocator": alloc_name, "regime": regime, "seed": seed,
        "avg_rt": round(avg_rt, 3), "comp_rate": round(n_comp / N_TASKS, 4),
        "sla_rate": round(sla_rate, 4), "n_adapt": n_adapt,
        "n_completed": n_comp, "n_rejected": len(rejected), "n_tasks": N_TASKS,
        "workload": "alibaba_trace" if os.path.exists(TRACE_PATH) else "synthetic",
    }


os.makedirs("results", exist_ok=True)
rows  = []
total = len(SEEDS) * len(REGIMES) * len(ALLOCATORS)
i     = 0
for seed in SEEDS:
    for regime in REGIMES:
        for alloc_name in ALLOCATORS:
            i += 1
            r = run_simulation(alloc_name, regime, seed)
            rows.append(r)
            print(f"[{i:2d}/{total}] {alloc_name:<12} | {regime:<7} | seed={seed} | "
                  f"rt={r['avg_rt']}s  comp={r['comp_rate']}  "
                  f"sla={r['sla_rate']}  adapt={r['n_adapt']}")

with open("results/summary_adaptive.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nDone -- results/summary_adaptive.csv")
