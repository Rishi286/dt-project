"""
Synthetic Google Cluster Trace 2019 workload replay.
Distributions derived from published statistics in:
  Tirmazi et al. "Borg: the next generation." EuroSys 2020.
  https://dl.acm.org/doi/10.1145/3342195.3387517

Key properties matched:
  - CPU request: log-normal (μ=0.06, σ=0.2) of machine capacity, capped at 0.5
  - Memory request: log-normal (μ=0.02, σ=0.15) of machine capacity, capped at 0.5
  - Duration: Pareto-tailed (shape=0.7, scale=10s) — most tasks short, heavy tail
  - Scheduling class: 0=free(41%), 1=best(30%), 2=prod(19%), 3=monitoring(10%)
  - Arrival: Poisson process, rate varies with regime
"""

import numpy as np
from dataclasses import dataclass
from typing import List


# Machine capacity reference (GCT2019 standard cell ≈ 0.5 CPU, 0.5 GB per mcu)
# We scale to our simulation units: 1 CPU unit, 1024 MB
MACHINE_CPU  = 1.0     # normalised
MACHINE_MEM  = 1024.0  # MB

# Scheduling class distribution (Table 1 in Tirmazi et al.)
SCHED_CLASS_PROBS = [0.41, 0.30, 0.19, 0.10]   # class 0,1,2,3

# Priority mapping: higher class → higher priority
PRIORITY_MAP = {0: 0, 1: 1, 2: 2, 3: 3}

# Deadline slack by priority (same as synthetic sim)
SLACK_FACTOR = {0: 1.8, 1: 1.5, 2: 1.3, 3: 1.1}


@dataclass
class TraceTask:
    task_id: str
    arrival_time: float
    cpu_req: float      # CPU units
    mem_req: float      # MB
    duration: float     # seconds
    priority: int
    deadline: float


def _cpu_req(rng: np.random.Generator) -> float:
    """Log-normal CPU fraction of machine, converted to CPU units."""
    frac = rng.lognormal(mean=np.log(0.06), sigma=0.2)
    frac = np.clip(frac, 0.01, 0.50)
    return round(frac * 8.0, 2)   # scale to 0–4 CPU units (8-core reference)


def _mem_req(rng: np.random.Generator) -> float:
    """Log-normal memory fraction → MB."""
    frac = rng.lognormal(mean=np.log(0.02), sigma=0.15)
    frac = np.clip(frac, 0.005, 0.50)
    return round(frac * 4096, 0)  # scale to 0–2048 MB (4 GB reference)


def _duration(rng: np.random.Generator) -> float:
    """Pareto heavy-tail duration in seconds."""
    # Lomax (Pareto II): most tasks < 60s, tail up to hours
    d = rng.pareto(0.7) * 10 + 1.0
    return round(min(d, 3600.0), 2)


def generate_trace(
    regime: str,
    duration_sim: float = 3600.0,
    seed: int = 42,
) -> List[TraceTask]:
    """
    Generate a list of TraceTask objects for one simulation run.

    regime: 'steady' | 'diurnal' | 'bursty'
    duration_sim: total simulation window in seconds
    seed: RNG seed
    """
    rng = np.random.default_rng(seed)
    tasks: List[TraceTask] = []
    t = 0.0
    task_id = 0

    base_rate = 2.0   # tasks/second

    while t < duration_sim:
        # Arrival rate varies by regime
        if regime == "steady":
            rate = base_rate
        elif regime == "diurnal":
            # gentle sine wave, ±50 % around base
            rate = base_rate * (1.0 + 0.5 * np.sin(2 * np.pi * t / duration_sim))
            rate = max(rate, 0.1)
        else:  # bursty
            # 3× spike every 300 s for 30 s
            cycle = t % 300
            rate = base_rate * 3.0 if cycle < 30 else base_rate * 0.8

        inter_arrival = rng.exponential(1.0 / rate)
        t += inter_arrival
        if t >= duration_sim:
            break

        cpu  = _cpu_req(rng)
        mem  = _mem_req(rng)
        dur  = _duration(rng)
        prio = int(rng.choice(4, p=SCHED_CLASS_PROBS))
        slack = SLACK_FACTOR[prio]
        deadline = t + dur * slack

        tasks.append(TraceTask(
            task_id=f"gct_{task_id:06d}",
            arrival_time=round(t, 4),
            cpu_req=cpu,
            mem_req=mem,
            duration=dur,
            priority=prio,
            deadline=round(deadline, 4),
        ))
        task_id += 1

    return tasks


if __name__ == "__main__":
    for regime in ("steady", "diurnal", "bursty"):
        tasks = generate_trace(regime, seed=42)
        print(f"{regime:8s}: {len(tasks):5d} tasks, "
              f"avg cpu={np.mean([t.cpu_req for t in tasks]):.3f}, "
              f"avg mem={np.mean([t.mem_req for t in tasks]):.0f} MB")



