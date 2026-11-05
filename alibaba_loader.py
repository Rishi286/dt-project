"""
alibaba_loader.py — Alibaba Cluster Trace 2018 workload loader.

Real batch_task.csv schema (8 columns, no header):
  0: start_time   seconds from trace epoch  (-1 if not started)
  1: end_time     seconds from trace epoch  (-1 if not finished)
  2: job_id
  3: task_id
  4: instance_num
  5: status       "Terminated" | "Waiting" | "Failed" | ...
  6: plan_cpu     fraction of one machine [0,1], typically 0.003–0.09
  7: plan_mem     fraction of one machine [0,1]

Trace covers ~60,000 s. We remap into sim_duration and apply
regime-specific arrival shaping (diurnal peaks, burst clusters).

Place the file at: data/alibaba/batch_task.csv
"""

import os
import csv
import numpy as np

# ── Trace-matched synthetic fallback stats ────────────────────────────────
# Derived from the real trace: median cpu~0.005, Weibull duration tail
_SYN_CPU_BUCKETS = [0.05, 0.1,  0.25, 0.5,  1.0,  2.0,  4.0]
_SYN_CPU_PROBS   = [0.35, 0.30, 0.15, 0.10, 0.05, 0.03, 0.02]


def _synthetic_alibaba_tasks(seed, regime, n, sim_duration):
    """Fallback: synthetic tasks whose stats mirror the Alibaba 2018 trace."""
    rng = np.random.default_rng(seed)

    if regime == "steady":
        arrivals = np.sort(rng.uniform(0, sim_duration * 0.9, n))
    elif regime == "diurnal":
        n_busy  = int(n * 0.70)
        n_quiet = n - n_busy
        p1 = rng.normal(sim_duration * 0.25, sim_duration * 0.06, n_busy // 2)
        p2 = rng.normal(sim_duration * 0.75, sim_duration * 0.06, n_busy - n_busy // 2)
        quiet = rng.uniform(0, sim_duration, n_quiet)
        arrivals = np.clip(np.sort(np.concatenate([p1, p2, quiet])), 0, sim_duration * 0.95)
    else:  # bursty
        n_burst = int(n * 0.65)
        centers = rng.uniform(sim_duration * 0.1, sim_duration * 0.9, 3)
        per_b   = n_burst // 3
        bursts  = []
        for c in centers:
            bursts.extend(rng.normal(c, sim_duration * 0.03, per_b))
        rest = list(rng.uniform(0, sim_duration * 0.95, n - len(bursts)))
        arrivals = np.clip(np.sort(np.array(bursts + rest)), 0, sim_duration * 0.95)

    cpu_arr  = rng.choice(_SYN_CPU_BUCKETS, size=n, p=_SYN_CPU_PROBS)
    mem_arr  = cpu_arr * rng.uniform(1.0, 3.0, n)
    dur_arr  = np.clip(rng.weibull(0.6, n) * 28.0, 2.0, 300.0)
    pri_arr  = rng.choice([1, 2, 3], size=n, p=[0.35, 0.25, 0.40])
    slack    = np.where(pri_arr <= 2, rng.uniform(1.5, 3.0, n), rng.uniform(3.0, 8.0, n))
    deadlines = arrivals + dur_arr * slack

    return [
        {
            "task_id":      f"syn_{i}",
            "cpu_req":      float(cpu_arr[i]),
            "mem_req":      float(mem_arr[i]),
            "duration":     float(dur_arr[i]),
            "deadline":     float(deadlines[i]),
            "priority":     int(pri_arr[i]),
            "arrival_time": float(arrivals[i]),
            "source":       "synthetic_alibaba",
        }
        for i in range(n)
    ]


def load_alibaba_tasks(seed, regime, n, sim_duration,
                       trace_path="data/alibaba/batch_task.csv"):
    """
    Load n tasks from the Alibaba 2018 batch_task trace, shaped by regime.
    Falls back to synthetic if the file is missing.
    """
    rng = np.random.default_rng(seed)

    if not os.path.exists(trace_path):
        print(f"[alibaba] {trace_path!r} not found — using synthetic fallback.")
        return _synthetic_alibaba_tasks(seed, regime, n, sim_duration)

    print(f"[alibaba] Loading real trace from {trace_path!r} …")

    raw = []
    with open(trace_path, newline="", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            if parts[5].strip() != "Terminated":
                continue
            try:
                start   = float(parts[0])
                end     = float(parts[1])
                cpu_frac = float(parts[7])
                dur     = end - start
                if start < 0 or dur <= 0 or cpu_frac <= 0:
                    continue
                raw.append((start, dur, cpu_frac))
            except ValueError:
                continue

    if not raw:
        print("[alibaba] No usable rows — using synthetic fallback.")
        return _synthetic_alibaba_tasks(seed, regime, n, sim_duration)

    print(f"[alibaba] Parsed {len(raw):,} Terminated tasks.")

    raw.sort(key=lambda x: x[0])
    t_min = raw[0][0]
    t_max = raw[-1][0]
    trace_span = t_max - t_min if t_max > t_min else 1.0

    # ── Scale CPU to simulation units ─────────────────────────────────────
    # plan_cpu is fraction of one ~96-core machine in the trace.
    # We map to our per-task cpu_req units (1 unit ≈ 1 sim core).
    # Trace max cpu_frac ~0.09 → ~8.6 cores on 96-core machine.
    # We cap at 4 sim-cores to match existing baseline assumptions.
    MACHINE_CORES = 96.0

    def scale_cpu(cpu_frac):
        cores = cpu_frac * MACHINE_CORES
        return float(np.clip(cores, 0.05, 4.0))

    def scale_mem(cpu_frac):
        # mem not always present; estimate as 2× cpu cores in GB
        return scale_cpu(cpu_frac) * 2.0

    # ── Remap arrivals into sim window per regime ─────────────────────────
    arrivals_norm = [(r[0] - t_min) / trace_span for r in raw]  # [0,1]

    if regime == "steady":
        arrivals_sim = [a * sim_duration * 0.9 for a in arrivals_norm]

    elif regime == "diurnal":
        # Compress high-cpu tasks (online jobs) into two peaks;
        # low-cpu (batch) tasks fill the quiet period
        arr_sim = []
        for i, (_, _, cpu_f) in enumerate(raw):
            a = arrivals_norm[i]
            if cpu_f >= 0.01:   # "online" tier
                if a < 0.5:
                    arr_sim.append(rng.uniform(sim_duration * 0.20, sim_duration * 0.45))
                else:
                    arr_sim.append(rng.uniform(sim_duration * 0.65, sim_duration * 0.90))
            else:               # "batch" tier — spread across quiet window
                arr_sim.append(rng.uniform(0, sim_duration * 0.95))
        arrivals_sim = arr_sim

    else:  # bursty
        centers = rng.uniform(sim_duration * 0.1, sim_duration * 0.9, 3)
        arr_sim = []
        for i in range(len(raw)):
            bucket = i % 3
            if arrivals_norm[i] < 0.65:   # burst portion
                arr_sim.append(float(np.clip(
                    rng.normal(centers[bucket], sim_duration * 0.03),
                    0, sim_duration * 0.95
                )))
            else:
                arr_sim.append(rng.uniform(0, sim_duration * 0.95))
        arrivals_sim = arr_sim

    arrivals_sim = np.array(arrivals_sim)

    # ── Build task dicts ──────────────────────────────────────────────────
    tasks = []
    for i, (_, dur, cpu_f) in enumerate(raw):
        arr   = float(arrivals_sim[i])
        cpu   = scale_cpu(cpu_f)
        mem   = scale_mem(cpu_f)
        dur_c = float(np.clip(dur, 1.0, 300.0))
        pri   = 1 if cpu >= 1.0 else (2 if cpu >= 0.25 else 3)
        slack = rng.uniform(1.5, 3.0) if pri <= 2 else rng.uniform(3.0, 8.0)
        tasks.append({
            "task_id":      f"ali_{i}",
            "cpu_req":      round(cpu, 3),
            "mem_req":      round(mem, 3),
            "duration":     round(dur_c, 2),
            "deadline":     round(arr + dur_c * slack, 2),
            "priority":     pri,
            "arrival_time": round(arr, 3),
            "source":       "alibaba_trace",
        })

    tasks.sort(key=lambda x: x["arrival_time"])

    # ── Resample to exactly n tasks ───────────────────────────────────────
    if len(tasks) >= n:
        idx   = np.linspace(0, len(tasks) - 1, n, dtype=int)
        tasks = [tasks[i] for i in idx]
    else:
        # Tile with small jitter if trace is smaller than n
        while len(tasks) < n:
            extra = dict(tasks[len(tasks) % len(raw)])
            extra["arrival_time"] = float(np.clip(
                extra["arrival_time"] + rng.uniform(-60, 60), 0, sim_duration * 0.95
            ))
            extra["deadline"] = extra["arrival_time"] + extra["duration"] * rng.uniform(1.5, 5.0)
            extra["task_id"]  = extra["task_id"] + "_x"
            tasks.append(extra)
        tasks.sort(key=lambda x: x["arrival_time"])

    print(f"[alibaba] Returning {len(tasks)} tasks for regime={regime!r}, seed={seed}.")
    return tasks
