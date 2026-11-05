import csv, random, math
from dataclasses import dataclass
from typing import List

ALIBABA_PATH = r"C:\dt-project\data\alibaba\batch_task.csv"
SLACK_FACTOR = {0: 4.0, 1: 3.0, 2: 2.5, 3: 2.0}

@dataclass
class TraceTask:
    task_id:      str
    arrival_time: float
    cpu_req:      float
    mem_req:      float
    duration:     float
    priority:     int
    deadline:     float

def load_alibaba_trace(regime: str = "steady", duration_sim: float = 3600.0, seed: int = 42) -> List[TraceTask]:
    rng = random.Random(seed)
    raw = []

    with open(ALIBABA_PATH, "r") as f:
        for row in csv.reader(f):
            if len(row) < 8:
                continue
            try:
                start  = float(row[2])
                end    = float(row[3])
                cpu    = float(row[4])
                status = row[5].strip()
                mem    = float(row[6])
            except (ValueError, IndexError):
                continue
            if status != "Terminated":
                continue
            if end <= start or cpu <= 0 or mem <= 0:
                continue
            raw.append((start, end, cpu, mem))

    if not raw:
        raise ValueError("No valid rows found")

    # Normalise times to start at 0, scale to fit simulation duration
    min_t = min(r[0] for r in raw)
    max_t = max(r[0] for r in raw) - min_t
    scale = duration_sim / max_t if max_t > 0 else 1.0

    tasks_raw = []
    for (start, end, cpu, mem) in raw:
        arrival  = (start - min_t) * scale
        duration = max(1.0, (end - start) * scale)
        tasks_raw.append((arrival, duration, cpu, mem))

    tasks_raw.sort(key=lambda x: x[0])

    result = []
    for i, (arrival, duration, cpu, mem) in enumerate(tasks_raw):
        if arrival > duration_sim:
            break

        if regime == "diurnal":
            factor = 0.5 + 0.5 * math.sin(2 * math.pi * arrival / duration_sim)
            arrival = min(arrival / (factor + 0.1), duration_sim - 1)
        elif regime == "bursty":
            if int(arrival) % 300 < 30:
                arrival = max(0, arrival - rng.uniform(0, 20))

        priority = rng.choices([0,1,2,3], weights=[41,30,19,10])[0]
        deadline = arrival + duration * SLACK_FACTOR[priority]

        result.append(TraceTask(
            task_id=f"ali_{i:06d}",
            arrival_time=round(arrival, 4),
            cpu_req=round(min(cpu / 100.0, 1.0), 4),
            mem_req=round(min(mem, 100.0), 4),
            duration=round(duration, 4),
            priority=priority,
            deadline=round(deadline, 4),
        ))

    return result

if __name__ == "__main__":
    for regime in ["steady", "diurnal", "bursty"]:
        tasks = load_alibaba_trace(regime, seed=42)
        print(f"{regime}: {len(tasks)} tasks, first={tasks[0].arrival_time:.1f}s, last={tasks[-1].arrival_time:.1f}s")
        print(f"  sample cpu={tasks[0].cpu_req} mem={tasks[0].mem_req} dur={tasks[0].duration:.1f}s deadline={tasks[0].deadline:.1f}s")
