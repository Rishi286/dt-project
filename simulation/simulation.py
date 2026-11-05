"""
SimPy-based cloud-edge simulation engine.
Drives task arrivals, dispatches to allocator, records metrics.
"""

import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SimConfig:
    # Tier capacities
    cloud_cpu: float = 100.0
    cloud_mem: float = 204800.0   # MB  (200 GB)
    edge_cpu:  float = 20.0
    edge_mem:  float = 20480.0    # MB  (20 GB)

    # Workload regime: "steady" | "diurnal" | "bursty"
    regime: str = "steady"

    # Simulation duration (simulated seconds)
    sim_duration: float = 3600.0

    # Random seed
    seed: int = 42

    # Base task arrival rate (tasks/second)
    base_arrival_rate: float = 5.0

    # Edge processing is faster (lower latency multiplier)
    edge_latency_factor: float = 0.6
    cloud_latency_factor: float = 1.0


@dataclass
class TaskRecord:
    task_id: str
    arrival_time: float
    cpu_req: float
    mem_req: float
    duration: float
    deadline: float
    priority: int
    decision: str = ""
    start_time: float = 0.0
    finish_time: float = 0.0
    response_time: float = 0.0
    sla_violated: bool = False
    rejected: bool = False


class Simulator:

    def __init__(self, allocator, config: SimConfig):
        self.allocator = allocator
        self.cfg       = config
        self.records: list[TaskRecord] = []
        self.env       = simpy.Environment()
        self.rng       = np.random.default_rng(config.seed)

        # Shared resources (capacity = number of concurrent tasks)
        self.cloud_res = simpy.Resource(self.env, capacity=50)
        self.edge_res  = simpy.Resource(self.env, capacity=10)

    # ------------------------------------------------------------------
    # Workload generation
    # ------------------------------------------------------------------

    def _arrival_rate(self, t: float) -> float:
        r = self.cfg.base_arrival_rate
        if self.cfg.regime == "steady":
            return r
        elif self.cfg.regime == "diurnal":
            # Sine wave: peak at t=1800, trough at t=0/3600
            return r * (0.5 + 0.5 * np.sin(2 * np.pi * t / self.cfg.sim_duration - np.pi / 2) + 0.5)
        elif self.cfg.regime == "bursty":
            # Occasional 3x spikes every ~300s
            if int(t) % 300 < 30:
                return r * 3.0
            return r
        return r

    def _make_task(self, task_id: int, t: float):
        from dt_controller import Task  # local import to avoid circular
        cpu  = float(self.rng.uniform(0.5, 4.0))
        mem  = float(self.rng.uniform(256, 2048))
        dur  = float(self.rng.exponential(10.0))      # mean 10s
        prio = int(self.rng.integers(0, 4))           # 0-3
        # Deadline tighter for high-priority tasks
        slack_factor = {0: 5.0, 1: 3.0, 2: 2.0, 3: 1.5}[prio]
        deadline = t + dur * slack_factor

        task_obj = Task(
            task_id=str(task_id),
            cpu_req=cpu, mem_req=mem,
            duration=dur, deadline=deadline,
            priority=prio, arrival_time=t,
        )
        record = TaskRecord(
            task_id=str(task_id), arrival_time=t,
            cpu_req=cpu, mem_req=mem,
            duration=dur, deadline=deadline, priority=prio,
        )
        return task_obj, record

    # ------------------------------------------------------------------
    # SimPy processes
    # ------------------------------------------------------------------

    def _task_process(self, task_obj, record: TaskRecord):
        t = self.env.now
        decision = self.allocator.allocate(task_obj, t)

        if decision == "reject":
            record.rejected = True
            record.decision = "reject"
            self.records.append(record)
            return

        record.decision    = decision
        record.start_time  = t
        resource           = self.cloud_res if decision == "cloud" else self.edge_res
        latency_factor     = (self.cfg.cloud_latency_factor if decision == "cloud"
                              else self.cfg.edge_latency_factor)

        with resource.request() as req:
            yield req
            actual_duration = task_obj.duration * latency_factor
            yield self.env.timeout(actual_duration)
            finish = self.env.now
            rt     = finish - record.arrival_time
            viol   = finish > task_obj.deadline

            record.finish_time   = finish
            record.response_time = rt
            record.sla_violated  = viol
            self.records.append(record)

            # Notify allocator to free resources
            self.allocator.release(decision, task_obj, rt, viol)

    def _arrival_process(self):
        task_id = 0
        while True:
            rate = self._arrival_rate(self.env.now)
            iat  = self.rng.exponential(1.0 / rate)  # inter-arrival time
            yield self.env.timeout(iat)
            task_obj, record = self._make_task(task_id, self.env.now)
            task_id += 1
            self.env.process(self._task_process(task_obj, record))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        self.env.process(self._arrival_process())
        self.env.run(until=self.cfg.sim_duration)
        return pd.DataFrame([vars(r) for r in self.records])
