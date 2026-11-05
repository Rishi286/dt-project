"""
Digital Twin Controller
Maintains a virtual model of cloud and edge resource states.
Makes allocation decisions based on current state snapshot.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TierState:
    """Resource state for one tier (cloud or edge)."""
    cpu_capacity: float        # total CPU units
    mem_capacity: float        # total memory MB
    cpu_used: float = 0.0
    mem_used: float = 0.0
    queue_length: int = 0
    sla_violations: int = 0
    completed_tasks: int = 0
    total_response_time: float = 0.0

    @property
    def cpu_util(self) -> float:
        return self.cpu_used / self.cpu_capacity if self.cpu_capacity > 0 else 1.0

    @property
    def mem_util(self) -> float:
        return self.mem_used / self.mem_capacity if self.mem_capacity > 0 else 1.0

    @property
    def avg_response_time(self) -> float:
        if self.completed_tasks == 0:
            return 0.0
        return self.total_response_time / self.completed_tasks


@dataclass
class Task:
    task_id: str
    cpu_req: float       # CPU units required
    mem_req: float       # memory MB required
    duration: float      # estimated execution time (seconds)
    deadline: float      # absolute deadline (sim time)
    priority: int        # scheduling_class 0-3 (3 = highest)
    arrival_time: float = 0.0


class DigitalTwinController:
    """
    DT-based resource allocator.
    Uses MDP-inspired scoring:
      reward = -(alpha * latency_risk + beta * wastage + gamma * sla_risk)
    alpha=0.4, beta=0.3, gamma=0.3
    """

    ALPHA = 0.4   # latency weight
    BETA  = 0.3   # resource wastage weight
    GAMMA = 0.3   # SLA risk weight

    # Thresholds for risk assessment
    UTIL_HIGH = 0.80
    UTIL_CRIT = 0.90

    def __init__(self, cloud_cpu, cloud_mem, edge_cpu, edge_mem):
        self.cloud = TierState(cpu_capacity=cloud_cpu, mem_capacity=cloud_mem)
        self.edge  = TierState(cpu_capacity=edge_cpu,  mem_capacity=edge_mem)
        self.allocation_log: list = []

    # ------------------------------------------------------------------
    # State update (called by simulation after each event)
    # ------------------------------------------------------------------

    def update_cloud(self, cpu_used, mem_used, queue_len, sla_violations,
                     completed, total_rt):
        self.cloud.cpu_used          = cpu_used
        self.cloud.mem_used          = mem_used
        self.cloud.queue_length      = queue_len
        self.cloud.sla_violations    = sla_violations
        self.cloud.completed_tasks   = completed
        self.cloud.total_response_time = total_rt

    def update_edge(self, cpu_used, mem_used, queue_len, sla_violations,
                    completed, total_rt):
        self.edge.cpu_used           = cpu_used
        self.edge.mem_used           = mem_used
        self.edge.queue_length       = queue_len
        self.edge.sla_violations     = sla_violations
        self.edge.completed_tasks    = completed
        self.edge.total_response_time = total_rt

    # ------------------------------------------------------------------
    # Allocation decision
    # ------------------------------------------------------------------

    def allocate(self, task: Task, current_time: float) -> Literal["cloud", "edge", "reject"]:
        """Return 'cloud', 'edge', or 'reject' if neither tier can fit the task."""

        cloud_fits = (self.cloud.cpu_used + task.cpu_req <= self.cloud.cpu_capacity and
                      self.cloud.mem_used + task.mem_req <= self.cloud.mem_capacity)
        edge_fits  = (self.edge.cpu_used  + task.cpu_req <= self.edge.cpu_capacity  and
                      self.edge.mem_used  + task.mem_req <= self.edge.mem_capacity)

        if not cloud_fits and not edge_fits:
            return "reject"
        if cloud_fits and not edge_fits:
            decision = "cloud"
        elif edge_fits and not cloud_fits:
            decision = "edge"
        else:
            # Both fit — score each tier
            cloud_score = self._score(self.cloud, task, current_time, "cloud")
            edge_score  = self._score(self.edge,  task, current_time, "edge")
            decision    = "cloud" if cloud_score >= edge_score else "edge"

        self._apply(decision, task)
        self.allocation_log.append({
            "task_id": task.task_id,
            "decision": decision,
            "time": current_time,
            "priority": task.priority,
            "cloud_cpu_util": self.cloud.cpu_util,
            "edge_cpu_util":  self.edge.cpu_util,
        })
        return decision

    def _score(self, tier: TierState, task: Task, current_time: float,
               tier_name: str = "") -> float:
        """Higher score = better placement."""
        # Latency risk: higher utilisation → longer queue wait
        latency_risk = tier.cpu_util * 0.6 + (tier.queue_length / 50.0) * 0.4
        latency_risk = min(latency_risk, 1.0)

        # Wastage: how much of capacity remains unused after placement
        post_cpu_util = (tier.cpu_used + task.cpu_req) / tier.cpu_capacity
        post_mem_util = (tier.mem_used + task.mem_req) / tier.mem_capacity
        wastage = 1.0 - (post_cpu_util * 0.5 + post_mem_util * 0.5)

        # SLA risk: time remaining vs estimated duration
        time_slack = task.deadline - current_time
        sla_risk = 0.0
        if time_slack <= task.duration:          # will almost certainly miss
            sla_risk = 1.0
        elif time_slack < task.duration * 1.5:  # tight
            sla_risk = 0.5
        # boost SLA risk if tier is already violating
        if tier.sla_violations > 0 and tier.completed_tasks > 0:
            violation_rate = tier.sla_violations / tier.completed_tasks
            sla_risk = min(sla_risk + violation_rate * 0.3, 1.0)

        cost = self.ALPHA * latency_risk + self.BETA * wastage + self.GAMMA * sla_risk
        return -cost   # higher (less negative) = better

    def _apply(self, decision: str, task: Task):
        tier = self.cloud if decision == "cloud" else self.edge
        tier.cpu_used += task.cpu_req
        tier.mem_used += task.mem_req
        tier.queue_length += 1

    def release(self, decision: str, task: Task, response_time: float, violated: bool):
        tier = self.cloud if decision == "cloud" else self.edge
        tier.cpu_used      = max(0, tier.cpu_used - task.cpu_req)
        tier.mem_used      = max(0, tier.mem_used - task.mem_req)
        tier.queue_length  = max(0, tier.queue_length - 1)
        tier.completed_tasks      += 1
        tier.total_response_time  += response_time
        if violated:
            tier.sla_violations += 1

    # ------------------------------------------------------------------
    # Metrics snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "cloud_cpu_util":    self.cloud.cpu_util,
            "cloud_mem_util":    self.cloud.mem_util,
            "cloud_queue":       self.cloud.queue_length,
            "cloud_sla_viol":    self.cloud.sla_violations,
            "cloud_avg_rt":      self.cloud.avg_response_time,
            "edge_cpu_util":     self.edge.cpu_util,
            "edge_mem_util":     self.edge.mem_util,
            "edge_queue":        self.edge.queue_length,
            "edge_sla_viol":     self.edge.sla_violations,
            "edge_avg_rt":       self.edge.avg_response_time,
        }
