"""
Baseline allocators for comparison against the DT controller.

1. ThresholdAllocator  — rule-based: send to edge if edge CPU < 70%, else cloud
2. GreedyBestFitAllocator — greedy: pick tier with most available CPU
"""

from dataclasses import dataclass, field
from typing import Literal


# Reuse TierState and Task from dt_controller
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dt_controller'))
from dt_controller import TierState, Task


class ThresholdAllocator:
    """
    Rule-based threshold policy.
    Send to edge if edge CPU utilisation < EDGE_THRESHOLD, else cloud.
    Falls back to the other tier if the preferred one cannot fit the task.
    """
    EDGE_THRESHOLD = 0.70

    def __init__(self, cloud_cpu, cloud_mem, edge_cpu, edge_mem):
        self.cloud = TierState(cpu_capacity=cloud_cpu, mem_capacity=cloud_mem)
        self.edge  = TierState(cpu_capacity=edge_cpu,  mem_capacity=edge_mem)
        self.allocation_log: list = []

    def allocate(self, task: Task, current_time: float) -> Literal["cloud", "edge", "reject"]:
        cloud_fits = (self.cloud.cpu_used + task.cpu_req <= self.cloud.cpu_capacity and
                      self.cloud.mem_used + task.mem_req <= self.cloud.mem_capacity)
        edge_fits  = (self.edge.cpu_used  + task.cpu_req <= self.edge.cpu_capacity  and
                      self.edge.mem_used  + task.mem_req <= self.edge.mem_capacity)

        if not cloud_fits and not edge_fits:
            return "reject"

        prefer_edge = self.edge.cpu_util < self.EDGE_THRESHOLD

        if prefer_edge and edge_fits:
            decision = "edge"
        elif cloud_fits:
            decision = "cloud"
        elif edge_fits:
            decision = "edge"
        else:
            return "reject"

        self._apply(decision, task)
        self.allocation_log.append({
            "task_id": task.task_id, "decision": decision, "time": current_time,
            "cloud_cpu_util": self.cloud.cpu_util, "edge_cpu_util": self.edge.cpu_util,
        })
        return decision

    def release(self, decision: str, task: Task, response_time: float, violated: bool):
        tier = self.cloud if decision == "cloud" else self.edge
        tier.cpu_used     = max(0, tier.cpu_used - task.cpu_req)
        tier.mem_used     = max(0, tier.mem_used - task.mem_req)
        tier.queue_length = max(0, tier.queue_length - 1)
        tier.completed_tasks     += 1
        tier.total_response_time += response_time
        if violated:
            tier.sla_violations += 1

    def _apply(self, decision: str, task: Task):
        tier = self.cloud if decision == "cloud" else self.edge
        tier.cpu_used     += task.cpu_req
        tier.mem_used     += task.mem_req
        tier.queue_length += 1


class GreedyBestFitAllocator:
    """
    Greedy best-fit policy.
    Always pick the tier with the most remaining CPU capacity.
    Falls back if that tier cannot fit the task.
    """

    def __init__(self, cloud_cpu, cloud_mem, edge_cpu, edge_mem):
        self.cloud = TierState(cpu_capacity=cloud_cpu, mem_capacity=cloud_mem)
        self.edge  = TierState(cpu_capacity=edge_cpu,  mem_capacity=edge_mem)
        self.allocation_log: list = []

    def allocate(self, task: Task, current_time: float) -> Literal["cloud", "edge", "reject"]:
        cloud_fits = (self.cloud.cpu_used + task.cpu_req <= self.cloud.cpu_capacity and
                      self.cloud.mem_used + task.mem_req <= self.cloud.mem_capacity)
        edge_fits  = (self.edge.cpu_used  + task.cpu_req <= self.edge.cpu_capacity  and
                      self.edge.mem_used  + task.mem_req <= self.edge.mem_capacity)

        if not cloud_fits and not edge_fits:
            return "reject"

        cloud_free = self.cloud.cpu_capacity - self.cloud.cpu_used
        edge_free  = self.edge.cpu_capacity  - self.edge.cpu_used

        if cloud_free >= edge_free:
            decision = "cloud" if cloud_fits else "edge"
        else:
            decision = "edge" if edge_fits else "cloud"

        self._apply(decision, task)
        self.allocation_log.append({
            "task_id": task.task_id, "decision": decision, "time": current_time,
            "cloud_cpu_util": self.cloud.cpu_util, "edge_cpu_util": self.edge.cpu_util,
        })
        return decision

    def release(self, decision: str, task: Task, response_time: float, violated: bool):
        tier = self.cloud if decision == "cloud" else self.edge
        tier.cpu_used     = max(0, tier.cpu_used - task.cpu_req)
        tier.mem_used     = max(0, tier.mem_used - task.mem_req)
        tier.queue_length = max(0, tier.queue_length - 1)
        tier.completed_tasks     += 1
        tier.total_response_time += response_time
        if violated:
            tier.sla_violations += 1

    def _apply(self, decision: str, task: Task):
        tier = self.cloud if decision == "cloud" else self.edge
        tier.cpu_used     += task.cpu_req
        tier.mem_used     += task.mem_req
        tier.queue_length += 1
