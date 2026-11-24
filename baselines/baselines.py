"""
Baseline allocators for comparison against the DT controller.

1. ThresholdAllocator       — rule-based: send to edge if edge CPU < 70%, else cloud
2. GreedyBestFitAllocator   — greedy: pick tier with most available CPU
3. EWMAAllocator            — predictive: EWMA-smoothed utilisation, pre-empts saturation
"""

from dataclasses import dataclass, field
from typing import Union
Literal = Union  # py3.7 compat


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

    def allocate(self, task: Task, current_time: float) -> str:
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

    def allocate(self, task: Task, current_time: float) -> str:
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


class EWMAAllocator:
    """
    Exponentially Weighted Moving Average predictive allocator.

    Tracks cloud and edge CPU utilisation with EWMA smoothing (alpha).
    When the smoothed cloud forecast exceeds high_water, it pre-emptively
    routes new tasks to edge — mirroring an EWMA-based Kubernetes controller
    that bumps minReplicas before the threshold is crossed.

    Parameters
    ----------
    alpha      : smoothing factor (0 < alpha <= 1); higher = more reactive
    high_water : cloud utilisation fraction above which edge is preferred
    """

    def __init__(self, cloud_cpu, cloud_mem, edge_cpu, edge_mem,
                 alpha: float = 0.3, high_water: float = 0.80):
        self.cloud_cpu   = cloud_cpu
        self.cloud_mem   = cloud_mem
        self.edge_cpu    = edge_cpu
        self.edge_mem    = edge_mem
        self.alpha       = alpha
        self.high_water  = high_water

        self._ewma_cloud = 0.0
        self._ewma_edge  = 0.0

        self._used_cloud_cpu = 0.0
        self._used_edge_cpu  = 0.0
        self._used_cloud_mem = 0.0
        self._used_edge_mem  = 0.0

        self.allocation_log: list = []
        self.weight_history: list = []   # checked via hasattr in experiment runner

    def _update_ewma(self):
        obs_cloud = self._used_cloud_cpu / self.cloud_cpu
        obs_edge  = self._used_edge_cpu  / self.edge_cpu
        self._ewma_cloud = self.alpha * obs_cloud + (1 - self.alpha) * self._ewma_cloud
        self._ewma_edge  = self.alpha * obs_edge  + (1 - self.alpha) * self._ewma_edge

    def allocate(self, task: Task, current_time: float) -> str:
        can_cloud = (self._used_cloud_cpu + task.cpu_req <= self.cloud_cpu and
                     self._used_cloud_mem + task.mem_req <= self.cloud_mem)
        can_edge  = (self._used_edge_cpu  + task.cpu_req <= self.edge_cpu  and
                     self._used_edge_mem  + task.mem_req <= self.edge_mem)

        if not can_cloud and not can_edge:
            return "reject"

        # Predictive: if EWMA forecast says cloud is heading toward saturation, offload early
        if self._ewma_cloud >= self.high_water and can_edge:
            decision = "edge"
        elif can_cloud:
            decision = "cloud"
        elif can_edge:
            decision = "edge"
        else:
            return "reject"

        if decision == "cloud":
            self._used_cloud_cpu += task.cpu_req
            self._used_cloud_mem += task.mem_req
        else:
            self._used_edge_cpu  += task.cpu_req
            self._used_edge_mem  += task.mem_req

        self.allocation_log.append({
            "task_id": task.task_id, "decision": decision, "time": current_time,
            "ewma_cloud": round(self._ewma_cloud, 4), "ewma_edge": round(self._ewma_edge, 4),
        })
        return decision

    def release(self, decision: str, task: Task, response_time: float, violated: bool):
        if decision == "cloud":
            self._used_cloud_cpu = max(0, self._used_cloud_cpu - task.cpu_req)
            self._used_cloud_mem = max(0, self._used_cloud_mem - task.mem_req)
        else:
            self._used_edge_cpu  = max(0, self._used_edge_cpu  - task.cpu_req)
            self._used_edge_mem  = max(0, self._used_edge_mem  - task.mem_req)
        self._update_ewma()

    def set_pending(self, pending_dict):
        pass   # EWMA tracks its own usage internally
