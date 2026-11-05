from dataclasses import dataclass, field
from typing import List


@dataclass
class TierState:
    cpu_capacity: float
    mem_capacity: float
    cpu_used: float = 0.0
    mem_used: float = 0.0
    sla_violations: int = 0
    completed_tasks: int = 0
    total_response_time: float = 0.0
    in_flight: int = 0

    @property
    def cpu_util(self):
        return self.cpu_used / self.cpu_capacity if self.cpu_capacity > 0 else 1.0

    @property
    def viol_rate(self):
        return self.sla_violations / self.completed_tasks if self.completed_tasks else 0.0

    @property
    def load_ratio(self):
        return self.in_flight / self.cpu_capacity if self.cpu_capacity > 0 else 1.0


@dataclass
class Task:
    task_id: str
    cpu_req: float
    mem_req: float
    duration: float
    deadline: float
    priority: int
    arrival_time: float = 0.0


class AdaptiveDTController:
    ALPHA_INIT = 0.4
    BETA_INIT  = 0.3
    GAMMA_INIT = 0.3
    ADAPT_INTERVAL = 50
    MIN_WEIGHT = 0.10
    MAX_WEIGHT = 0.70
    # Reject when backlog >= CAP_RATIO * tier capacity.
    # At 0.5x cloud (50 tasks) queue wait already exceeds most deadlines.
    BACKLOG_CAP_RATIO = 0.5

    def __init__(self, cloud_cpu, cloud_mem, edge_cpu, edge_mem):
        self.cloud = TierState(cpu_capacity=cloud_cpu, mem_capacity=cloud_mem)
        self.edge  = TierState(cpu_capacity=edge_cpu,  mem_capacity=edge_mem)
        self.alpha = self.ALPHA_INIT
        self.beta  = self.BETA_INIT
        self.gamma = self.GAMMA_INIT
        self.weight_history = []
        self._tasks_since_adapt = 0
        self._pending = {"cloud": 0, "edge": 0}

    def set_pending(self, pending_dict):
        """Receive live reference to runner's pending backlog dict."""
        self._pending = pending_dict

    def _adapt_weights(self):
        total_comp = self.cloud.completed_tasks + self.edge.completed_tasks
        total_viol = self.cloud.sla_violations  + self.edge.sla_violations
        if total_comp == 0:
            return
        viol_rate = total_viol / total_comp
        if viol_rate > 0.15:
            delta = min(0.05, (viol_rate - 0.15) * 0.25)
            self.gamma = min(self.gamma + delta, self.MAX_WEIGHT)
            self.alpha = max(self.alpha - delta * 0.5, self.MIN_WEIGHT)
            self.beta  = max(self.beta  - delta * 0.5, self.MIN_WEIGHT)
        elif viol_rate < 0.05:
            delta = 0.02
            self.beta  = min(self.beta  + delta, self.MAX_WEIGHT)
            self.gamma = max(self.gamma - delta * 0.5, self.MIN_WEIGHT)
            self.alpha = max(self.alpha - delta * 0.5, self.MIN_WEIGHT)
        s = self.alpha + self.beta + self.gamma
        self.alpha /= s
        self.beta  /= s
        self.gamma /= s
        self.weight_history.append({
            "completed": total_comp,
            "alpha": round(self.alpha, 4),
            "beta":  round(self.beta,  4),
            "gamma": round(self.gamma, 4),
            "viol_rate": round(viol_rate, 4),
        })

    def _admission_check(self, task, current_time):
        pcloud = self._pending.get("cloud", 0)
        pedge  = self._pending.get("edge",  0)

        # Hard backlog cap — primary load-shedding gate
        cloud_cap = self.cloud.cpu_capacity * self.BACKLOG_CAP_RATIO
        edge_cap  = self.edge.cpu_capacity  * self.BACKLOG_CAP_RATIO
        cloud_over = pcloud >= cloud_cap
        edge_over  = pedge  >= edge_cap
        if cloud_over or edge_over:
            return False
        # Reject if the only non-saturated tier cannot absorb more
        if cloud_over and pedge >= self.edge.cpu_capacity:
            return False
        if edge_over and pcloud >= self.cloud.cpu_capacity:
            return False

        # Temporal feasibility
        if task.deadline <= current_time:
            return False
        time_slack = task.deadline - current_time
        if time_slack < task.duration * 1.1:
            return False

        # Queue-wait feasibility
        avg_service = 30.0
        cloud_wait = (pcloud * avg_service / max(self.cloud.cpu_capacity, 1)
                      if not cloud_over else float("inf"))
        edge_wait  = (pedge  * avg_service / max(self.edge.cpu_capacity,  1)
                      if not edge_over  else float("inf"))
        best_wait = min(cloud_wait, edge_wait)
        if best_wait > time_slack * 0.3:
            return False

        return True

    def _score(self, tier, queue_depth, task, current_time):
        load_pressure  = min(tier.load_ratio, 1.0)
        queue_pressure = min(queue_depth / max(tier.cpu_capacity, 1), 1.0)
        latency_risk   = load_pressure * 0.7 + queue_pressure * 0.3

        post_util = (tier.cpu_used + task.cpu_req) / tier.cpu_capacity
        wastage   = max(0.0, 1.0 - min(post_util, 1.0))

        time_slack = task.deadline - current_time
        if time_slack <= task.duration:
            sla_risk = 1.0
        elif time_slack < task.duration * 2.0:
            sla_risk = 0.6
        elif time_slack < task.duration * 3.0:
            sla_risk = 0.2
        else:
            sla_risk = 0.0
        if tier.completed_tasks > 0:
            sla_risk = min(sla_risk + tier.viol_rate * 0.4, 1.0)

        return -(self.alpha * latency_risk + self.beta * wastage + self.gamma * sla_risk)

    def allocate(self, task, current_time):
        if not self._admission_check(task, current_time):
            return "reject"

        pcloud = self._pending.get("cloud", 0)
        pedge  = self._pending.get("edge",  0)

        cs = self._score(self.cloud, pcloud, task, current_time) + 0.15
        es = self._score(self.edge,  pedge,  task, current_time)
        decision = "cloud" if cs >= es else "edge"

        tier = self.cloud if decision == "cloud" else self.edge
        tier.cpu_used += task.cpu_req
        tier.mem_used += task.mem_req
        return decision

    def release(self, decision, task, response_time, violated):
        tier = self.cloud if decision == "cloud" else self.edge
        tier.cpu_used             = max(0.0, tier.cpu_used - task.cpu_req)
        tier.mem_used             = max(0.0, tier.mem_used - task.mem_req)
        tier.in_flight            = max(0, tier.in_flight - 1)
        tier.completed_tasks     += 1
        tier.total_response_time += response_time
        if violated:
            tier.sla_violations += 1
        self._tasks_since_adapt += 1
        if self._tasks_since_adapt >= self.ADAPT_INTERVAL:
            self._adapt_weights()
            self._tasks_since_adapt = 0

    def snapshot(self):
        return {
            "alpha": round(self.alpha, 4),
            "beta":  round(self.beta,  4),
            "gamma": round(self.gamma, 4),
        }
