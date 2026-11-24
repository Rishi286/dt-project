"""
locustfile.py  —  Locust load test replicating Ebin's three load patterns.

Install:  pip install locust

Run spike (200 users, 3 min):
  locust -f locustfile.py --headless -u 200 -r 50 --run-time 3m \
         --host http://<EC2-PUBLIC-IP>:8080 \
         --html results/locust_spike.html --csv results/locust_spike

Run ramp (gradual 0→150 over 5 min, hold 5 min):
  locust -f locustfile.py --headless --shape-class RampShape \
         --host http://<EC2-PUBLIC-IP>:8080 \
         --html results/locust_ramp.html --csv results/locust_ramp

Run sustained (100 users, 10 min):
  locust -f locustfile.py --headless -u 100 -r 10 --run-time 10m \
         --host http://<EC2-PUBLIC-IP>:8080 \
         --html results/locust_sustained.html --csv results/locust_sustained

Key CSV columns to report in dissertation:
  50%ile, 95%ile, 99%ile, Failure Count  (from locust_*_stats.csv)
"""

import random
from locust import HttpUser, task, between, LoadTestShape


class TaskUser(HttpUser):
    wait_time = between(0.3, 1.0)

    @task(4)
    def send_task(self):
        payload = {
            "cpu_req": round(random.uniform(0.1, 1.0), 2),
            "mem_req": round(random.uniform(0.1, 1.0), 2),
            "duration": round(random.uniform(0.05, 0.25), 3),
        }
        with self.client.post("/task", json=payload, catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def health_check(self):
        self.client.get("/health")


class RampShape(LoadTestShape):
    """Gradual ramp 0 → 150 over 5 min, hold 5 min, ramp down 2 min."""
    stages = [
        {"duration": 300, "users": 150, "spawn_rate": 0.5},
        {"duration": 600, "users": 150, "spawn_rate": 0},
        {"duration": 720, "users":   0, "spawn_rate": 1},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
            run_time -= stage["duration"]
        return None
