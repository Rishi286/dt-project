"""
api_app.py  —  FastAPI endpoint deployed on EC2 for real HTTP latency measurement.
Gives the dissertation a real cloud HTTP target (vs pure SimPy simulation).

Deploy on EC2:
  pip install fastapi uvicorn
  nohup uvicorn api_app:app --host 0.0.0.0 --port 8080 &
"""

import time, random, math
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="DT Research Endpoint", version="1.0.0")


class TaskRequest(BaseModel):
    cpu_req: float = 1.0      # normalised 0-1
    mem_req: float = 1.0      # normalised 0-1
    duration: float = 0.1     # simulated processing time in seconds


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/task")
def process_task(req: TaskRequest):
    """Accept a task and simulate CPU-bound work proportional to cpu_req."""
    start = time.perf_counter()
    deadline = time.perf_counter() + max(0.01, req.duration * req.cpu_req)
    x = 0.0
    while time.perf_counter() < deadline:
        x += math.sqrt(random.random())
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "status": "completed",
        "response_time_ms": round(elapsed_ms, 2),
        "cpu_req": req.cpu_req,
        "duration": req.duration,
    }


@app.get("/")
def root():
    return {"service": "dt-research-api", "docs": "/docs"}
