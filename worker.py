import redis
import time
import json
import os
import random

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
WORKER_ID = os.environ.get("WORKER_ID", "worker-0")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

print(f"[{WORKER_ID}] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}", flush=True)

while True:
    # Block waiting for a task (timeout 5s)
    item = r.blpop("task_queue", timeout=5)
    if item is None:
        continue

    _, raw = item
    task = json.loads(raw)
    task_id = task["task_id"]
    cpu_cores = task["cpu_cores"]
    deadline = task["deadline"]
    enqueue_time = task["enqueue_time"]

    # Simulate processing: cpu_cores determines work duration
    process_time = cpu_cores * random.uniform(0.8, 1.2)
    time.sleep(min(process_time, 2.0))  # cap at 2s per task for speed

    finish_time = time.time()
    response_time = finish_time - enqueue_time
    sla_violated = 1 if response_time > deadline else 0

    result = {
        "task_id": task_id,
        "worker_id": WORKER_ID,
        "enqueue_time": enqueue_time,
        "finish_time": finish_time,
        "response_time": round(response_time, 3),
        "deadline": deadline,
        "sla_violated": sla_violated,
        "cpu_cores": cpu_cores
    }

    r.rpush("results", json.dumps(result))
    print(f"[{WORKER_ID}] task={task_id} rt={response_time:.2f}s sla_violated={sla_violated}", flush=True)
