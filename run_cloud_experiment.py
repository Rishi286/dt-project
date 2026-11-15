import redis
import pandas as pd
import json
import time
import sys

sys.path.insert(0, '.')
from dt_controller.adaptive_dt_controller import AdaptiveDTController, Task

REDIS_HOST = "localhost"
CLOUD_CPU = 100.0
EDGE_CPU = 20.0
DEADLINE = 150.0
N_TASKS = 2000

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
r.delete("cloud_queue", "edge_queue", "results", "worker_results")

print("Loading Alibaba trace...", flush=True)
df = pd.read_csv("batch_task.csv", header=None)
df.columns = ["task_name","instance_num","job_name","task_type","status","start_time","end_time","plan_cpu","plan_mem"]
df = df[df["status"] == "Terminated"].head(N_TASKS).reset_index(drop=True)
print(f"Loaded {len(df)} tasks", flush=True)

controller = AdaptiveDTController(
    cloud_cpu=CLOUD_CPU, cloud_mem=1000.0,
    edge_cpu=EDGE_CPU,   edge_mem=200.0
)
pending = {"cloud": 0, "edge": 0}
controller.set_pending(pending)

stats = {"admitted": 0, "rejected": 0, "cloud": 0, "edge": 0}
start_time = time.time()

print("Dispatching tasks...", flush=True)

for i, row in df.iterrows():
    cpu = max(0.05, min(4.0, float(row["plan_cpu"]) * 96))
    mem = max(0.1, cpu * 2.0)
    now = time.time()

    task = Task(
        task_id=f"task-{i}",
        cpu_req=cpu,
        mem_req=mem,
        duration=cpu * 10,
        deadline=now + DEADLINE,
        priority=1,
        arrival_time=now
    )

    decision = controller.allocate(task, now)

    if decision == "reject":
        stats["rejected"] += 1
        r.rpush("worker_results", json.dumps({
            "task_id": task.task_id,
            "decision": "reject",
            "response_time": 0,
            "sla_violated": 0,
            "cpu": cpu,
            "enqueue_time": now
        }))
    else:
        stats["admitted"] += 1
        stats[decision] += 1
        pending[decision] = pending.get(decision, 0) + 1
        payload = json.dumps({
            "task_id": task.task_id,
            "cpu": cpu,
            "duration": min(cpu * 0.1, 1.0),
            "deadline": task.deadline,
            "enqueue_time": now,
            "decision": decision
        })
        r.rpush(f"{decision}_queue", payload)

    time.sleep(0.02)

    if i % 200 == 0:
        weights = controller.snapshot()
        print(f"[{i}/{len(df)}] admitted={stats['admitted']} rejected={stats['rejected']} cloud={stats['cloud']} edge={stats['edge']} weights={weights}", flush=True)

elapsed = time.time() - start_time
print(f"\nDispatch complete in {elapsed:.1f}s", flush=True)
print(f"Admitted: {stats['admitted']} | Rejected: {stats['rejected']} | Cloud: {stats['cloud']} | Edge: {stats['edge']}", flush=True)
print("Waiting 60s for workers to finish...", flush=True)
time.sleep(60)

results = []
for _ in range(r.llen("worker_results")):
    raw = r.lpop("worker_results")
    if raw:
        results.append(json.loads(raw))

df_res = pd.DataFrame(results)
completed = df_res[df_res["response_time"] > 0] if len(df_res) > 0 else pd.DataFrame()
violated = completed[completed["sla_violated"] == 1] if len(completed) > 0 else pd.DataFrame()

print(f"\n=== CLOUD EXPERIMENT RESULTS ===")
print(f"Total tasks:       {len(df)}")
print(f"Admitted:          {stats['admitted']} ({stats['admitted']/len(df)*100:.1f}%)")
print(f"Rejected:          {stats['rejected']} ({stats['rejected']/len(df)*100:.1f}%)")
print(f"Completed:         {len(completed)}")
print(f"SLA Violations:    {len(violated)} ({len(violated)/max(len(completed),1)*100:.1f}%)")
if len(completed) > 0:
    print(f"Avg Response Time: {completed['response_time'].mean():.2f}s")

df_res.to_csv("results_cloud.csv", index=False)
print("Saved to results_cloud.csv")
