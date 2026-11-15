import redis
import pandas as pd
import json
import time
import random
import os
import subprocess

REDIS_HOST = "localhost"
CLOUD_CPU = 300.0
EDGE_CPU = 150.0
DEADLINE = 300.0
N_TASKS = 2000

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

def load_tasks():
    df = pd.read_csv("batch_task.csv", header=None)
    df.columns = ["job_id","task_id","inst_num","task_type","start_time","status","end_time","plan_cpu"]
    df = df[df["status"] == "Terminated"].head(N_TASKS).reset_index(drop=True)
    return df

def run_policy(policy_name, df):
    r.delete("cloud_queue", "edge_queue", "worker_results")
    stats = {"admitted": 0, "rejected": 0, "cloud": 0, "edge": 0}
    rr_counter = 0

    for i, row in df.iterrows():
        cpu = max(0.05, min(4.0, float(row["plan_cpu"]) * 96))
        now = time.time()
        deadline = now + DEADLINE

        if policy_name == "random":
            decision = random.choice(["cloud", "edge"])
        elif policy_name == "roundrobin":
            decision = "cloud" if rr_counter % 2 == 0 else "edge"
            rr_counter += 1
        elif policy_name == "threshold":
            decision = "cloud" if cpu > 1.0 else "edge"
        else:
            decision = "cloud"

        stats["admitted"] += 1
        stats[decision] += 1
        payload = json.dumps({
            "task_id": "task-" + str(i),
            "cpu": cpu,
            "duration": min(cpu * 0.1, 1.0),
            "deadline": deadline,
            "enqueue_time": now,
            "decision": decision
        })
        r.rpush(decision + "_queue", payload)
        time.sleep(0.02)

    print("[" + policy_name + "] dispatched admitted=" + str(stats["admitted"]) + " cloud=" + str(stats["cloud"]) + " edge=" + str(stats["edge"]), flush=True)
    print("Waiting 60s for workers...", flush=True)
    time.sleep(60)

    results = []
    count = r.llen("worker_results")
    for _ in range(count):
        raw = r.lpop("worker_results")
        if raw:
            results.append(json.loads(raw))

    if len(results) == 0:
        print("[" + policy_name + "] no results collected")
        return {"policy": policy_name, "admitted": stats["admitted"], "rejected": 0,
                "cloud": stats["cloud"], "edge": stats["edge"],
                "completed": 0, "violations": 0, "avg_rt": 0}

    df_res = pd.DataFrame(results)
    completed = df_res[df_res["response_time"] > 0]
    violated = completed[completed["sla_violated"] == 1]
    avg_rt = round(completed["response_time"].mean(), 3) if len(completed) > 0 else 0

    print("[" + policy_name + "] completed=" + str(len(completed)) + " violations=" + str(len(violated)) + " avg_rt=" + str(avg_rt) + "s", flush=True)
    df_res.to_csv("results_" + policy_name + ".csv", index=False)

    return {
        "policy": policy_name,
        "admitted": stats["admitted"],
        "rejected": stats["rejected"],
        "cloud": stats["cloud"],
        "edge": stats["edge"],
        "completed": len(completed),
        "violations": len(violated),
        "avg_rt": avg_rt
    }

df = load_tasks()
print("Loaded " + str(len(df)) + " tasks", flush=True)

all_results = []
for policy in ["random", "roundrobin", "threshold"]:
    print("\n--- Running " + policy + " ---", flush=True)
    subprocess.run(["pkill", "-f", "cloud_worker"], capture_output=True)
    time.sleep(2)
    env = os.environ.copy()
    env["TIER"] = "cloud"
    env["WORKER_ID"] = "cloud-0"
    subprocess.Popen(["python3", "cloud_worker.py"], env=env, stdout=open("/tmp/" + policy + "_c0.log", "w"), stderr=subprocess.STDOUT)
    env2 = os.environ.copy()
    env2["TIER"] = "cloud"
    env2["WORKER_ID"] = "cloud-1"
    subprocess.Popen(["python3", "cloud_worker.py"], env=env2, stdout=open("/tmp/" + policy + "_c1.log", "w"), stderr=subprocess.STDOUT)
    env3 = os.environ.copy()
    env3["TIER"] = "edge"
    env3["WORKER_ID"] = "edge-0"
    subprocess.Popen(["python3", "cloud_worker.py"], env=env3, stdout=open("/tmp/" + policy + "_e0.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(3)
    result = run_policy(policy, df)
    all_results.append(result)

summary = pd.DataFrame(all_results)
summary.to_csv("results_baselines.csv", index=False)
print("\n=== BASELINE COMPARISON ===")
print(summary.to_string(index=False))
