"""
prometheus_push.py  —  push SimPy experiment results to Prometheus Pushgateway.
Add `from prometheus_push import push_results` and call `push_results(rows)`
at the end of run_experiments_adaptive.py.

Install:  pip install prometheus-client

Pushgateway on EC2 (run once):
  docker run -d -p 9091:9091 prom/pushgateway

Free Grafana Cloud:
  1. Sign up at https://grafana.com/auth/sign-up/create-user
  2. Stack → Connections → Add new connection → Prometheus (remote write)
  3. Set env vars:
       export PROM_URL="https://<id>.grafana.net/api/prom/push"
       export PROM_USER="<numeric-id>"
       export PROM_TOKEN="<api-token>"
"""

import os, base64, urllib.request
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

PUSHGATEWAY_URL = os.getenv("PROM_URL", "http://localhost:9091")
PROM_USER       = os.getenv("PROM_USER", "")
PROM_TOKEN      = os.getenv("PROM_TOKEN", "")
JOB_NAME        = "dt_experiments"


def _auth_handler(url, method, timeout, headers, data):
    if PROM_USER and PROM_TOKEN:
        creds = base64.b64encode(f"{PROM_USER}:{PROM_TOKEN}".encode()).decode()
        headers["Authorization"] = f"Basic {creds}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def push_results(rows: list):
    registry = CollectorRegistry()
    labels   = ["allocator", "regime", "seed"]

    g_rt   = Gauge("dt_avg_response_time_s",   "Mean task response time",             labels, registry=registry)
    g_sla  = Gauge("dt_sla_violation_rate",     "Fraction of tasks violating deadline", labels, registry=registry)
    g_comp = Gauge("dt_completion_rate",         "Fraction of tasks completed",          labels, registry=registry)
    g_adpt = Gauge("dt_weight_adaptations",      "Weight adaptations (AdaptiveDT)",      labels, registry=registry)
    g_rej  = Gauge("dt_tasks_rejected",          "Tasks rejected",                       labels, registry=registry)

    for r in rows:
        lv = (r["allocator"], r["regime"], str(r["seed"]))
        g_rt.labels(*lv).set(r["avg_rt"])
        g_sla.labels(*lv).set(r["sla_rate"])
        g_comp.labels(*lv).set(r["comp_rate"])
        g_adpt.labels(*lv).set(r.get("n_adapt", 0))
        g_rej.labels(*lv).set(r["n_rejected"])

    handler = _auth_handler if (PROM_USER and PROM_TOKEN) else None
    push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry, handler=handler)
    print(f"[prometheus] pushed {len(rows)} rows → {PUSHGATEWAY_URL}")
