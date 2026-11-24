#!/usr/bin/env bash
# deploy_to_ec2.sh
# Syncs dt-project to EC2, runs experiments there, pulls results back,
# then starts the FastAPI endpoint + Pushgateway in the background.
#
# Usage (from inside C:\dt-project in Git Bash / WSL):
#   export EC2_HOST=<your-ec2-public-dns-or-ip>
#   bash deploy_to_ec2.sh
#
# The key path defaults to ./dt-project-key.pem (the file already in the repo).
# Never commit dt-project-key.pem to git.

set -euo pipefail

KEY_PATH="${KEY_PATH:-./dt-project-key.pem}"
EC2_HOST="${EC2_HOST:-35.170.59.182}"
EC2_USER="${EC2_USER:-ec2-user}"   # Amazon Linux: ec2-user | Ubuntu: ubuntu
REMOTE_DIR="/home/${EC2_USER}/dt-project"


chmod 400 "$KEY_PATH"
SSH="ssh -i ${KEY_PATH} -o StrictHostKeyChecking=no ${EC2_USER}@${EC2_HOST}"

echo "=== 1. Syncing project to EC2 ==="
rsync -avz --progress --ignore-errors \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pem' \
  --exclude 'venv' '.venv' \
  --exclude 'results/raw_tasks.csv' \
  -e "ssh -i ${KEY_PATH} -o StrictHostKeyChecking=no" \
  ./ "${EC2_USER}@${EC2_HOST}:${REMOTE_DIR}/" || true

echo "=== 2. Installing dependencies on EC2 ==="
$SSH bash <<'REMOTE'
set -euo pipefail
cd ~/dt-project
python3 -m pip install --quiet --upgrade pip
pip3 install --quiet simpy numpy fastapi uvicorn locust prometheus-client pandas
echo "Dependencies OK."
REMOTE

echo "=== 3. Running SimPy experiments on EC2 ==="
$SSH bash <<'REMOTE'
set -euo pipefail
cd ~/dt-project
mkdir -p results
export PYTHONPATH=.
python3 run_experiments_adaptive.py 2>&1 | tee results/run_log.txt
REMOTE

echo "=== 4. Pulling results back ==="
mkdir -p results
scp -i "${KEY_PATH}" -o StrictHostKeyChecking=no \
  "${EC2_USER}@${EC2_HOST}:${REMOTE_DIR}/results/summary_adaptive.csv" \
  results/summary_adaptive_ec2.csv
echo "Saved: results/summary_adaptive_ec2.csv"

echo "=== 5. Starting FastAPI + Pushgateway on EC2 ==="
$SSH bash <<'REMOTE'
set -euo pipefail
cd ~/dt-project
pkill -f "uvicorn api_app" 2>/dev/null || true
nohup uvicorn api_app:app --host 0.0.0.0 --port 8080 > /tmp/api.log 2>&1 &
echo "FastAPI PID $!"

if command -v docker &>/dev/null; then
  docker rm -f pushgateway 2>/dev/null || true
  docker run -d --name pushgateway -p 9091:9091 prom/pushgateway
  echo "Pushgateway running on 9091"
else
  echo "Docker not found — install with: sudo yum install -y docker && sudo service docker start"
fi
REMOTE

echo ""
echo "=== Done ==="
echo "FastAPI:     http://${EC2_HOST}:8080/health"
echo "Pushgateway: http://${EC2_HOST}:9091"
echo ""
echo "Next: run locust load tests:"
echo "  locust -f locustfile.py --headless -u 200 -r 50 --run-time 3m \\"
echo "         --host http://${EC2_HOST}:8080 \\"
echo "         --html results/locust_spike.html --csv results/locust_spike"
