# Smart Transaction Categorization - Serving Baseline

This repository is the first baseline version of the serving stack and can be used directly as the starting point for your project's initial implementation.

## Goals
- Provide a stable prediction API contract
- Support both single-request `/predict` and batch `/predict_batch`
- Use a dummy sklearn pipeline as a stand-in for the training team's real model
- Make it easy to later replace the dummy artifact with a real model and add ONNX, batching, multi-worker, Prometheus, Grafana, and cAdvisor
- Provide a CPU-only model optimization notebook that can be executed on Chameleon

## Repository layout
- `app/`: FastAPI service code
- `models/`: model artifacts (currently the dummy baseline)
- `scripts/`: fake data generation, dummy model creation, and benchmark scripts
- `notebooks/`: CPU-only model optimization notebook
- `Dockerfile`: baseline serving container image definition
- `Dockerfile.modelopt`: CPU-only Jupyter image for model optimization experiments

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_dummy_model.py
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- Swagger docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/healthz`
- Readiness check: `http://127.0.0.1:8000/readyz`
- Prometheus metrics: `http://127.0.0.1:8000/metrics`

## Run with Docker
```bash
python scripts/make_dummy_model.py
docker build -t tx-serving-baseline .
docker run --rm -p 8000:8000 tx-serving-baseline
```

## Generate sample requests and run benchmarks
```bash
python scripts/generate_requests.py --output scripts/sample_requests.jsonl --num-samples 200
python scripts/benchmark_single.py --url http://127.0.0.1:8000/predict --requests-file scripts/sample_requests.jsonl --num-requests 100
python scripts/benchmark_batch.py --url http://127.0.0.1:8000/predict_batch --requests-file scripts/sample_requests.jsonl --batch-size 16 --num-batches 20
```

## Push to GitHub
```bash
git init
git add .
git commit -m "Add serving baseline FastAPI scaffold"
gh repo create smart-tx-serving --public --source=. --remote=origin --push
```

---

# Chameleon CPU workflow

This section adapts the lab flow for this project. The lab notebooks create a server, associate a floating IP, install Docker, prepare data, and launch Jupyter. This project follows the same sequence, but uses a CPU VM on `KVM@TACC` and does **not** need the Food-11 dataset volume because the current baseline and model optimization notebook use generated synthetic transaction data instead. The online evaluation lab provisions an `m1.medium` VM on `KVM@TACC` with the `CC-Ubuntu24.04` image and opens ports such as 22, 8000, 8888, 3000, and 9090. That is the correct starting point for this repository.

## 1. Create a VM and floating IP from Chameleon Jupyter

Run the following in the **Chameleon Jupyter environment**, not on your laptop. Replace `proj99` with your actual project ID suffix, because course staff may delete resources that do not include the project suffix in their names.

```python
from chi import server, context, lease, network
import chi, os, datetime

PROJECT_SUFFIX = "proj99"   # change this
context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")

username = os.getenv("USER")

lease_name = f"lease-tx-serving-{username}-{PROJECT_SUFFIX}"
node_name = f"node-tx-serving-{username}-{PROJECT_SUFFIX}"

l = lease.Lease(lease_name, duration=datetime.timedelta(hours=6))
l.add_flavor_reservation(id=chi.server.get_flavor_id("m1.medium"), amount=1)
l.submit(idempotent=True)
l.show()

s = server.Server(
    node_name,
    image_name="CC-Ubuntu24.04",
    flavor_name=l.get_reserved_flavors()[0].name
)
s.submit(idempotent=True)
```

## 2. Attach security groups and associate a floating IP

The online evaluation lab adds security groups before associating a floating IP. For this project, the most useful ports are:
- `22` for SSH
- `8000` for FastAPI
- `8888` for Jupyter
- `3000` for Grafana
- `9090` for Prometheus

```python
security_groups = [
    {'name': f"allow-ssh-{PROJECT_SUFFIX}", 'port': 22, 'description': "SSH"},
    {'name': f"allow-8000-{PROJECT_SUFFIX}", 'port': 8000, 'description': "FastAPI"},
    {'name': f"allow-8888-{PROJECT_SUFFIX}", 'port': 8888, 'description': "Jupyter"},
    {'name': f"allow-3000-{PROJECT_SUFFIX}", 'port': 3000, 'description': "Grafana"},
    {'name': f"allow-9090-{PROJECT_SUFFIX}", 'port': 9090, 'description': "Prometheus"},
]

for sg in security_groups:
    secgroup = network.SecurityGroup({
        'name': sg['name'],
        'description': sg['description'],
    })
    secgroup.add_rule(direction='ingress', protocol='tcp', port=sg['port'])
    secgroup.submit(idempotent=True)
    s.add_security_group(sg['name'])

s.associate_floating_ip()
s.refresh()
s.check_connectivity()
s.show(type="widget")
```

## 3. Install Docker and clone your repo

The labs use `s.execute(...)` from Chameleon Jupyter to set up Docker and retrieve code. The same pattern works here.

```python
REPO_URL = "https://github.com/YOUR_GITHUB_USERNAME/smart-tx-serving.git"  # change this

s.execute(f"git clone {REPO_URL}")
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
```

## 4. SSH into the VM from your local terminal

Use your own Chameleon SSH key and the floating IP shown in the previous step.

```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

## 5. Finish Docker access inside the SSH session

After Docker is installed and your user is added to the `docker` group, either log out and log back in, or run:

```bash
newgrp docker
docker --version
docker ps
```

If `docker ps` works without `sudo`, you are ready.

## 6. Prepare project data and directories

The Food-11 labs create a Docker volume and populate it with dataset files. For **this** project, that is unnecessary at the baseline/model-optimization stage because:
- `scripts/make_dummy_model.py` creates a local dummy model artifact
- `scripts/generate_requests.py` creates synthetic request payloads
- `notebooks/model_optimization_cpu.ipynb` generates its own synthetic transaction data

So instead of pulling a dataset volume, create a few working directories:

```bash
cd smart-tx-serving
mkdir -p results logs tmp
python scripts/make_dummy_model.py
python scripts/generate_requests.py --output scripts/sample_requests.jsonl --num-samples 500
ls -lh models/
head -n 2 scripts/sample_requests.jsonl
```

## 7. Build and run the baseline FastAPI container

Inside the SSH session on the VM:

```bash
cd ~/smart-tx-serving
docker build -t tx-serving-baseline -f Dockerfile .
docker run -d --rm \
  -p 8000:8000 \
  --name tx-serving-baseline \
  tx-serving-baseline
docker ps
```

Check health and docs:
```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

From your browser, open:
- `http://A.B.C.D:8000/docs`
- `http://A.B.C.D:8000/metrics`

## 8. Run benchmark scripts on the VM

Inside the SSH session on the VM:

```bash
cd ~/smart-tx-serving
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/benchmark_single.py \
  --url http://127.0.0.1:8000/predict \
  --requests-file scripts/sample_requests.jsonl \
  --num-requests 100

python scripts/benchmark_batch.py \
  --url http://127.0.0.1:8000/predict_batch \
  --requests-file scripts/sample_requests.jsonl \
  --batch-size 16 \
  --num-batches 20
```

Record the latency and throughput outputs. Those numbers will later populate your serving options table.

---

# CPU-only model optimization workflow

The model optimization lab launches a Jupyter container after the server and Docker are ready. It then runs separate notebooks to measure Torch, ONNX Runtime, and other execution providers. This repository uses the same pattern, but with one consolidated CPU-only notebook for eager, compiled, ONNX Runtime CPU, and ONNX dynamic quantization.

## 9. Build the model-optimization Jupyter image

Inside the SSH session on the VM:

```bash
cd ~/smart-tx-serving
docker build -t tx-modelopt-cpu -f Dockerfile.modelopt .
```

## 10. Launch the Jupyter container

```bash
docker run -d --rm \
  -p 8888:8888 \
  --shm-size 4G \
  -v ~/smart-tx-serving:/workspace \
  --name tx-modelopt \
  tx-modelopt-cpu
```

## 11. Get the Jupyter token

```bash
docker exec tx-modelopt jupyter server list
```

You should see a URL like:
```text
http://localhost:8888/lab?token=XXXXXXXXXXXXXXXX
```

Replace `localhost` with the floating IP:
```text
http://A.B.C.D:8888/lab?token=XXXXXXXXXXXXXXXX
```

Open that in your browser and then run:
- `notebooks/model_optimization_cpu.ipynb`

## 12. Optional headless execution

If you do not want to click through the notebook manually, run it non-interactively inside the container:

```bash
docker exec tx-modelopt bash -lc \
  "cd /workspace && jupyter nbconvert --to notebook --execute --inplace notebooks/model_optimization_cpu.ipynb"
```

After it finishes, inspect:
```bash
ls -lh results/
```

The notebook writes summary outputs that you can later map to model-level rows in your serving options table.

---

# Minimal command checklist

## From Chameleon Jupyter
```python
from chi import server, context, lease, network
import chi, os, datetime

PROJECT_SUFFIX = "proj99"  # change this
context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")

username = os.getenv("USER")
lease_name = f"lease-tx-serving-{username}-{PROJECT_SUFFIX}"
node_name = f"node-tx-serving-{username}-{PROJECT_SUFFIX}"

l = lease.Lease(lease_name, duration=datetime.timedelta(hours=6))
l.add_flavor_reservation(id=chi.server.get_flavor_id("m1.medium"), amount=1)
l.submit(idempotent=True)

s = server.Server(
    node_name,
    image_name="CC-Ubuntu24.04",
    flavor_name=l.get_reserved_flavors()[0].name
)
s.submit(idempotent=True)

security_groups = [
    {'name': f"allow-ssh-{PROJECT_SUFFIX}", 'port': 22, 'description': "SSH"},
    {'name': f"allow-8000-{PROJECT_SUFFIX}", 'port': 8000, 'description': "FastAPI"},
    {'name': f"allow-8888-{PROJECT_SUFFIX}", 'port': 8888, 'description': "Jupyter"},
    {'name': f"allow-3000-{PROJECT_SUFFIX}", 'port': 3000, 'description': "Grafana"},
    {'name': f"allow-9090-{PROJECT_SUFFIX}", 'port': 9090, 'description': "Prometheus"},
]
for sg in security_groups:
    secgroup = network.SecurityGroup({'name': sg['name'], 'description': sg['description']})
    secgroup.add_rule(direction='ingress', protocol='tcp', port=sg['port'])
    secgroup.submit(idempotent=True)
    s.add_security_group(sg['name'])

s.associate_floating_ip()
s.refresh()
s.show(type="widget")

REPO_URL = "https://github.com/YOUR_GITHUB_USERNAME/smart-tx-serving.git"  # change this
s.execute(f"git clone {REPO_URL}")
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
```

## From your local terminal
```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

## Inside the VM
```bash
newgrp docker
git -C ~/smart-tx-serving pull
cd ~/smart-tx-serving

mkdir -p results logs tmp
python3 scripts/make_dummy_model.py
python3 scripts/generate_requests.py --output scripts/sample_requests.jsonl --num-samples 500

docker build -t tx-serving-baseline -f Dockerfile .
docker run -d --rm -p 8000:8000 --name tx-serving-baseline tx-serving-baseline

docker build -t tx-modelopt-cpu -f Dockerfile.modelopt .
docker run -d --rm -p 8888:8888 --shm-size 4G -v ~/smart-tx-serving:/workspace --name tx-modelopt tx-modelopt-cpu

docker exec tx-modelopt jupyter server list
```

---

# Notes

This baseline matches the current project and course requirements:
- It has a clear API contract
- It provides a baseline serving option
- It includes containerized artifacts
- It includes benchmark scripts and a model-level experiment notebook
- It can be built and run on Chameleon immediately after `git clone`
