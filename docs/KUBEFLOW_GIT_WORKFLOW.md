# Git + Kubeflow workflow

This repo already has `origin` configured. Use the workflow below so your local edits are always available in Kubeflow notebooks.

## 1) Local machine: commit + push in one command

From the repo root:

```bash
chmod +x scripts/git_sync.sh scripts/kubeflow_sync.sh
./scripts/git_sync.sh "describe your change"
```

If you omit the commit message, it auto-generates one with a timestamp.

## 2) Kubeflow notebook: first clone, then pull on later sessions

First time in a notebook:

```bash
%%bash
set -e
git clone -b clean-main https://github.com/rym173/euh-fyp.git /home/jovyan/work/EoH-modified
python -m pip install -e /home/jovyan/work/EoH-modified/eoh
```

Later notebook sessions:

```bash
%%bash
set -e
bash /home/jovyan/work/EoH-modified/scripts/kubeflow_sync.sh \
  https://github.com/rym173/euh-fyp.git \
  clean-main \
  /home/jovyan/work/EoH-modified
```

## 3) Run your project in Kubeflow

After sync:

```bash
cd /home/jovyan/work/EoH-modified
python examples/bp_online.py
```

## Optional: private repo auth

If the repo is private, use one of:

- HTTPS + Personal Access Token in Kubeflow secrets
- SSH key mounted in notebook pod (`git@github.com:...`)
