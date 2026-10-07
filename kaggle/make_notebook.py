"""Generates kaggle/btp_kaggle.ipynb (kept as code so the notebook stays reviewable in git)."""
import json
from pathlib import Path

CELLS = [
    ("md", "# BTP: explainable AI-content detection, Kaggle GPU runner\n"
           "Settings: **Accelerator = GPU T4**, **Internet = On**. Then *Save Version -> Save & Run All (Commit)* "
           "so it runs in the background. Outputs (models, features, metrics, logs) appear under *Output*."),
    ("code", "# Stop early with a clear message if the session is misconfigured\n"
             "import shutil, socket\n"
             "problems = []\n"
             "if not shutil.which('nvidia-smi'):\n"
             "    problems.append('No GPU: right panel -> Session options -> Accelerator -> GPU T4 x2 (needs phone verification)')\n"
             "try:\n"
             "    socket.create_connection(('github.com', 443), timeout=10).close()\n"
             "except OSError:\n"
             "    problems.append('No internet: right panel -> Session options -> Internet -> On (needs phone verification)')\n"
             "if problems:\n"
             "    raise SystemExit('FIX SESSION SETTINGS FIRST:\\n- ' + '\\n- '.join(problems))\n"
             "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    ("code", "%cd /kaggle/working\n"
             "!rm -rf BTP-1 && git clone -q --depth 1 https://github.com/lakshayg2005/BTP-1.git\n"
             "%cd BTP-1\n"
             "# big, re-downloadable things go to /kaggle/tmp so the saved output stays small\n"
             "!mkdir -p /kaggle/tmp/data /kaggle/tmp/hf && ln -sfn /kaggle/tmp/data data\n"
             "import os\n"
             "os.environ.update(PY='python', HF_HOME='/kaggle/tmp/hf', TMPDIR_BTP='/kaggle/tmp/t',\n"
             "                  IMG_TRAIN='3000', IMG_TEST='500', SID_TRAIN='1500', SID_TEST='500', TXT_TRAIN='2000', TXT_TEST='400', BTP_TEXT_LM_SIZE='1.5B')"),
    ("code", "!bash scripts/run_phase.sh deps 2>&1 | tail -3\n"
             "!python -c \"import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())\""),
    ("md", "## Phase 1: image AI detector"),
    ("code", "!bash scripts/run_phase.sh image 2>&1 | grep -v -E 'it/s\\]|s/it\\]' | tail -60"),
    ("md", "## Phase 2: edit localizer (SID-Set masks + synthetic edits)"),
    ("code", "!bash scripts/run_phase.sh tamper 2>&1 | grep -v -E 'it/s\\]|s/it\\]' | tail -30"),
    ("md", "## Phase 3: audio"),
    ("code", "!bash scripts/run_phase.sh audio 2>&1 | grep -v -E 'it/s\\]|s/it\\]' | tail -60"),
    ("md", "## Phase 4: text"),
    ("code", "!bash scripts/run_phase.sh text 2>&1 | grep -v -E 'it/s\\]|s/it\\]' | tail -60"),
    ("md", "## Evaluation: results tables + example explanations"),
    ("code", "!bash scripts/run_phase.sh evaluate 2>&1 | grep -v -E 'it/s\\]|s/it\\]' | tail -150"),
    ("code", "from IPython.display import Markdown, display\n"
             "display(Markdown(open('results/summary.md').read()))"),
    ("code", "# keep small outputs only; big data stays in /kaggle/tmp\n"
             "!cd /kaggle/working && zip -qr btp_outputs.zip BTP-1/models BTP-1/results BTP-1/logs BTP-1/features -x '*.npz' && ls -la btp_outputs.zip"),
    ("md", "## Live demo (interactive sessions only)\nOpen the notebook in edit mode, run all cells, then this one prints a public gradio link."),
    ("code", "import os\n"
             "if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':\n"
             "    !python -m app.demo --share\n"
             "else:\n"
             "    print('Batch run: demo skipped')"),
]


def cell(kind, src):
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


nb = {"cells": [cell(k, s) for k, s in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
Path(__file__).with_name("btp_kaggle.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("written")
