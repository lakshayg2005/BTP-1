"""Generates kaggle/btp_kaggle.ipynb (kept as code so the notebook stays reviewable in git)."""
import json
from pathlib import Path

CELLS = [
    ("md", "# BTP: explainable AI-content detection, Kaggle GPU runner\n"
           "Settings: **Accelerator = GPU T4**, **Internet = On**. Then *Save Version -> Save & Run All (Commit)* "
           "so it runs in the background. Outputs (models, features, metrics, logs) appear under *Output*."),
    ("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    ("code", "%cd /kaggle/working\n"
             "!rm -rf BTP-1 && git clone -q --depth 1 https://github.com/lakshayg2005/BTP-1.git\n"
             "%cd BTP-1\n"
             "# big, re-downloadable things go to /kaggle/tmp so the saved output stays small\n"
             "!mkdir -p /kaggle/tmp/data /kaggle/tmp/hf && ln -sfn /kaggle/tmp/data data\n"
             "import os\n"
             "os.environ.update(PY='python', HF_HOME='/kaggle/tmp/hf', TMPDIR_BTP='/kaggle/tmp/t',\n"
             "                  IMG_TRAIN='3000', IMG_TEST='500', TXT_TRAIN='2000', TXT_TEST='400', BTP_TEXT_LM_SIZE='1.5B')"),
    ("code", "!bash scripts/run_phase.sh deps 2>&1 | tail -3\n"
             "!python -c \"import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())\""),
    ("md", "## Phase 1: image"),
    ("code", "!bash scripts/run_phase.sh image 2>&1 | grep -v -E 'it/s\\]|s/it\\]' | tail -60"),
    ("md", "## Phase 3: text"),
    ("code", "!bash scripts/run_phase.sh text 2>&1 | grep -v -E 'it/s\\]|s/it\\]' | tail -60"),
    ("md", "## Sanity check: full pipeline + explanations on a few test images"),
    ("code", "import glob, subprocess\n"
             "for p in sorted(glob.glob('data/images/test/ai/*.png'))[:2] + sorted(glob.glob('data/images/test/real/*.png'))[:2]:\n"
             "    print('=' * 80, '\\n', p)\n"
             "    print(subprocess.run(['python', '-m', 'scripts.analyze', p, '--explainer', 'llm'], capture_output=True, text=True).stdout[-1500:])"),
    ("code", "!ls -la models features && cat models/*.metrics.json"),
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
