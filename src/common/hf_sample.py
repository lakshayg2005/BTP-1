"""Fast, balanced sampling from large Hugging Face parquet datasets without streaming every image.

Streaming a dataset whose parquet files are grouped by class/generator forces downloading huge numbers of
images just to reach the rarer label. Instead we:
  1. open parquet files remotely (HfFileSystem, HTTP range requests),
  2. read only the small label column of a row group,
  3. pick the rows still needed (capped per row group and per file, so samples span many files / generators),
  4. read the heavy image column of that row group only when it contributes rows.
"""
from __future__ import annotations

import random
import time
from typing import Callable, Iterator

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem


def sample_rows(repo: str, pattern: str, label_col: str, label_of: Callable, want: dict, columns: list[str],
                cap_per_group: int = 40, groups_per_file: int = 2, seed: int = 0, max_files: int | None = None) -> Iterator[tuple[object, dict]]:
    """Yield (label, row dict with `columns`) until `want[label]` rows of every label are produced."""
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{repo}/{pattern}"))
    if not files:
        raise FileNotFoundError(f"no parquet files match datasets/{repo}/{pattern}")
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_files:
        files = files[:max_files]
    got = {k: 0 for k in want}
    t0, last = time.time(), 0
    print(f"{repo}: {len(files)} parquet files, target {want}", flush=True)
    for path in files:
        pf = pq.ParquetFile(fs.open(path, "rb", block_size=4 * 2 ** 20))
        groups = list(range(pf.num_row_groups))
        rng.shuffle(groups)
        groups = groups[:groups_per_file]          # spread the sample over many files (= generators)
        for g in groups:
            if all(got[k] >= want[k] for k in want):
                return
            labels = [label_of(x) for x in pf.read_row_group(g, columns=[label_col]).column(0).to_pylist()]
            idx = {}
            for i, lab in enumerate(labels):
                if lab in want and got[lab] + len(idx.get(lab, [])) < want[lab] and len(idx.get(lab, [])) < cap_per_group:
                    idx.setdefault(lab, []).append(i)
            if not idx:
                continue
            rows = pf.read_row_group(g, columns=columns).to_pylist()
            for lab, ids in idx.items():
                for i in ids:
                    got[lab] += 1
                    yield lab, rows[i]
            done = sum(got.values())
            if done - last >= 250:
                last = done
                print(f"  {done}/{sum(want.values())} rows {got}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"WARNING: ran out of files; got {got} of {want}", flush=True)
