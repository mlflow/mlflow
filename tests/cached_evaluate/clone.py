import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import evaluate

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    # Sparse clone the evaluate repository
    subprocess.check_call(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "https://github.com/huggingface/evaluate.git",
            tmpdir,
        ]
    )
    paths = [
        "metrics/rouge",
        "metrics/bleu",
        "measurements/toxicity",
    ]
    subprocess.check_call(["git", "sparse-checkout", "set"] + paths, cwd=tmpdir)
    subprocess.check_call(["git", "checkout"], cwd=tmpdir)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmpdir).decode().strip()

    # Remove non-python files and app.py
    for p in tmpdir.rglob("*"):
        if p.is_file() and p.suffix != ".py" or p.name == "app.py":
            p.unlink()

    # Ensure `evaluate.load` works fine
    for p in paths:
        evaluate.load(str(tmpdir / p))

    # Copy the files in the evaluate directory
    this_dir = Path(__file__).parent
    dst_dir = this_dir / "evaluate"
    shutil.rmtree(dst_dir)
    for f in paths:
        shutil.copytree(tmpdir / f, dst_dir / f)

    # Write metadata
    with this_dir.joinpath("info.json").open("w") as f:
        json.dump({"head": head, "timestamp": datetime.now().isoformat()}, f)
