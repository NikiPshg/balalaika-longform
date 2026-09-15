#!/usr/bin/env python3
"""Launch the frozen F5 training recipe using the selected Python environment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('config', type=Path)
p.add_argument('--dry-run', action='store_true')
a = p.parse_args()
config = json.loads(a.config.read_text())
cmd = [sys.executable, '-m', 'src.f5_training.train']
for key, value in config.items():
    cmd.extend(['--' + key.replace('_', '-'), str(value)])
if a.dry_run:
    import shlex
    print(shlex.join(cmd))
else:
    if (ROOT / config['output_dir']).exists():
        raise SystemExit('Output already exists; select a fresh output_dir in a copied config.')
    subprocess.run(cmd, cwd=ROOT, check=True)
