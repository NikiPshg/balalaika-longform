#!/usr/bin/env python3
"""Build both paper PDFs from main.tex in an isolated output directory."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
ASSETS = (
    'refs.bib', 'spconf.sty', 'IEEEbib.bst',
    'fig_wer_limited50_wide.pdf', 'fig_quality_limited50.pdf',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['tectonic', 'latexmk'], default='tectonic')
    parser.add_argument('--compiler', help='Override the selected compiler executable')
    parser.add_argument('--only-cached', action='store_true', help='Tectonic: use cached TeX resources only')
    parser.add_argument('--variant', choices=['all', 'draft', 'submission'], default='all')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'build')
    args = parser.parse_args()
    executable = shutil.which(args.compiler or args.engine)
    if executable is None:
        parser.error(f'Compiler not found: {args.compiler or args.engine}. Install it or pass --compiler PATH.')
    if args.only_cached and args.engine != 'tectonic':
        parser.error('--only-cached applies to Tectonic')
    out = args.output_dir.resolve()
    if out == ROOT:
        parser.error('Choose a separate output directory to preserve the supplied PDF snapshots.')
    out.mkdir(parents=True, exist_ok=True)
    source = (ROOT / 'main.tex').read_text()
    if r'\workingdrafttrue' not in source:
        parser.error('Expected the frozen main.tex to contain \\workingdrafttrue.')
    variants = ['draft', 'submission'] if args.variant == 'all' else [args.variant]
    for variant in variants:
        stem = 'main' if variant == 'draft' else 'submission-preview'
        text = source if variant == 'draft' else source.replace(r'\workingdrafttrue', r'\workingdraftfalse', 1)
        with tempfile.TemporaryDirectory(prefix=f'.{stem}-', dir=out) as temporary:
            work = Path(temporary)
            for name in ASSETS:
                shutil.copy2(ROOT / name, work / name)
            (work / f'{stem}.tex').write_text(text)
            if args.engine == 'tectonic':
                command = [executable, '--keep-intermediates', '--keep-logs', '--untrusted']
                if args.only_cached:
                    command.append('--only-cached')
                command.append(f'{stem}.tex')
            else:
                command = [executable, '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                           '-file-line-error', '-no-shell-escape', f'{stem}.tex']
            result = subprocess.run(command, cwd=work, capture_output=True, text=True)
            (out / f'{stem}.build.log').write_text(result.stdout + '\n' + result.stderr)
            for suffix in ['.pdf', '.tex', '.aux', '.bbl', '.blg', '.log']:
                path = work / f'{stem}{suffix}'
                if path.exists():
                    shutil.copy2(path, out / path.name)
            if result.returncode:
                raise SystemExit(f'Compilation failed: see {out / (stem + ".build.log")}')
        print(f'Built {out / (stem + ".pdf")}', flush=True)


if __name__ == '__main__':
    main()
