# Paper source

This folder contains the source and assets used for the supplied `main.pdf` and `submission-preview.pdf`. It is self-contained for LaTeX compilation and can be copied out of the code repository or uploaded to Overleaf.

| File | Purpose |
| --- | --- |
| `main.tex` | Complete manuscript: text, tables, captions and citations |
| `refs.bib` | Bibliography |
| `spconf.sty`, `IEEEbib.bst` | ICASSP formatting and bibliography style |
| `fig_wer_limited50_wide.pdf` | WER figure |
| `fig_quality_limited50.pdf` | Windowed quality and speaker-similarity figure |
| `main.pdf` | Current working-draft PDF snapshot |
| `submission-preview.pdf` | Current PDF snapshot with working-draft blocks hidden |
| `build.py` | Rebuild both variants from the single manuscript source |

## Build

The supplied PDFs were compiled with **Tectonic 0.16.9**. With Python 3 and Tectonic installed, run from this folder:

```bash
python build.py
```

This compiles the bibliography and both PDF variants into `build/`. It leaves the supplied snapshots in place. Tectonic downloads its TeX resources on first use; `--only-cached` disables downloads when those resources are already present. Use `--compiler /path/to/tectonic` if the executable is not on `PATH`.

A conventional TeX Live installation with `latexmk` can also be used:

```bash
python build.py --engine latexmk
```

For Overleaf, upload this folder's files and select `main.tex` as the main document. Standard pdfLaTeX can compile it, although layout can differ slightly from the Tectonic snapshots because of engine and package versions.

## PDF variants

`main.tex` contains `\workingdrafttrue`. The main snapshot therefore retains the working-draft blocks. The build script creates the submission preview by replacing that switch with `\workingdraftfalse` in a temporary copy. Both variants come from the same manuscript; edit `main.tex` when updating the paper.

These files preserve the current manuscript exactly, including its anonymous author/affiliation fields and the dataset URL placeholder. The preview switch only controls the working-draft blocks. It does not fill those fields or update the paper text.
