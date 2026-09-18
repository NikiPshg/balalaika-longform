# Paper source

This folder contains the source and assets used for the supplied `main.pdf` and `submission-preview.pdf`. It is self-contained for LaTeX compilation and can be copied out of the code repository or uploaded to Overleaf.

| File | Purpose |
| --- | --- |
| `main.tex` | Complete manuscript: text, tables, captions and citations |
| `refs.bib` | Bibliography |
| `spconf.sty`, `IEEEbib.bst` | ICASSP formatting and bibliography style |
| `fig_wer_limited50_wide.pdf` | WER-by-length figure; not used in the current manuscript (its data is in Table 3), kept because `build.py` copies it |
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

`build.py` produces `main.pdf` and `submission-preview.pdf` from the single `main.tex` by flipping the `\workingdrafttrue` switch in a temporary copy. The manuscript currently contains no working-draft blocks, so the two PDFs have the same content. Edit `main.tex` when updating the paper; the dataset and code links are the `\datasetlink` and `\codelink` macros near the top of that file.
