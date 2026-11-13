# Handover — state on 2026-11-13

For whoever (human or AI) continues this project. Final submission deadline: **19 Nov 2026**.

## Done
- Experiments, statistics, Kaggle re-score: all complete. `./reproduce.sh` rebuilds and checks everything
  (6 checks, ~90 s, laptop only). Every number lives in `results.md`.
- Proposal: `proposal/UCS532_Proposal_P49.pdf` (submitted).
- Report: `report/main.pdf`, CVPR format, 4 pages + references. Source `report/main.tex`;
  rebuild with `cd report && tectonic -X compile main.tex`. Figures come from `src/make_figures.py`
  (writes `proposal/figs/{pdf,png}/`).
- Notebooks: `notebooks/01–06`, executed, print the real source from `src/`.
- Viva pack: `docs/VIVA.md` (30 questions with file:line pointers). Learning doc: `docs/Project_Course.pdf`.

## Left for the author
1. **AI declaration** — last line of `report/main.tex` says `[to be completed by the author]`. Fill it
   honestly (Rule 4 of the course document: undeclared AI use = plagiarism). Then recompile.
2. **Push and submit**: `git push`; submit `report/main.pdf` and the repo link.
3. **Viva practice** with `docs/VIVA.md` and the notebooks.

## Optional (not required for the deadline)
- 5th model: distil → prune → quantise (closes P49's "combine all three"). Needs GPU; Kaggle T4 run time
  not yet verified — do a smoke run first (`docs/GPU_RUN.md` describes the Kaggle setup).
- Intermediate stages for routes B and C (ternary-only, 4-bit-only, prune-only, no fine-tuning).

## Rules that kept this project correct
See `CLAUDE.md` / `AGENTS.md`: compute every number, check it a second way, quote primary sources, never
state a p-value without the right test. Commits are plain, authored by the user, no AI attribution lines.
