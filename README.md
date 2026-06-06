# EF-CamDAT multitask learner profiling

Multi-task **RoBERTa** that predicts **CEFR**, **L1**, and **nationality** from learner English, using **`[RAW]` + `[CORRECTED]`** dual input. Trained on [EFCAMDAT](https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html)–style data (not shipped here).

## Repo layout

| Path | Role |
|------|------|
| `Cambridge_Models_final.ipynb` | Main notebook: train, eval, embeddings, SHAP demos |
| `cambridge_exp/` | CLI: `train_cli`, `eval_cli`, `shap_export`, topic families, metrics |
| `api/` | FastAPI inference + optional local small-LM + SHAP narrative |
| `scripts/compare_fair_eval.sh` | Same test essays for baseline vs balanced `eval_cli` |
| `tests/test_smoke.py` | Import smoke tests (`python -m unittest`) |
| `web/` | React (Vite) demo UI |
| `README_EXPERIMENTS.md` | Commands, GPU/OOM, resume, SHAP export, demo |
| `THESIS_EXPERIMENTS_PLAN.md` | Thesis experiment design + annotation taxonomy |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional: API + web demo
pip install -r requirements-api.txt
cd web && npm install && npm run dev   # in another terminal: uvicorn (see README_EXPERIMENTS.md)
```

Open **`Cambridge_Models_final.ipynb`** for the full interactive pipeline. For headless training:

```bash
export PYTHONPATH=.
python -m cambridge_exp.train_cli --csv efcamdat_full_with_corrected.csv --checkpoint-dir checkpoints/run1
```

Smoke tests (imports only; no data):

```bash
export PYTHONPATH=.
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Data

CSV columns (minimum): `text`, `text_corrected`, `cefr`, `l1`, `nationality`, `topic`. Large files and checkpoints are **gitignored**; obtain data via your EF/Cambridge access.

## Author

Darragh — MSc Language Analysis and Processing, UPV/EHU · [VerbalAid](https://github.com/VerbalAid)
