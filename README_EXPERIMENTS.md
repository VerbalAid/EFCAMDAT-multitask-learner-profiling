# Thesis experiment pipeline (implemented)

This repo adds a **CLI training pipeline**, **topic-controlled SHAP export** for the 100-essay annotation study, and a **FastAPI + React** defence demo. The Jupyter notebook remains your interactive baseline.

## Layout

| Piece | Location |
|-------|----------|
| Train (dual / raw ablation, balance, early stopping) | `python -m cambridge_exp.train_cli` |
| **TF-IDF + LR baseline** (same split, no GPU) | `python -m cambridge_exp.tfidf_baseline_cli` |
| **Single-task RoBERTa** (`--heads cefr\|l1\|nat`) | `python -m cambridge_exp.train_cli --heads l1` |
| **Evaluate** a saved checkpoint (acc + reports) | `python -m cambridge_exp.eval_cli` |
| Shared train/val accuracy helper | `cambridge_exp/metrics.py` |
| **Migrate** old notebook `.pt` (add metadata keys) | `python -m cambridge_exp.migrate_checkpoint` |
| Topic families (cross-level prompt groups) | `cambridge_exp/topic_families.json` |
| Export SHAP rows for annotation | `python -m cambridge_exp.shap_export` |
| Aggregate labels | `python -m cambridge_exp.aggregate_annotations` |
| REST API | `api/main.py` |
| Web UI | `web/` (Vite + React) |
| Tail a background train log | `bash scripts/watch_training.sh` |

Set **`PYTHONPATH`** to the project root (the directory that contains `cambridge_exp/`), or run all commands from that directory with:

```bash
export PYTHONPATH=.
```

## While a long training job runs

- **Watch the log** (default path matches resumed baseline):

  ```bash
  bash scripts/watch_training.sh
  # or: tail -f checkpoints/baseline_dual/train_resume.log
  ```

- **Evaluate an older checkpoint** (e.g. notebook export) after migrating metadata:

  ```bash
  python -m cambridge_exp.migrate_checkpoint path/to/best_model.pt --out checkpoints/migrated.pt
  python -m cambridge_exp.eval_cli --checkpoint checkpoints/migrated.pt --csv efcamdat_full_with_corrected.csv
  ```

- **Prep the demo** (does not need the training process): `pip install -r requirements-api.txt`, then `cd web && npm install` so the React app is ready when the API has a checkpoint.

## CUDA out of memory

- **Colab vs your laptop:** Colab GPUs are often **15–40 GiB** and **exclusive** to your session. A **7–8 GiB** laptop GPU with **other apps** using half of it behaves totally differently — same code, not enough free VRAM.
- **Other processes** often use most of the GPU (your log showed ~5 GiB already taken on an ~8 GiB card). List them:

  ```bash
  nvidia-smi
  ```

  Note the **PID**s in the process table, then stop **your** jobs only, for example:

  ```bash
  kill 398804 396791    # replace with PIDs from nvidia-smi; use kill -9 only if they ignore SIGTERM
  ```

  Also quit other Jupyter kernels, ComfyUI, games, browsers with WebGPU, etc.

- **Allocator hint** (can help fragmentation): `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

- **Stopped during `[val]`** (e.g. 17%): the process exited (OOM, **OOM killer**, Ctrl+C, or crash). Check the terminal for `OutOfMemoryError`; on Linux run `dmesg -T | tail -30` for “Out of memory: Killed process”. Lower eval memory: `--eval-batch-size 8` (or `4`). Then **resume** from the last good checkpoint:

  ```bash
  PYTHONPATH=. python -m cambridge_exp.train_cli --csv efcamdat_full_with_corrected.csv \
    --checkpoint-dir checkpoints/baseline_dual --resume --eval-batch-size 8
  ```
- **Defaults** are tuned for tight VRAM: `--batch-size 16`, `--eval-batch-size 32`, `--grad-accum-steps 1`. If you still OOM, try `--batch-size 8 --eval-batch-size 8` or train on CPU with `--cpu` (slow).
- To approximate the old **64 × micro-batch** behaviour when you have headroom: `--batch-size 16 --grad-accum-steps 4`.

## 1. Training

Requires `efcamdat_full_with_corrected.csv` (or pass `--csv`).

**Baseline (dual-text, same spirit as the notebook):**

```bash
PYTHONPATH=. python -m cambridge_exp.train_cli \
  --csv efcamdat_full_with_corrected.csv \
  --checkpoint-dir checkpoints/baseline_dual
```

**Raw-text ablation** (`dual_text` = learner `text` only):

```bash
PYTHONPATH=. python -m cambridge_exp.train_cli \
  --dual-mode raw_only \
  --checkpoint-dir checkpoints/ablation_raw_only
```

**Balanced CEFR undersampling** (per-level counts matched to the smallest band *before* the stratified split):

```bash
PYTHONPATH=. python -m cambridge_exp.train_cli \
  --balance-cefr \
  --checkpoint-dir checkpoints/balanced_cefr
```

**Comparing balanced vs full-data accuracy:** `--balance-cefr` undersamples *before* the split, so the default `eval_cli` test set differs from the full-data run. To compare fairly on the **same essays**, export test indices from the **unbalanced** pipeline once, then point **both** checkpoints at that file:

```bash
# Once: same CSV / --dual-mode / --seed / topic flags as your baseline training (defaults match train_cli)
PYTHONPATH=. python -m cambridge_exp.export_test_split \
  --csv efcamdat_full_with_corrected.csv \
  --out splits/test_indices.json

PYTHONPATH=. python -m cambridge_exp.eval_cli -c checkpoints/baseline_dual/best_model.pt \
  --csv efcamdat_full_with_corrected.csv --test-indices splits/test_indices.json
PYTHONPATH=. python -m cambridge_exp.eval_cli -c checkpoints/balanced_cefr/best_model.pt \
  --csv efcamdat_full_with_corrected.csv --test-indices splits/test_indices.json
```

Or use **`scripts/compare_fair_eval.sh`** (creates `splits/test_indices.json` if missing):

```bash
chmod +x scripts/compare_fair_eval.sh   # once
export CSV="$PWD/efcamdat_full_with_corrected.csv"
./scripts/compare_fair_eval.sh checkpoints/baseline_dual/best_model.pt checkpoints/balanced_cefr/best_model.pt
```

`eval_cli` checks that checkpoint metadata (`dual_mode`, topic filters) matches the split JSON and that the prepared row count is unchanged.

## 1b. Baselines (run before writing up results)

**TF-IDF + logistic regression** — same stratified split as RoBERTa, runs in under a minute on CPU:

```bash
PYTHONPATH=. python -m cambridge_exp.tfidf_baseline_cli \
  --csv efcamdat_full_with_corrected.csv \
  --out-json results/tfidf_raw.json

# Match dual-input RoBERTa (concatenate learner + corrected text):
PYTHONPATH=. python -m cambridge_exp.tfidf_baseline_cli \
  --text-source dual \
  --out-json results/tfidf_dual.json

# Fair comparison on the same test essays as a saved checkpoint:
PYTHONPATH=. python -m cambridge_exp.tfidf_baseline_cli \
  --test-indices splits/test_indices.json \
  --out-json results/tfidf_fair.json
```

Reports include **accuracy** and **macro-F1** per head (lead with macro-F1 for L1/nationality).

**Single-task RoBERTa** — one head, same encoder; compare L1 accuracy to the multi-task L1 head:

```bash
PYTHONPATH=. python -m cambridge_exp.train_cli \
  --heads l1 \
  --checkpoint-dir checkpoints/single_task_l1

PYTHONPATH=. python -m cambridge_exp.eval_cli \
  -c checkpoints/single_task_l1/best_model.pt \
  --test-indices splits/test_indices.json
```

Repeat with `--heads cefr` and `--heads nat` for the full single-task vs multi-task table.

**Train only on one cross-level topic family** (small subset — for specialised experiments):

```bash
PYTHONPATH=. python -m cambridge_exp.train_cli \
  --train-topic-family workplace_across_levels \
  --checkpoint-dir checkpoints/topic_workplace_only
```

**Longer run + early stopping** (already enabled; tune patience / epochs):

```bash
PYTHONPATH=. python -m cambridge_exp.train_cli \
  --epochs 5 \
  --early-stop-patience 3 \
  --checkpoint-dir checkpoints/ep5_es
```

**Run several experiments in one go** (`scripts/run_experiment_matrix.sh`): by default passes `--eval-batch-size 8` for smaller GPUs. Override with `export TRAIN_FLAGS=""` or `export TRAIN_FLAGS="--eval-batch-size 16"`.

Checkpoints include **`cefr_classes`**, **`l1_classes`**, **`nat_classes`**, **`dual_mode`**, and tokenizer extras so the API and SHAP export load without the notebook session.

### Notebook checkpoints without metadata

If `best_model.pt` comes from the notebook and lacks `cefr_classes`, add keys once:

```python
import torch
p = "checkpoints/multitask_model/best_model.pt"
ckpt = torch.load(p, map_location="cpu", weights_only=False)
ckpt.setdefault("cefr_classes", ["A1", "A2", "B1", "B2", "C1"])
ckpt.setdefault("l1_classes", ["Arabic", "French", "German", "Italian", "Japanese", "Mandarin", "Portuguese", "Russian", "Spanish", "Turkish"])
ckpt.setdefault("nat_classes", ["br", "cn", "de", "fr", "it", "jp", "mx", "ru", "sa", "tr", "tw"])
ckpt.setdefault("dual_mode", "dual")
ckpt.setdefault("max_length", 128)
ckpt.setdefault("model_name", "roberta-base")
torch.save(ckpt, p)
```

## 2. Topic-controlled SHAP → annotation CSV

EF-CamDAT maps **one prompt topic to one CEFR band**. `topic_families.json` groups **semantically related** prompts across bands (workplace, correspondence, etc.) so you can restrict analysis to a **cross-level** slice.

Example: **100 test essays**, **workplace family**, **L1 head**, top 15 tokens each:

```bash
PYTHONPATH=. python -m cambridge_exp.shap_export \
  --checkpoint checkpoints/baseline_dual/best_model.pt \
  --csv efcamdat_full_with_corrected.csv \
  --n 100 \
  --topic-family workplace_across_levels \
  --split test \
  --heads l1 \
  --output annotation/shap_top_tokens.csv
```

Optional: `--exclude-place-heavy-topics` drops a few prompts that often elicit heavy geography (heuristic list in `cambridge_exp/topics.py`).

Fill **`annotation_label`** using `annotation/ANNOTATION_README.md`, then:

```bash
PYTHONPATH=. python -m cambridge_exp.aggregate_annotations annotation/shap_top_tokens.csv --head l1
```

## 3. FastAPI + React demo

**API** (from repo root):

```bash
pip install -r requirements.txt -r requirements-api.txt
export CAMBRIDGE_CHECKPOINT=checkpoints/baseline_dual/best_model.pt
# optional: small local narrator (default Qwen2.5-0.5B-Instruct; first call downloads weights)
# export OLLAMA_HOST=http://127.0.0.1:11434
# export CAMBRIDGE_OLLAMA_MODEL=mistral   # required for POST /errors
PYTHONPATH=. uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Web** (separate terminal):

```bash
cd web
npm install
npm run dev
```

Open the Vite URL (default `http://127.0.0.1:5173`). The dev server proxies `/predict`, `/predict/shap_explain`, `/upload/essay`, `/errors`, and `/health` to port 8000. The web UI includes **Predict**, **Explain attributions** (head-specific gradient attribution, fast), and **error analysis** on high-signal sentences (Ollama).

### Smoke tests

From repo root:

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py' -v
```

**Upload essay (OCR)** — `POST /upload/essay` multipart field `file` (.txt, PDF, or image). PDFs use embedded text when available; scanned pages/images use Tesseract:

```bash
# Fedora
sudo dnf install tesseract tesseract-langpack-eng
pip install -r requirements-api.txt

curl -s -X POST http://127.0.0.1:8000/upload/essay \
  -F "file=@scan.jpg"
```

Optional: `TESSERACT_CMD`, `CAMBRIDGE_OCR_LANG` (default `eng`).

**Gradient attribution + template summary** (fast; no SHAP). `POST /predict/shap_explain` with the same `text` / `dual_mode` body as `/predict`, plus optional `heads` (default `["cefr","l1","nat"]`) and `summarize` (default **true** — deterministic one-liner from sentence attributions). Tick **Summary** in the UI (`ollama_narrate: true`) for a one-sentence LLM restatement (Groq in production; optional local Ollama). Uses embedding×gradient per head so CEFR / L1 / nationality attributions differ.

```bash
curl -s -X POST http://127.0.0.1:8000/predict/shap_explain \
  -H "Content-Type: application/json" \
  -d '{"text":"...","dual_mode":"raw_only","heads":["cefr","l1"],"summarize":true,"ollama_narrate":true}'
```

Check `/health` returns `"attribution_version": "shap_primary_v6"`, `"summary_mode": "openrouter_v1"`, and `"llm_available": true` when `OPENROUTER_API_KEY` or `CAMBRIDGE_USE_OLLAMA=1` is set.

**LLM summaries (local dev)**

| Mode | Env |
|------|-----|
| OpenRouter (production) | `OPENROUTER_API_KEY=...` |
| Local Ollama | `CAMBRIDGE_USE_OLLAMA=1` + `ollama serve` |
| No key | Template narrative only; deterministic fallback if Summary is on |

**Error analysis** (`POST /errors`) — same LLM transport as summaries:

```bash
curl -s -X POST http://127.0.0.1:8000/errors \
  -H "Content-Type: application/json" \
  -d '{"text":"He go to school.","cefr":"A1","l1":"Portuguese","sentences":["He go to school."]}'
```

**Production deploy (Vercel + Hugging Face + OpenRouter)**

| Layer | Host | Notes |
|-------|------|-------|
| Frontend | Vercel (`web/`) | Set `VITE_API_URL` to your HF Space URL |
| Backend | HF Space (`hf-space/`) | Docker, CPU, gradient attr; see `hf-space/README.md` |
| Summaries | OpenRouter | `OPENROUTER_API_KEY` in HF Space secrets; default model `meta-llama/llama-3.1-8b-instruct` |

```bash
./hf-space/prepare.sh   # sync api/, cambridge_exp/, checkpoint
# Push hf-space/ to your HF Space repo (Git LFS for *.pt)
# Vercel: root directory web/, env VITE_API_URL=https://USER-l2-profiler.hf.space
# HF Space: CORS_ORIGINS=https://your-app.vercel.app,http://localhost:5173
```

**Production build** (output in `web/dist/`):

```bash
cd web
npm install   # if not already
npm run build
```

For a **local smoke test** of the built files only: `npm run preview` (opens another port; it does **not** proxy to FastAPI — use `npm run dev` for the full demo, or put `dist/` behind nginx and reverse-proxy `/predict` and `/health` to the API).

## EF-CamDAT note on topics

There is **no single topic string** that appears at every CEFR level. Cross-level “topic control” is implemented as **curated families** of official prompt titles in `topic_families.json`. Edit that file if your thesis narrows to different prompt groups.
