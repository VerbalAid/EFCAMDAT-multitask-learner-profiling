---
title: EF-CamDAT L2 Profiler
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# EF-CamDAT L2 Profiler API

FastAPI backend: multitask CEFR / L1 / nationality predictions with gradient attribution and OpenRouter summaries.

**Endpoints:** `GET /health` · `POST /predict` · `POST /predict/shap_explain`

Space URL: https://darragh11dec-l2-profiler.hf.space

## Weights (separate Model repo)

Checkpoint: **`darragh11dec/weights`** — downloaded at Docker build time (must be **Public**).  
Deploy with `bash scripts/upload_hf_space.sh` from the Cambridge repo root.

## Secrets (Settings → Variables)

| Variable | Required |
|----------|----------|
| `OPENROUTER_API_KEY` | Yes — summaries |
| `CORS_ORIGINS` | Yes — e.g. `https://your-app.vercel.app,http://localhost:5173` |

Optional: `OPENROUTER_MODEL` (default `meta-llama/llama-3.1-8b-instruct`).
