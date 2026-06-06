# L2 Profiler — Hugging Face Space (Docker)

FastAPI backend for EF-CamDAT L2 Profiler. Pairs with the Vite frontend on Vercel.

## Prepare

From the repo root:

```bash
chmod +x hf-space/prepare.sh
./hf-space/prepare.sh
```

This copies `api/`, `cambridge_exp/`, and (if present) `checkpoints/baseline_dual/best_model.pt`.

## Create the Space

1. [huggingface.co/new-space](https://huggingface.co/new-space) — SDK **Docker**, hardware **CPU basic**, name e.g. `l2-profiler`.
2. Clone the Space repo and copy everything from `hf-space/` into it.
3. Track the checkpoint with Git LFS:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes model/best_model.pt
git commit -m "add model checkpoint"
git push
```

## Secrets (Space settings → Variables)

| Variable | Value |
|----------|--------|
| `OPENROUTER_API_KEY` | From [openrouter.ai/keys](https://openrouter.ai/keys) |
| `CORS_ORIGINS` | `https://YOUR-APP.vercel.app,http://localhost:5173` |

Optional: `OPENROUTER_MODEL` (default `meta-llama/llama-3.1-8b-instruct`).

## Test

```bash
curl https://YOUR_USER-l2-profiler.hf.space/health
curl -X POST https://YOUR_USER-l2-profiler.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "In my office there are some desks.", "dual_mode": "raw_only"}'
```

## Notes

- Uses **gradient attribution** (`CAMBRIDGE_ATTR=gradient`) — no SHAP dependency in the container.
- Free tier sleeps after ~48h idle; first request may take 30–60s.
- Set `VITE_API_URL` on Vercel to this Space URL (no trailing slash).
