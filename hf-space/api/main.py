"""
FastAPI service: multitask predictions + optional local small-LM explanations.

From repo root (parent of `api/` and `cambridge_exp/`):

  export CAMBRIDGE_CHECKPOINT=checkpoints/baseline_dual/best_model.pt
  # optional: export CAMBRIDGE_NARRATOR_MODEL=Qwen/Qwen2.5-0.5B-Instruct
  PYTHONPATH=. uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder

from cambridge_exp.data_pipeline import build_model_input_text, make_tokenizer
from cambridge_exp.model import MultiTaskRoberta

from .error_analysis import analyze_errors
from .attribution import head_token_attribution
from .local_narrate import hf_generate_chat
from .ocr import transcribe_upload
from .ollama_narrate import SUMMARY_MODE, narrate_attribution_per_head
from .llm_client import llm_available
from .sentence_shap import aggregate_to_sentences, build_head_comparison, narrate_shap_per_head

app = FastAPI(title="EF-CamDAT L2 Profiler", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.environ.get(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if o.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_DEVICE = torch.device(
    "cpu"
    if os.environ.get("CAMBRIDGE_DEVICE", "").strip().lower() == "cpu"
    or os.environ.get("SPACE_ID")
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
_MODEL = None
_TOKENIZER = None
_CEFR_ENC = _L1_ENC = _NAT_ENC = None
_MAXLEN = 128
_REPO_ROOT = Path(__file__).resolve().parent.parent
_WEB_DIST = _REPO_ROOT / "web" / "dist"
_DEFAULT_CHECKPOINT = _REPO_ROOT / "checkpoints" / "baseline_dual" / "best_model.pt"
_HF_CHECKPOINT = Path("/app/model/best_model.pt")


def _load_dotenv() -> None:
    env_file = _REPO_ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


def _resolve_checkpoint() -> Path:
    raw = os.environ.get("CAMBRIDGE_CHECKPOINT", "").strip()
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else _REPO_ROOT / p)
    candidates.extend(
        [
            _HF_CHECKPOINT,
            _DEFAULT_CHECKPOINT,
            _REPO_ROOT / "checkpoints" / "multitask_model" / "best_model.pt",
            _REPO_ROOT / "checkpoints" / "balanced_cefr" / "best_model.pt",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError(
        "Set CAMBRIDGE_CHECKPOINT to best_model.pt from training "
        f"(tried {candidates[0] if candidates else _DEFAULT_CHECKPOINT})."
    )


class PredictRequest(BaseModel):
    text: str = Field(..., description="Learner raw essay")
    text_corrected: Optional[str] = Field(None, description="Teacher text; if omitted, same as raw")
    dual_mode: str = Field("dual", description="'dual' or 'raw_only' — must match checkpoint")


class ProbItem(BaseModel):
    label: str
    prob: float


class PredictResponse(BaseModel):
    cefr: str
    l1: str
    nationality: str
    probs_top_cefr: List[ProbItem]
    explanation: Optional[str] = None


class PredictShapExplainRequest(BaseModel):
    text: str
    text_corrected: Optional[str] = None
    dual_mode: str = "dual"
    heads: List[str] = Field(
        default_factory=lambda: ["cefr", "l1", "nat"],
        description="Which heads to attribute (SHAP per head by default)",
    )
    summarize: bool = Field(True, description="If false, omit template summary per head")
    ollama_narrate: bool = Field(
        False,
        description="If true, add per-head one-sentence summary from attribution data",
    )


class TokenHighlight(BaseModel):
    text: str
    start: int
    end: int
    attribution: float
    direction: str


class SentenceAttribution(BaseModel):
    sentence: str
    attribution: float
    signed_attribution: float
    direction: str
    signed_mass: float
    toward_mass: float = 0.0
    tokens: List[TokenHighlight] = Field(default_factory=list)


class PredictShapExplainResponse(BaseModel):
    cefr: str
    l1: str
    nationality: str
    probs_top_cefr: List[ProbItem]
    sentence_shap: Dict[str, List[SentenceAttribution]]
    narrative: Optional[Dict[str, str]] = Field(
        None,
        description="Per-head template summary from sentence SHAP (keys: cefr, l1, nat)",
    )
    head_comparison: Optional[str] = Field(
        None,
        description="L1 vs nationality top-sentence comparison when both heads ran",
    )
    ollama_narrative: Optional[Dict[str, str]] = Field(
        None,
        description="Per-head one-sentence summary (keys: cefr, l1, nat)",
    )
    summary_mode: Optional[str] = Field(
        None,
        description="How summaries were produced: deterministic_v1, or ollama_v1 if CAMBRIDGE_OLLAMA_SUMMARY=1",
    )
    attribution_methods: Optional[Dict[str, str]] = Field(
        None,
        description="Per-head method: shap, gradient, or integrated_gradients",
    )


class ErrorItem(BaseModel):
    span: str
    correction: str
    type: str
    explanation: str


class AttributionContext(BaseModel):
    head: str = Field(description="cefr, l1, or nat")
    sentence: str
    toward_mass: float = 0.0
    tokens: Optional[List[str]] = Field(None, description="Top attributed token strings")


class ErrorsRequest(BaseModel):
    text: str
    cefr: str
    l1: str
    attributions: Optional[List[AttributionContext]] = Field(
        None,
        description="Sentence attributions from SHAP (all heads) for context",
    )


class ErrorsResponse(BaseModel):
    errors: List[ErrorItem]


class UploadEssayResponse(BaseModel):
    text: str
    method: str = Field(description="text | pdf_text | ocr")
    filename: str


def _load():
    global _MODEL, _TOKENIZER, _CEFR_ENC, _L1_ENC, _NAT_ENC, _MAXLEN
    if _MODEL is not None:
        return
    ckpt_path = _resolve_checkpoint()

    ckpt = torch.load(ckpt_path, map_location=_DEVICE, weights_only=False)
    _MAXLEN = int(ckpt.get("max_length", 128))
    model_name = ckpt.get("model_name", "roberta-base")

    _CEFR_ENC, _L1_ENC, _NAT_ENC = LabelEncoder(), LabelEncoder(), LabelEncoder()
    _CEFR_ENC.fit(ckpt["cefr_classes"])
    _L1_ENC.fit(ckpt["l1_classes"])
    _NAT_ENC.fit(ckpt["nat_classes"])

    _TOKENIZER = make_tokenizer(model_name)
    n_c, n_l, n_n = len(_CEFR_ENC.classes_), len(_L1_ENC.classes_), len(_NAT_ENC.classes_)
    _MODEL = MultiTaskRoberta(model_name, n_c, n_l, n_n).to(_DEVICE)
    _MODEL.encoder.resize_token_embeddings(len(_TOKENIZER))
    _MODEL.load_state_dict(ckpt["model_state_dict"])
    _MODEL.eval()


def _local_explain(payload: str) -> Optional[str]:
    system = (
        "You help thesis readers interpret a multilingual learner profiling model. "
        "Be concise and avoid overclaiming. Use British English spelling."
    )
    user = (
        "Given this JSON with predicted CEFR, L1, nationality and a short text preview, "
        "write 2–4 sentences in plain English on what the outputs mean and how to interpret them.\n\n"
        + payload
    )
    return hf_generate_chat(system, user, max_new_tokens=256)


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": _MODEL is not None,
        "attribution_version": "shap_primary_v6",
        "summary_mode": SUMMARY_MODE,
        "llm_available": llm_available(),
        "device": str(_DEVICE),
        "attribution_default": os.environ.get("CAMBRIDGE_ATTR", "shap"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    try:
        _load()
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e

    if body.dual_mode not in ("dual", "raw_only"):
        raise HTTPException(400, "dual_mode must be 'dual' or 'raw_only'")

    cor = body.text_corrected if body.text_corrected is not None else body.text
    row = {"text": body.text, "text_corrected": cor}
    s = build_model_input_text(row, body.dual_mode)

    assert _TOKENIZER is not None and _MODEL is not None
    enc = _TOKENIZER(
        s,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=_MAXLEN,
    )
    enc = {k: v.to(_DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = _MODEL(**enc)

    def topk(logits, enco, k=3):
        p = F.softmax(logits[0], dim=-1)
        vals, inds = torch.topk(p, k=min(k, p.numel()))
        return [
            ProbItem(label=enco.inverse_transform([int(i)])[0], prob=float(v))
            for v, i in zip(vals, inds)
        ]

    ci = int(out["cefr"].argmax(-1).item())
    li = int(out["l1"].argmax(-1).item())
    ni = int(out["nat"].argmax(-1).item())
    pred = PredictResponse(
        cefr=_CEFR_ENC.inverse_transform([ci])[0],
        l1=_L1_ENC.inverse_transform([li])[0],
        nationality=_NAT_ENC.inverse_transform([ni])[0],
        probs_top_cefr=topk(out["cefr"], _CEFR_ENC),
    )

    expl_payload = (
        f'{{"cefr":"{pred.cefr}","l1":"{pred.l1}","nationality":"{pred.nationality}",'
        f'"text_preview":{repr(s[:500])}}}'
    )
    if os.environ.get("CAMBRIDGE_PREDICT_NARRATIVE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        pred.explanation = _local_explain(expl_payload)
    return pred


@app.post("/predict/shap_explain", response_model=PredictShapExplainResponse)
def predict_shap_explain(body: PredictShapExplainRequest):
    """Forward pass + per-head SHAP attribution + optional template summary."""
    try:
        _load()
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e

    if body.dual_mode not in ("dual", "raw_only"):
        raise HTTPException(400, "dual_mode must be 'dual' or 'raw_only'")

    allowed = {"cefr", "l1", "nat"}
    heads = [h for h in body.heads if h in allowed]
    if not heads:
        raise HTTPException(400, "heads must include at least one of cefr, l1, nat")

    cor = body.text_corrected if body.text_corrected is not None else body.text
    row = {"text": body.text, "text_corrected": cor}
    s = build_model_input_text(row, body.dual_mode)

    assert _TOKENIZER is not None and _MODEL is not None
    enc = _TOKENIZER(
        s,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=_MAXLEN,
    )
    enc_d = {k: v.to(_DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = _MODEL(**enc_d)

    def topk(logits, enco, k=3):
        p = F.softmax(logits[0], dim=-1)
        vals, inds = torch.topk(p, k=min(k, p.numel()))
        return [
            ProbItem(label=enco.inverse_transform([int(i)])[0], prob=float(v))
            for v, i in zip(vals, inds)
        ]

    ci = int(out["cefr"].argmax(-1).item())
    li = int(out["l1"].argmax(-1).item())
    ni = int(out["nat"].argmax(-1).item())
    preds = {
        "cefr": _CEFR_ENC.inverse_transform([ci])[0],
        "l1": _L1_ENC.inverse_transform([li])[0],
        "nationality": _NAT_ENC.inverse_transform([ni])[0],
    }
    pred_i = {"cefr": ci, "l1": li, "nat": ni}

    sentence_shap: Dict[str, List[SentenceAttribution]] = {}
    attribution_methods: Dict[str, str] = {}
    for hk in heads:
        scored, method = head_token_attribution(
            s,
            _MODEL,
            _TOKENIZER,
            _DEVICE,
            hk,
            pred_i[hk],
            _MAXLEN,
        )
        attribution_methods[hk] = method
        rows = aggregate_to_sentences(body.text, scored)
        sentence_shap[hk] = [SentenceAttribution(**r) for r in rows]

    per_head_rows = {k: [r.model_dump() for r in v] for k, v in sentence_shap.items()}
    narrative = narrate_shap_per_head(preds, per_head_rows, heads) if body.summarize else None
    comparison = build_head_comparison(per_head_rows, heads)

    ollama_narrative = None
    summary_mode = None
    if body.ollama_narrate:
        ollama_narrative = narrate_attribution_per_head(
            preds, per_head_rows, heads, essay_text=body.text
        )
        if ollama_narrative:
            summary_mode = SUMMARY_MODE

    return PredictShapExplainResponse(
        cefr=preds["cefr"],
        l1=preds["l1"],
        nationality=preds["nationality"],
        probs_top_cefr=topk(out["cefr"], _CEFR_ENC),
        sentence_shap=sentence_shap,
        narrative=narrative,
        head_comparison=comparison,
        ollama_narrative=ollama_narrative,
        summary_mode=summary_mode,
        attribution_methods=attribution_methods,
    )


@app.post("/errors", response_model=ErrorsResponse)
def errors(body: ErrorsRequest):
    """LLM error analysis on the full essay (OpenRouter in production; optional Ollama locally)."""
    try:
        attr = [a.model_dump() for a in body.attributions] if body.attributions else None
        rows = analyze_errors(body.text, body.cefr, body.l1, attr)
    except Exception as e:
        raise HTTPException(502, f"Error analysis failed: {e}") from e
    return ErrorsResponse(errors=[ErrorItem(**e) for e in rows])


@app.post("/upload/essay", response_model=UploadEssayResponse)
async def upload_essay(file: UploadFile = File(...)):
    """Extract learner text from .txt, PDF (text layer or OCR), or image scan."""
    data = await file.read()
    try:
        out = transcribe_upload(file.filename or "upload", file.content_type, data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        raise HTTPException(502, f"OCR failed: {e}") from e
    return UploadEssayResponse(
        text=out["text"],
        method=out["method"],
        filename=file.filename or "upload",
    )


@app.get("/")
def root():
    """Web UI when built; otherwise point to Vite dev server."""
    index = _WEB_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {
        "message": "EF-CamDAT API is running.",
        "ui": "Start the web UI: cd web && npm run dev  →  http://127.0.0.1:5173",
        "health": "/health",
        "docs": "/docs",
    }


if _WEB_DIST.is_dir():
    assets = _WEB_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")
