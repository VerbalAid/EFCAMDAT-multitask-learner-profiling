# Thesis experiments and analysis plan

This file records the agreed experiment list, rationale, and the **100-essay SHAP annotation** design. Order reflects priority unless noted.

**Implementation:** CLI training, SHAP export, topic families, API, and React UI are in the repo — see **`README_EXPERIMENTS.md`**.

---

## 1. Topic-controlled SHAP (highest priority)

**Goal:** Remove topical confounds (e.g. France vs Ibirapuera) so SHAP reflects linguistic signal, not “which place the essay is about.”

**Approach:**

- Pick **2–3 neutral topics** that appear **across all CEFR levels** in EF-CamDAT.
- Good candidates: *describing your workplace*, *writing a letter* (validate against the corpus that these bands are well represented at A1–C1).
- **Restrict SHAP** (and any qualitative readout) to essays matching those topics only.

**Outcome:** Clean signal for interpretation; avoids misleading attributions driven purely by geography or culture in the text.

---

## 2. Raw-text ablation

**Goal:** Quantify how much the model uses **learner raw text** vs **teacher-corrected** text.

**Approach:**

- Retrain with **`dual_text = df['text']` only** (no `[RAW] … [CORRECTED] …` pairing); keep **everything else identical** (architecture, heads, splits, hyperparameters unless you intentionally match compute).

**Interpretation:**

- If **CEFR (and related heads) drop clearly** → corrected text contributes **real** supervised signal.
- If performance **barely moves** → length, vocabulary, and other surface cues may be doing most of the work — **also a strong thesis finding**.

---

## 3. Balanced undersampling

**Goal:** Mitigate class imbalance before spending long training runs.

**Approach:**

- Apply **balanced undersampling (or equivalent balancing)** **before** the main multi-epoch training — e.g. before any planned **5-epoch** run.
- **Do not** train 5 epochs on heavily imbalanced data if imbalance is already known to hurt or skew analysis.

---

## 4. Five epochs + early stopping (lower priority)

**Status:** Defer until topic control, ablation, and balancing are in place.

**Rationale:** The **3-epoch model is the baseline**. Extra GPU time on longer runs is low value until the **data slice and training regime** are aligned with the research questions.

---

## 5. FastAPI + React demo (build last)

**Goal:** Thesis defence demo for non-technical examiners.

**Stack:** **FastAPI** backend, **React** frontend.

**Highlight:** A **small local instruct model** (or similar) as an “explanation layer” that turns model outputs + SHAP attributions into plain language — the piece that makes what you built **interpretable** to a general audience.

---

## 6. Core contribution: ~100 essays, structured SHAP analysis

**Goal:** A **quantitative** link between SHAP attributions and **linguistic explanation types**, comparable to SLA / contrastive-analysis expectations.

### Design

- Run SHAP on a fixed set of **~100 essays** (after topic control if applicable).
- For each essay, consider **top SHAP tokens** (per task/head as defined in your protocol).
- **Annotate** each such token into **one** of four categories:

| Label | Definition |
|--------|------------|
| **`transfer_error`** | Aligns with a **known L1 contrastive-analysis** prediction (predictable L1 influence). |
| **`named_entity`** | **Geographic** or **cultural** reference (potential topical leakage). |
| **`register_marker`** | **Vocabulary or syntax** that signals level/register rather than a specific transfer story. |
| **`noise`** | **No clear linguistic explanation** in your coding guide. |

### Thesis-facing result

Aggregate over essays:

- **X%** of strong SHAP mass on the **L1 head** (or whichever head you foreground) falls in **`transfer_error`** vs **`named_entity`** vs **`register_marker`** vs **`noise`**.
- Compare **X** (and the split) to **SLA literature** predictions.

That percentage — not a single accuracy number in isolation — is the **central quantitative claim** the rest of the thesis supports.

---

## Suggested implementation order

1. Topic-controlled slice definition + filter in notebook or dataloader.  
2. Balanced undersample → train/eval pipeline update.  
3. Raw-text ablation run vs baseline dual-text run.  
4. 100-essay SHAP + annotation spreadsheet / tool + inter-rater note if applicable.  
5. Optional: 5 epochs + early stopping on the cleaned regime.  
6. Last: FastAPI + React + explanation API for defence.

---

## Open methods notes (fill as you lock the protocol)

- Exact **topic labels or metadata fields** in EF-CamDAT used for filtering.  
- Definition of **“top SHAP token”** per essay (top-k per head? threshold? subword merge rule?).  
- **Annotation guidelines** (1–2 pages) + pilot on 10 essays before scaling to 100.
