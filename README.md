# efcamdat-multitask-learner-profiling# efcamdat-multitask-learner-profiling

A multi-task Transformer model that predicts **CEFR proficiency level**, **L1 (native language)**, and **nationality** from learner English essays — with token-level SHAP explanations for each prediction head.

Built on RoBERTa and trained on the [EFCAMDAT](https://ef.com/assethub/cambridge) corpus of learner writing, using both the original (erroneous) text and its teacher-corrected version as a dual-input signal.

---

## What it does

| Task | Input | Output |
|------|-------|--------|
| CEFR classification | Learner essay | A1 → C2 proficiency level |
| L1 identification | Learner essay | Native language (e.g. Portuguese, Spanish) |
| Nationality prediction | Learner essay | Learner nationality code |

All three tasks are learned simultaneously via a **shared RoBERTa encoder** with three independent classification heads — no three separate models, just one.

---

## Architecture

```
"[RAW] I have 24 years [CORRECTED] I am 24 years old"
                        │
              RoBERTa encoder
                        │
                [CLS] representation
               /        |        \
        CEFR head    L1 head    Nat head
            │            │          │
          A1–C2       Portuguese   BR
```

The dual-text format (`[RAW] ... [CORRECTED] ...`) lets the model learn from **error patterns**, not just content. Special tokens `[RAW]` and `[CORRECTED]` are added to the tokenizer vocabulary.

Loss is the sum of three cross-entropy losses, one per head, backpropagated jointly.

---

## Project structure

```
efcamdat-multitask-learner-profiling/
├── Cambridge_Models.ipynb   # Main notebook (training, evaluation, SHAP)
├── requirements.txt
└── README.md
```

---

## Requirements

```
transformers>=4.40
datasets
torch>=2.0
scikit-learn
shap
pandas
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare data

Your CSV should have at minimum:

| Column | Description |
|--------|-------------|
| `text` | Raw learner essay |
| `text_corrected` | Teacher-corrected version |
| `cefr` | CEFR label string (A1, A2, …) |
| `l1` | Native language string |
| `nationality` | Nationality code |

### 2. Run the notebook

Open `Cambridge_Models.ipynb` in Google Colab (A100 GPU recommended). Mount your Drive, point the CSV path in Cell 3, and run all cells.

### 3. Inference on a new essay

```python
inputs = tokenizer(
    ["[RAW] " + essay_text + " [CORRECTED] " + corrected_text],
    truncation=True, padding='max_length', max_length=128, return_tensors='pt'
)

with torch.no_grad():
    outputs = multitask_model(**{k: v.to(device) for k, v in inputs.items()})

print(cefr_encoder.inverse_transform([outputs['cefr'].argmax().item()]))
print(l1_encoder.inverse_transform([outputs['l1'].argmax().item()]))
print(nat_encoder.inverse_transform([outputs['nat'].argmax().item()]))
```

### 4. Explain a prediction with SHAP

```python
explainer = shap.Explainer(make_shap_predictor(multitask_model, 'cefr', device), tokenizer)
shap_values = explainer(["[RAW] I have 24 years [CORRECTED] I am 24 years old"])
shap.plots.text(shap_values[0])
```

Red tokens push the prediction toward the predicted class; blue tokens push against it.

---

## Dataset

[EFCAMDAT (EF Cambridge Open Language Database)](https://ef.com/assethub/cambridge) — ~317,000 learner essays across CEFR levels A1–C1, with L1 and nationality metadata. Not included in this repo; access via the EF/Cambridge research portal.

---

## Results

> Training in progress. Evaluation metrics (accuracy, weighted F1, confusion matrices per head) will be added here once runs complete on the full dataset.

---

## Potential extensions

- Wrap in a FastAPI + React app: upload essay → predictions + SHAP highlight
- Add a feedback generation layer based on SHAP token importance
- Extend to other learner corpora (e.g. CasiMedicos, NUCLE)
- Experiment with `deberta-v3-base` as encoder for potential accuracy gains

---

## Author

Darragh — MSc Language Analysis and Processing, UPV/EHU  
[GitHub: VerbalAid](https://github.com/VerbalAid)
