# SHAP token annotation (thesis core contribution)

`cambridge_exp.shap_export` writes `shap_top_tokens.csv` with empty `annotation_label` and `annotator_notes`.

## Allowed values for `annotation_label`

| Value | Meaning |
|-------|---------|
| `transfer_error` | Matches a known L1 contrastive-analysis style prediction |
| `named_entity` | Geographic or cultural reference (topical leakage risk) |
| `register_marker` | Vocabulary or syntax signalling level/register |
| `noise` | No clear linguistic explanation under your coding guide |

Fill one label per row (each row is one top-SHAP token for one head).

## After labelling

```bash
PYTHONPATH=. python -m cambridge_exp.aggregate_annotations annotation/shap_top_tokens.csv
```

This prints counts and percentages by label (optionally `--head l1`).
