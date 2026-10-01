# Experiment and verification evidence

Public reports retain the evidence behind the README and model card:

- `experiment_summary.json`, `frozen_decisions.json`, `splits.json` and `split_manifest.csv`: recorded metadata, selection decisions and grouped partitions.
- `cv_results.csv`, `calibration_*.csv`, `validation_*.csv`, `v2_selection_audit.csv`: executed tuning, calibration and validation evidence.
- `test_predictions.csv`, `test_model_comparison.csv`, `bootstrap_ci.json`, `cost_sensitivity.csv`, `risk_segments.csv`: historical held-out predictions and derived results.
- `fairness_audit.csv`, `shap_importance.csv`, training EDA tables and `figures/`: descriptive audits, explanations and experiment figures.
- `model_card.md`, `v2_verification_report.md`, `artifact_verification.json`, `v2_http_smoke.json`, `api_example_*.json`: final documentation and concise verification evidence.
- `pre_github_verification.json` and `public_repository_files.txt`: cleanup checks and the exact intended first-commit file list.

Raw installation/test/server logs, the local V1 backup, an internal source patch/change list, the old package inventory and superseded verification summaries are kept locally but ignored by Git. These are transient engineering records; the public summaries retain executed outcomes, metrics, provenance and verification limitations. No experiment CSVs, JSON metric evidence or figures are ignored by a broad rule.

The stored test evidence is historical. Reproducing training overwrites local reports and does not create a new untouched evaluation cohort.
