# V2 engineering and methodology verification — 1 October 2026

## V1 baseline
Supplied executed V1 experiment: Logistic Regression, Random Forest and XGBoost; grouped train/calibration/validation/test; final Random Forest with sigmoid calibration, threshold 0.15; 5,996 held-out clients. Baseline tests executed against a preserved, isolated V1 source/artifact copy: **31 passed, 1 warning in 4.13s**, no skips. Raw local test logs are excluded from the public repository; this summary retains the executed outcome.

V1 artifact SHA-256: `3203e88feeb0200d5750c8cac4129f524b85df0b123fdf2acc5498a08748f122`. The local predecessor backup is excluded from the public repository. No retraining was performed during the audit.

## Problems found
- HIGH segment was coupled to the illustrative cost-selected review threshold.
- Model-family ranking compared calibrated AP; isotonic ties changed AP. Raw validation evidence still selects Random Forest, so retraining was unnecessary.
- SHAP exposed standardized numeric values without original-value context.
- Fairness rates lacked uncertainty estimates and mean-PD/calibration-gap context.
- Direct engineering accepted negative payments/fractional repayment codes; source validation accepted infinities. API request validation already rejected these inputs.
- Health/response schemas needed stricter output contracts; tests lacked explicit independent review boundaries and some invalid input checks.
- Pinned dependencies were installed and checked on Windows/Python 3.12. Docker execution remains unverified.

## Methodological changes
Family selection uses raw validation AP; calibration minimizes validation Brier, retaining uncalibrated/sigmoid/isotonic alternatives. ROC-AUC, log loss and reliability remain diagnostics, with no post-test selection. Existing validation evidence: RF raw AP 0.56588234, XGBoost 0.56193095, LR 0.55262664. RF sigmoid remains selected. Isotonic XGBoost AP 0.54485384 illustrates tie effects; its lower Brier does not imply better log loss.

Descriptive cuts are calibrated validation PD Q25/Q75 using numpy linear quantiles: **0.09535463063665424 / 0.21843748706391583**. LOW is below Q25, MEDIUM includes Q25 and excludes Q75, HIGH includes Q75. Boundaries use no test scores/outcome labels and do not depend on cost. Review remains **PD ≥ 0.15**, assumed FN:FP costs 5:1. MEDIUM cases may be flagged. Collapsed quartiles fail explicitly. These are not regulatory credit grades.

Wilson 95% intervals cover prevalence/flag rate/TPR/FPR for groups n ≥ 100, with relevant denominators ≥ 30. Small groups are marked and TPR/FPR suppressed where needed. Mean PD, calibration gap (mean PD minus prevalence) and Brier are separately reported as point estimates. Intervals condition on the model, approximate independent records and omit clustering, retraining/selection uncertainty and multiple comparisons. No fairness certification.

## Engineering changes
SHAP factors retain technical transformed values while adding original NT$ or unscaled engineered values and units. Repayment indicators expose observed code, tested category and active/inactive status. Contributions still explain the uncalibrated model, not displayed calibrated PD; no causal claims.

Dashboard separates PD, descriptive segment and illustrative review, explains the independent rules, and shows readable driver values. Pydantic prediction, factor and health outputs are strict; request validation remains strict. Shared service accepts Path/string artifact paths. Source and feature validation reject additional invalid histories.

Future training records raw ranking metrics independently of calibration, uses quartile segments and shares the fairness implementation. CI adds independent artifact verification, live smoke and dependency checks. These workflow changes have not run on GitHub. Imports and path handling were inspected; Dockerfile/Compose statically reviewed (same-image two-service setup, non-root user, model/checksum included, raw data excluded).

Artifact weights, preprocessing, calibration and threshold were preserved. Only bundled metadata changed; all stored held-out probabilities match. Updated artifact SHA-256: `8426e02f00fb7d3dce1d9ea345986f1ec5093fda0bf83b8e01521ec5a1953180`. External/embedded/frozen metadata agree. Windows full dependency resolution is `requirements-lock-windows.txt`; the Linux lock and direct pins are retained.

## Executed verification
- Final complete suite: **52 passed, 1 warning in 4.27s**, no skips. One existing Starlette/httpx test-client deprecation warning remains.
- Independent verifier: all **5,996** raw held-out predictions match stored values within 1e-12; all comparator metrics recomputed; source schema/hash validated; deployed-input hashes recomputed and split disjointness verified; validation quartiles and fairness rates/intervals recomputed; metadata agreement checked (`artifact_verification.json`).
- Compilation: `python -m compileall -q src app tests`, passed. Explicit imports of 19 non-dashboard modules passed; dashboard execution verified via AppTest. `pyproject.toml` parsed, version 2.0.0. `pip check`: no broken requirements.
- API: actual local uvicorn process, GET /health 200, POST /predict 200, repeated identical response, invalid input 422, OpenAPI 200. Real request/response retained; process terminated after checks.
- Streamlit: actual server health 200 and app shell 200; AppTest rendered and submitted LOW/MEDIUM/HIGH profiles without exceptions, verifying segment labels. No full browser visual inspection.
- Notebook: **6 code cells executed** through IPython; outputs regenerated from actual V2 artifacts, schema validated. Figures remain historical executed experiment figures; the notebook frontend was not manually reviewed.
- Docker: unavailable in the verification environment. Static review only; **no image build or container execution claimed**.
- Secret-pattern scan of source/app/tests/workflow/README/project metadata found no matching private keys/common token patterns; it is a heuristic, not a comprehensive secret audit. Large artifacts, raw data, local cache, virtual environment and local V1 backup are ignored.

## Metrics and evidence
No retraining was necessary. Existing aggregate test metrics are unchanged: ROC-AUC **0.7839707064**, AP **0.5671946180**, precision **0.4192634561**, recall **0.6701886792**, F1 **0.5158292187**, Brier **0.1343722299**, log loss **0.4298452274**. TN=3441, FP=1230, FN=437, TP=888; cost/customer=0.5695463642.

Only descriptive segment summaries changed: test LOW n=1414, mean PD 8.48%, prevalence 5.23%; MEDIUM n=3038, mean PD 12.96%, prevalence 15.87%; HIGH n=1544, mean PD 49.90%, prevalence 49.81%. README/model-card numeric results match executed reports. Existing comparator curves/test tables describe the retained V1 experiments, not newly trained models.

## Public evidence
The README, model card, experiment tables, figures, prediction evidence and concise verification JSON remain public. Raw installation/test/server logs, the predecessor backup and internal patch/change inventories are excluded from Git. Their executed outcomes and limitations are summarized here. See `reports/README.md` for the evidence inventory.

## Not verified and remaining limitations
Docker build/runtime; hosted GitHub Actions; future full training execution after the code changes; package build/wheel publication; browser pixel-level layout and notebook frontend rendering. The training path was inspected/imported/compiled, but **not re-executed**, preserving sound fitted evidence.

This is a retrospective V2 audit of a known historical test set, not a new blinded evaluation. Taiwan 2005 cohort, legacy demographic coding, no temporal/external validation, no EAD/LGD, imperfect subgroup/segment calibration, correlated features and noncausal SHAP remain. Wilson intervals omit dependence/selection uncertainty and there is no intersectional audit. The system is educational decision support, not production-ready or validated for real lending.

**Ready for human review**, with the explicit execution limitations above. No real-world lending suitability is claimed.

## Pre-GitHub presentation cleanup verification — 1 October 2026

The cleanup changed documentation and ignore rules only. No files were deleted and no model was retrained. Raw local logs and internal inventories/patches are retained locally but excluded from the intended public repository. Code, fitted artifacts, selection metadata, prediction tables and figures were checked against their pre-cleanup SHA-256 hashes: all 61 protected files were unchanged.

The complete test suite passed again: **52 passed, 1 warning in 5.58s**, with no skips. Compilation, independent artifact verification, API HTTP checks, Streamlit HTTP/AppTest and dependency consistency checks passed. All 5,996 held-out predictions and reported metrics still match the recorded evidence. The existing Starlette/httpx deprecation warning remains.

The intended first commit contains **87 files**, listed exactly in [public_repository_files.txt](public_repository_files.txt). The public text scan, including decoded notebook/JSON contents, found no local absolute filesystem paths or matching secret patterns. Relative virtual-environment setup instructions and the portable Docker container path are intentional. This pattern scan is not a comprehensive credential audit. See [pre_github_verification.json](pre_github_verification.json) for the concise cleanup record and the exact ignored-log inventory.

The repository is ready to initialize with Git for human review, subject to the verification limitations above. It has not been initialized, committed, pushed or published during this cleanup.
