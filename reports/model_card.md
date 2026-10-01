# Model card — V2 review of retained V1 model

## Intended use
Educational portfolio demonstration of next-month credit-card default probability, risk review priority and explanations. Not validated for lending approval, loan denial, pricing, regulatory ratings or automated lending.

## Training data and excluded attributes
30,000 historic UCI clients in Taiwan. Six months of histories, April–September 2005. Full source prevalence 22.12%. Sex, age, education, marriage and ID excluded. 19 deployed financial inputs; 15 derived features; 100 transformed dimensions (28 scaled numeric + 72 categorical indicators).

Dataset attribution and CC BY 4.0 licensing: see data/README.md. Code MIT. Archive SHA-256: `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`.

## Method
Grouped four-part partition; exact model-input duplicates never cross splits or CV folds. Training grouped three-fold AP tuning; disjoint calibration; raw validation AP/model selection followed by validation Brier/calibration selection (log loss and reliability remain separate diagnostics); independent test. Fixed seed 42. Final Random Forest: 160 trees, depth 8, minimum leaf 10, no class weighting. Sigmoid calibration via frozen estimator.

## Operating point
Validation illustrative costs: missed default=5, false flag=1. Frozen threshold 0.15. This is not a measured loss function. Independent descriptive segments use validation calibrated PD Q25=0.09535463063665424 and Q75=0.21843748706391583 with numpy linear interpolation. LOW < Q25; MEDIUM Q25 ≤ PD < Q75; HIGH ≥ Q75. Review flags use PD ≥ 0.15 independently, including some MEDIUM cases. These are relative segments, not regulatory credit grades.

## Held-out performance
5,996 clients, 1,325 positives. ROC-AUC 0.7840; AP 0.5672; precision 0.4193; recall 0.6702; F1 0.5158; Brier 0.1344; log loss 0.4298. Confusion: TN 3441, FP 1230, FN 437, TP 888. Illustrative cost/customer 0.5695.

400-record-bootstrap intervals condition on this fitted model; ROC [0.7679, 0.7974], AP [0.5411, 0.5959]. No retraining/selection uncertainty; record bootstrap does not account for identical-input clustering.

## Explainability
Tree-path-dependent SHAP for underlying uncalibrated probability. Positive/negative drivers, 500-case global sample, individual waterfall. Sum of base + all contributions equals raw model output, verified. Calibration is not decomposed. Correlated financial/delinquency features share attribution. No causal intervention interpretation.

## Fairness observations and uncertainty
In the held-out sex-coded groups, TPR is 69.18% for code 1 (male) and 65.38% for code 2 (female); FPR is 25.90% vs 26.59%. These descriptive gaps neither certify fairness nor establish discrimination. Age/education/marriage audits also show disparities and small groups. Denominators under 30 suppress TPR/FPR. Wilson 95% intervals accompany prevalence, flag rate, TPR and FPR for groups n ≥ 100 and relevant denominators ≥ 30. They condition on the fitted model, approximate independent records and exclude clustering, selection uncertainty and multiple comparisons. Brier, mean PD and calibration gap remain point estimates; no intersectional audit. Small-group results are unstable.

LOW mean predicted PD 8.48% versus observed default 5.23%; MEDIUM 12.96% versus 15.87%; HIGH 49.90% versus 49.81%. Overall Brier does not ensure uniform segment or demographic calibration.

## Operational limitations
Historic single cohort; no temporal, external, current or geographic validation. No EAD/LGD, policy fairness optimization, monitoring or production authentication. Narrow/legacy demographic coding. Undocumented repayment −2/0 semantics. Same-month payment/bill ratio is only a behavior proxy. Input ranges are not a robust OOD detector. Exact-input groups reduce leakage but future outcomes for related households remain unknown. Dataset does not document such relationships.

## Artifact and reproduction
`models/credit_risk.joblib`, `models/metadata.json`, `models/credit_risk.sha256`. Artifact contains engineered/preprocessed pipeline, calibrated predictor and 100 training background rows. Load only trusted joblib artifacts. Metadata records Python 3.12.14, package versions, seed, source hash and training time. Source experiment evidence lives in reports/. Independent verifier re-predicts all held-out raw rows from the shipped artifact.

## V2 audit and provenance
The raw validation AP comparison retains Random Forest (0.56588234), ahead of XGBoost (0.56193095) and Logistic Regression (0.55262664). No model refitting, recalibration or threshold reselection occurred. Isotonic ties can change AP; calibrated AP does not select the family in future training. V1 test probabilities and all reported aggregate test metrics are unchanged and independently reverified. Existing test evidence is historical, not a fresh blinded assessment.

SHAP factors now include original NT$ amounts, unscaled derived values and explicit repayment indicator context. Technical transformed values remain separately identified; contributions still explain the uncalibrated model and do not establish causality. See `v2_verification_report.md` for exact checks, limitations and file changes.
