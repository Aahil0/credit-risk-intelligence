# Dataset provenance

**Default of Credit Card Clients**, 30,000 clients, Taiwan; historical repayment records April–September 2005; target: default payment next month.

- UCI: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Citation: Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
- Data license: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/), separate from the code's MIT license.
- Download: `python -m src.data.acquire`, or train with `python -m src.models.train`.
- Downloaded archive: `data/raw/uci.zip` (original UCI bytes). SHA-256 and validation in `reports/data_validation.json`.
- Original amounts are **New Taiwan dollars**. Do not interpret them as INR or current purchasing power.
- Raw data and model binaries are ignored by git. A fresh GitHub clone must acquire the dataset and train locally before launching the application. Training produces the model artifact and checksum. A trusted prebuilt artifact could be distributed separately in a future release; no such release is assumed.

The project's MIT license applies to its code and project documentation, not to the UCI dataset. Dataset reuse remains subject to its separate CC BY 4.0 license and attribution requirements.

## Data cleaning decisions

No missing cells in the validated data. Preserve legitimate negative bill statements as credits; never replace them with zero globally. Bills <= 0 get a zero payment/bill ratio and an explicit nonpositive bill count, so zero denominators never create infinities. Ratios capped at 10 represent a fixed, documented robustness transform, not a learned statistic. IDs are excluded. No winsorizing from all-data quantiles or removal based on target values.

Repayment -2 and 0 occur but their meanings are not specified on the cited UCI dataset page. Keep all codes as categorical indicators; count only positive values as delinquency, and >=2 as severe delay. Do not invent semantics for undocumented codes.

EDUCATION has extra codes 0/5/6 and MARRIAGE has 0. These remain separate unknown-code groups in the audit. All demographic fields are excluded from model inputs. ID, target and demographics cannot enter the engineered feature matrix.

Identical deployed financial inputs are assigned to one split together, even if labels or demographics differ. No rows are removed to make metrics look better.
