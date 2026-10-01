"""Run: python -m streamlit run app/dashboard.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
import altair as alt
import streamlit as st
from pydantic import ValidationError
from app.schemas import Applicant
from src.config import ROOT, STATUS, BILLS, PAYMENTS
from src.models.service import get_service

st.set_page_config(page_title='Credit Risk Intelligence', page_icon='◈', layout='wide')
st.markdown('''<style>
.stApp {background: #0b1220; color: #e7edf5;}
.block-container {padding-top: 2rem; max-width: 1320px;}
h1 {letter-spacing: -1.5px;}
[data-testid="stMetric"] {background: #152238; padding: 20px; border-radius: 12px; border: 1px solid #263851;}
</style>''', unsafe_allow_html=True)
st.caption('RISK ANALYTICS / EXPLAINABLE MACHINE LEARNING / V2')
st.title('Credit Risk Intelligence')
st.markdown('Historical payment behavior → calibrated default probability → interpretable risk drivers')
st.info('Educational portfolio demonstration · Taiwan, 2005 · Not suitable for automated lending decisions.')
try:
    service = get_service()
except (FileNotFoundError, ValueError) as exc:
    st.error(f'Model unavailable: {exc}. Run python -m src.models.train from the project root.')
    st.stop()
metadata = service.metadata
profiles = json.loads((ROOT/'app/demo_profiles.json').read_text())
assess, evidence, governance = st.tabs(['Applicant assessment', 'Model evidence', 'Limitations & fairness'])
with assess:
    left, right = st.columns([1, 1.25], gap='large')
    with left:
        st.subheader('Applicant history')
        selected = st.selectbox('Start from a held-out demonstration profile', list(profiles))
        profile = profiles[selected]
        st.caption('Demo profiles are anonymized historical records. Edit all six months below. Amounts are NT$.')
        with st.form(f'applicant_{selected}'):
            credit = st.number_input('Credit limit (NT$)', min_value=1, max_value=10_000_000, value=profile['LIMIT_BAL'], step=1000)
            history = pd.DataFrame({'Month': ['September','August','July','June','May','April'],
                        'Repayment code': [profile[c] for c in STATUS],
                        'Bill amount': [profile[c] for c in BILLS],
                        'Payment amount': [profile[c] for c in PAYMENTS]})
            edited = st.data_editor(history, hide_index=True, disabled=['Month'], num_rows='fixed',
                column_config={'Repayment code': st.column_config.NumberColumn(min_value=-2, max_value=9, step=1),
                               'Bill amount': st.column_config.NumberColumn(min_value=-10_000_000, max_value=10_000_000, step=1),
                               'Payment amount': st.column_config.NumberColumn(min_value=0, max_value=10_000_000, step=1)},
                key=f'history_{selected}')
            submit = st.form_submit_button('Assess default risk', type='primary', width='stretch')
        st.caption('Codes: −1 = duly paid; positive = months of delay. −2 and 0 occur in the data but are not defined on the UCI page; retained as categories. A positive code counts as delinquency.')
        if submit:
            try:
                # Reject fractional amounts instead of silently truncating edited cells.
                arrays = []
                for c in ['Repayment code','Bill amount','Payment amount']:
                    values = edited[c].tolist()
                    if any(pd.isna(v) or float(v) != int(v) for v in values):
                        raise ValueError('Enter finite whole numbers for every history field.')
                    arrays.append([int(v) for v in values])
                applicant = Applicant(credit_limit=int(credit), repayment_status=arrays[0], bill_amounts=arrays[1], payment_amounts=arrays[2])
                st.session_state['prediction'] = service.predict(applicant)
                st.session_state['assessed_profile'] = selected
            except (ValidationError, ValueError, OverflowError) as exc:
                st.error(str(exc))
    with right:
        st.subheader('Risk assessment')
        result = st.session_state.get('prediction')
        if result:
            st.caption(f'Last submitted assessment · {st.session_state.get("assessed_profile")} starting profile. Submit edits to refresh.')
            a,b,c = st.columns(3)
            a.metric('Probability of default', f'{result["probability_of_default"]:.1%}')
            b.metric('Descriptive risk segment', result['risk_category'])
            c.metric('Illustrative review flag', 'YES' if result['flagged_for_review'] else 'NO')
            low = metadata['segments']['low_upper_exclusive']
            high = metadata['segments']['high_lower_inclusive']
            st.caption(f'LOW < {low:.1%} · MEDIUM {low:.1%} to < {high:.1%} · HIGH ≥ {high:.1%}')
            st.caption(f'Segments use validation PD quartiles. Review uses PD ≥ {result["decision_threshold"]:.0%} under assumed missed-default / false-flag costs of 5:1. A MEDIUM segment can be flagged. These are not regulatory credit grades.')
            st.markdown('#### What drives the score?')
            factors = pd.DataFrame(result['important_risk_factors'])
            chart = alt.Chart(factors).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X('contribution:Q', title=f'SHAP contribution ({result["explanation_unit"]})'),
                y=alt.Y('label:N', sort='-x', title=None),
                color=alt.Color('direction:N', scale=alt.Scale(domain=['increases_risk','decreases_risk'], range=['#ef7354','#35a7b5']), legend=alt.Legend(title=None)),
                tooltip=['label','feature_value','value_unit','indicator_active','contribution','direction']).properties(height=350)
            st.altair_chart(chart, width='stretch')
            st.dataframe(factors[['label', 'feature_value', 'value_unit', 'indicator_active', 'contribution', 'direction']], hide_index=True)
            st.caption('Largest positive and negative contributions to the underlying score. Calibrated PD is displayed above; this chart does not decompose it. Features are correlated and explanations are not causal.')
            for warning in result['warnings']:
                st.warning(warning)
            with st.expander('Prediction details'):
                st.json(result)
        else:
            st.markdown('Submit an applicant history to see probability, risk segment and signed explanations.')
        st.divider()
        st.caption(f'Model: {metadata["selected_model"]} · Calibration: {metadata["calibration"]} · Seed: {metadata["seed"]}')
with evidence:
    st.subheader('Frozen test evidence')
    metrics = metadata['test_metrics']
    cols = st.columns(4)
    for col, (label, value) in zip(cols, [('ROC-AUC',metrics['roc_auc']), ('PR-AUC (AP)',metrics['pr_auc_ap']), ('Recall',metrics['recall']), ('Brier',metrics['brier'])]):
        col.metric(label, f'{value:.4f}')
    st.caption(f'Held-out n={metrics["n"]:,}. Model/calibration/threshold selected before test evaluation. AP is average precision, not trapezoidal PR area.')
    st.dataframe(pd.read_csv(ROOT/'reports/test_model_comparison.csv'), hide_index=True)
    st.caption('Comparators use threshold 0.5; final model uses its validation-selected threshold. Ranking metrics are threshold independent.')
    c1,c2 = st.columns(2)
    c1.image(str(ROOT/'reports/figures/model_curves.png'), caption='Held-out ROC and precision–recall curves')
    c2.image(str(ROOT/'reports/figures/calibration_curve.png'), caption='Held-out reliability curve')
    st.image(str(ROOT/'reports/figures/shap_summary.png'), caption='Global SHAP summary on 500 held-out records')
    st.dataframe(pd.read_csv(ROOT/'reports/risk_segments.csv'), hide_index=True)
with governance:
    st.subheader('Use the evidence within its limits')
    st.markdown('''
- Predictors contain financial history only. Sex, age, education and marital status are excluded; proxies and group disparities can remain.
- This is one historical Taiwan cohort. Random group splits measure within-cohort generalization, not future or geographic robustness.
- Descriptive risk segments use validation PD Q25/Q75 boundaries independently of review costs. They are not regulatory ratings.
- The cost ratio of five for a missed default versus one for a false flag is an assumption, not a bank's loss model. Credit limit is not exposure-at-default; LGD is unavailable.
- SHAP explains model associations. It does not identify causal interventions or certify fairness.
- Group TPR/FPR is suppressed when its relevant denominator is below 30. Small samples and uncertain demographic codes limit interpretation.
- Wilson 95% intervals for groups of at least 100 use an independent-record approximation; they exclude clustering, model-selection uncertainty and multiple comparisons. Brier and calibration gap are point estimates.
''')
    st.dataframe(pd.read_csv(ROOT/'reports/fairness_audit.csv'), hide_index=True)
    st.caption('No applicant inputs are logged or stored by this app. Run locally; public deployment needs authentication, rate limits and a privacy review.')
