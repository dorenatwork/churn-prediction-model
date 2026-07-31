"""Assembles the EDA, model choice, evaluation, and architecture sections into a single PDF report."""
import base64
import json
import subprocess
from pathlib import Path

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ROOT_DIR

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build_html() -> str:
    comparison = json.loads((REPORTS_DIR / "model_comparison.json").read_text())
    test_metrics = json.loads((REPORTS_DIR / "test_metrics.json").read_text())
    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text())

    def img(name):
        return _img_b64(FIGURES_DIR / name)

    cm = test_metrics["confusion_matrix"]

    def row(name, m):
        return f"""<tr><td>{name}</td><td>{m['roc_auc']:.3f}</td><td>{m['pr_auc']:.3f}</td>
        <td>{m['precision']:.3f}</td><td>{m['recall']:.3f}</td><td>{m['f1']:.3f}</td></tr>"""

    comparison_rows = "\n".join(
        row(n.replace("_", " ").title(), m) for n, m in comparison.items()
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>E-commerce Customer Churn Prediction — Project Report</title>
<style>
  @page {{ size: A4; margin: 20mm 18mm; }}
  body {{ font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; color: #1a1a1a; line-height: 1.5; font-size: 10.5pt; }}
  h1 {{ font-size: 22pt; margin-bottom: 4pt; }}
  h2 {{ font-size: 15pt; margin-top: 26pt; border-bottom: 2px solid #2c3e50; padding-bottom: 4pt; page-break-after: avoid; }}
  h3 {{ font-size: 12pt; margin-top: 16pt; page-break-after: avoid; }}
  .cover {{ text-align: center; padding-top: 30%; page-break-after: always; }}
  .cover h1 {{ font-size: 28pt; }}
  .cover .subtitle {{ font-size: 13pt; color: #555; margin-top: 8pt; }}
  .cover .meta {{ margin-top: 60pt; font-size: 10pt; color: #777; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 9.5pt; }}
  th, td {{ border: 1px solid #ccc; padding: 5pt 8pt; text-align: left; }}
  th {{ background: #2c3e50; color: white; }}
  tr:nth-child(even) {{ background: #f7f8fa; }}
  .highlight-row {{ background: #fdf3d9 !important; font-weight: 600; }}
  img {{ max-width: 100%; display: block; margin: 10pt auto; }}
  .fig-caption {{ text-align: center; font-size: 9pt; color: #666; margin-top: -6pt; margin-bottom: 14pt; }}
  .two-col {{ display: flex; gap: 12pt; }}
  .two-col > div {{ flex: 1; }}
  .callout {{ background: #f0f4f8; border-left: 4px solid #2c3e50; padding: 8pt 12pt; margin: 10pt 0; font-size: 9.8pt; }}
  .callout.warn {{ background: #fdf0ec; border-left-color: #c0392b; }}
  code {{ background: #eef1f4; padding: 1pt 4pt; border-radius: 3px; font-size: 9pt; }}
  ul, ol {{ margin: 4pt 0; padding-left: 20pt; }}
  li {{ margin-bottom: 3pt; }}
  .section {{ page-break-inside: avoid; }}
  .weight-tag {{ float: right; font-size: 8.5pt; color: #888; font-weight: normal; }}
</style>
</head>
<body>

<div class="cover">
  <h1>E-commerce Customer Churn Prediction</h1>
  <div class="subtitle">Data preparation, model comparison, explainability, and production architecture</div>
  <div class="meta">Muneeb Rana &nbsp;·&nbsp; AI/ML/Data Science Take-Home Project</div>
</div>

<h2 id="overview">1. Overview & Dataset</h2>
<p>This project predicts e-commerce customer churn end to end: data cleaning and feature engineering, a comparison of
three modelling approaches with tracked experiments, explainability, a FastAPI prediction service, automated tests,
and a production architecture design.</p>

<p>Source dataset: <b>ankitverma2010/ecommerce-customer-churn-analysis-and-prediction</b> (Kaggle), 5,630 customers,
16.8% churn rate. The example schema I started from (customer ID, age, tenure, order count/value, recency, support
contact, subscription tier, country) doesn't map 1:1 onto this dataset, so several fields were adapted — most notably
the support-contact signal, which in this data is a binary "did the customer ever complain" flag rather than a ticket
count, so I named and constrained it (<code>complain</code>, 0/1) to reflect that honestly rather than implying a count
semantic the data doesn't have. Full field-by-field mapping is documented in the project README.</p>

<h2 id="eda" class="section">2. Data Analysis &amp; Preparation <span class="weight-tag">1.5 / 10</span></h2>
<p>EDA was run on the training split only, so no decision about preparation is influenced by data the model is later
evaluated on.</p>

<h3>Data quality issues found and fixed</h3>
<ul>
  <li>Inconsistent category labels: <code>PreferredLoginDevice</code> had both <i>'Phone'</i> and <i>'Mobile Phone'</i>;
  the order-category field had both <i>'Mobile'</i> and <i>'Mobile Phone'</i>; the payment-mode field had both
  <i>'COD'</i>/<i>'Cash on Delivery'</i> and <i>'CC'</i>/<i>'Credit Card'</i>. All folded to one canonical label.</li>
  <li>A small number of duplicate customer IDs were dropped so the same customer can't appear in more than one split.</li>
  <li>7 numeric columns each have ~4-6% missing values, with no pattern tied to churn — treated as missing-at-random
  and median-imputed inside the modelling pipeline (fit on train only, so val/test never leak into the imputation
  statistics).</li>
</ul>

<div class="two-col">
  <div>
    <img src="data:image/png;base64,{img('class_balance.png')}">
    <div class="fig-caption">Fig 1. Class balance (train) — 16.8% churn</div>
  </div>
  <div>
    <img src="data:image/png;base64,{img('correlation_heatmap.png')}">
    <div class="fig-caption">Fig 2. Correlation heatmap (train)</div>
  </div>
</div>

<p>Strongest univariate correlations with churn: tenure (r=-0.35), complaint flag (r=+0.25), days since last order
(r=-0.16, see caveat in §5), average order value proxy (r=-0.15). The class imbalance (16.8% churn) directly informs
both the evaluation metric choice (§4) and the class weighting used during training.</p>

<img src="data:image/png;base64,{img('numeric_distributions_by_churn.png')}" style="max-width: 92%;">
<div class="fig-caption">Fig 3. Numeric feature distributions split by churn outcome (train)</div>

<h2 id="model" class="section">3. Model Development &amp; Model Choice <span class="weight-tag">1.5 / 10</span></h2>
<p>Three approaches were trained on identical preprocessing (median/most-frequent imputation, scaling, one-hot
encoding — fit on train only) and tuned with 5-fold stratified cross-validation on train, scored by average precision
(PR-AUC):</p>

<table>
<tr><th>Model</th><th>Why chosen</th></tr>
<tr><td><b>Logistic Regression</b><br>(baseline)</td><td>Simple, fast, fully interpretable coefficients — establishes
the floor every other model has to beat.</td></tr>
<tr><td><b>Random Forest</b></td><td>Handles non-linear feature interactions and mixed feature types with minimal
tuning; a robust ensemble baseline.</td></tr>
<tr><td><b>XGBoost</b><br>(selected)</td><td>Typically the strongest choice for tabular data of this size and shape;
handles class imbalance natively via <code>scale_pos_weight</code>; trains fast on CPU; has first-class SHAP
(<code>TreeExplainer</code>) support, which mattered directly for the explainability requirement in §5.</td></tr>
</table>

<div class="callout">
Neural networks were considered and ruled out: at ~4k training rows with mostly tabular, low-cardinality features,
gradient-boosted trees outperform NNs here and add unnecessary training/deployment complexity for no accuracy benefit.
</div>

<h2 id="eval" class="section">4. Model Evaluation &amp; Validation <span class="weight-tag">2.0 / 10</span></h2>

<h3>Validation results (never seen during hyperparameter search)</h3>
<table>
<tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
{comparison_rows}
</table>
<p>XGBoost was selected for the best validation PR-AUC (the criterion I optimised model selection on), with Random
Forest close behind and Logistic Regression a clear, expected floor.</p>

<h3>Held-out test results (scored exactly once)</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>ROC-AUC</td><td>{test_metrics['roc_auc']:.3f}</td></tr>
<tr><td>PR-AUC</td><td>{test_metrics['pr_auc']:.3f}</td></tr>
<tr><td>Precision</td><td>{test_metrics['precision']:.3f}</td></tr>
<tr><td>Recall</td><td>{test_metrics['recall']:.3f}</td></tr>
<tr><td>F1</td><td>{test_metrics['f1']:.3f}</td></tr>
<tr><td>Decision threshold</td><td>{metadata['decision_threshold']:.2f} (F1-optimal, chosen on validation)</td></tr>
</table>
<p>Confusion matrix (test, n=845): TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}.</p>

<div class="callout">
<b>Metric prioritisation:</b> with 16.8% churn, accuracy is misleading — predicting "no churn" for everyone would
score ~83%. I prioritise <b>recall / PR-AUC</b> over precision or plain ROC-AUC: a missed churner (false negative) —
a customer who leaves with no retention attempt — is typically costlier than a false positive (an unnecessary
retention offer to someone who would have stayed). Precision still matters operationally since it bounds wasted
retention spend, so it's reported alongside, and the decision threshold is chosen to F1-optimise the
precision/recall trade-off on validation rather than defaulting to 0.5.
</div>

<h3>Validation strategy &amp; leakage prevention</h3>
<ul>
  <li><b>70/15/15 stratified split</b> on the target, computed once and reused identically by every downstream step.</li>
  <li><b>Train</b>: hyperparameter search (5-fold CV) for all three candidates.</li>
  <li><b>Validation</b>: model selection (best PR-AUC) and decision-threshold selection (best F1) — each used exactly once.</li>
  <li><b>Test</b>: scored exactly once, after the model and threshold were already fixed. It never informs model
  choice, hyperparameters, or the threshold.</li>
  <li>The preprocessing pipeline (imputers, scaler, encoder) is fit only inside <code>Pipeline.fit()</code> on train,
  so validation/test statistics never leak into imputation or scaling; duplicate customer IDs are dropped before
  splitting so no customer appears in two splits.</li>
</ul>

<div class="callout warn">
<b>Caveat worth flagging:</b> "days since last order" has a counter-intuitive negative correlation with churn in this
dataset — churned customers show <i>fewer</i> days since their last order on average (3.2 vs 4.8 for retained
customers). This is very likely an artifact of how the source system's churn label / snapshot was constructed (e.g.
a fixed observation date rather than "no purchase in N days"), not a general e-commerce truth. I flagged it rather
than silently dropping or flipping the signal, since doing that without understanding it would be its own
data-integrity risk.
</div>

<h2 id="explain" class="section">5. Explainability &amp; Business Reasoning <span class="weight-tag">1.0 / 10</span></h2>

<div class="two-col">
  <div><img src="data:image/png;base64,{img('shap_summary.png')}"></div>
  <div><img src="data:image/png;base64,{img('feature_importance.png')}"></div>
</div>
<div class="fig-caption">Fig 4-5. SHAP summary and top feature importances (held-out test set)</div>

<h3>In plain language, for a non-technical stakeholder</h3>
<p>The model looks at each customer's recent behaviour and flags anyone whose pattern resembles customers who left
before. The strongest signals it uses are: <b>how long they've been a customer</b> (newer customers churn more — the
single biggest driver); <b>whether they've ever raised a complaint</b> (a meaningfully stronger churn signal);
<b>how recently and how much they've ordered</b>; and, somewhat unexpectedly, <b>how many delivery addresses are on
their account</b> — likely a proxy for account complexity or shared/household usage rather than a direct cause of
churn, and worth validating with the business before acting on it.</p>

<p><b>Limitations to communicate:</b> this is trained on a single historical snapshot of ~5,600 customers from one
dataset; it needs retraining and re-validation once real production data with a business-agreed churn definition is
available, and predictions should be treated as prioritisation signals for a retention team, not an automatic cutoff
for customer treatment.</p>

<h2 id="api" class="section">6. Prediction API <span class="weight-tag">1.5 / 10</span></h2>
<p>A FastAPI service exposes <code>POST /predict</code> (5 required fields: tenure, orders, average order value,
days since last order, and a validated 0/1 complaint flag — extra behavioural fields are optional and imputed if
omitted) and <code>GET /health</code>. Example:</p>
<table>
<tr><th>Request</th><th>Response</th></tr>
<tr><td><code>{{"tenure_months": 14, "orders": 8,<br>"average_order_value": 72.5,<br>"days_since_last_order": 63,<br>"complain": 1}}</code></td>
<td><code>{{"churn_probability": 0.37,<br>"prediction": "medium_risk"}}</code></td></tr>
</table>
<p>The <code>complain</code> field is strictly validated to 0/1 — an earlier version accepted any non-negative
integer, which silently fed out-of-distribution values (e.g. a literal ticket count) into a model that had only ever
seen 0 or 1 during training. Constraining the field to match what the source data actually represents turns that
class of input error into a clean <code>422</code> instead of a misleading prediction.</p>

<h2 id="arch" class="section">7. Production Architecture <span class="weight-tag">1.5 / 10</span></h2>
<img src="data:image/png;base64,{img('architecture_diagram.png')}">
<div class="fig-caption">Fig 6. End-to-end production architecture</div>

<h3>Monitoring, drift, retraining, versioning</h3>
<ul>
  <li><b>Monitoring:</b> every prediction (input features, output probability, model version, timestamp) is logged
  for later comparison against realised churn outcomes.</li>
  <li><b>Data/model drift:</b> incoming feature distributions are compared against the training distribution on a
  rolling window (e.g. population stability index per feature); prediction-score distribution drift is tracked over
  time, with alerts when it crosses a threshold.</li>
  <li><b>Performance decay:</b> once realised churn labels are available, PR-AUC/recall is recomputed on recent
  cohorts and compared against the validation baseline.</li>
  <li><b>Retraining:</b> scheduled (e.g. monthly) or drift/decay-triggered, reusing the same pipeline; a new
  candidate is only promoted if it beats the current production model on the same validation metric.</li>
  <li><b>Versioning:</b> the MLflow Model Registry versions every promoted model, keeps the previous production
  version archived (not deleted) for instant rollback, and every prediction log line records the serving model
  version for traceability.</li>
</ul>

<h2 id="test" class="section">8. Testing &amp; Reproducibility <span class="weight-tag">1.0 / 10</span></h2>
<p>13 automated tests cover data cleaning/mapping/dedup logic, split ratios and stratification, the preprocessing
pipeline (missing-value handling, fit/predict on synthetic data), evaluation metric correctness, and the API
(a valid payload, input validation including the complaint flag being rejected outside 0/1, health checks, and
behaviour with no model loaded). The full pipeline — data prep, EDA, training, explainability, tests — was verified
end to end from a clean state and reproduces identical results, using only CPU (no GPU needed for this dataset size).</p>

<h2 id="limits" class="section">9. Assumptions, Limitations &amp; Proposed Improvements <span class="weight-tag">1.5 / 10</span></h2>
<h3>Assumptions</h3>
<p>Churn is defined by the source dataset's churn flag; its exact business definition isn't documented by the
dataset provider, and a real deployment would need this confirmed with the business — especially given the
days-since-last-order anomaly in §4.</p>

<h3>Limitations</h3>
<ul>
  <li>Single static dataset/snapshot — no temporal validation (training on an earlier cohort, testing on a later
  one), which a production system should do to catch behavioural drift.</li>
  <li>Average order value, country, and the complaint field are proxies for signals the source dataset doesn't
  literally contain — a real deployment should source these from the actual order and ticketing systems.</li>
  <li>No calibration check on predicted probabilities (e.g. a reliability diagram) — worth adding if raw
  probabilities, not just risk bands, are shown to business users.</li>
</ul>

<h3>Proposed production improvements</h3>
<p>Temporal train/test splits and periodic backtesting; automated drift monitoring wired to real alerting; a feature
store shared between training and serving to guarantee train/serve consistency; canary/shadow deployment of newly
promoted models before flipping the production alias; a CI pipeline that runs the test suite and a training smoke
run on every change.</p>

<h2 id="ai" class="section">10. Use of AI-Assisted Development Tools</h2>
<p>I used Claude Code (Anthropic) throughout this project — for research into comparable datasets and approaches,
and as a pair-programmer for implementation, under my direction: I made the modelling, metric, and architecture
decisions, reviewed and tested the generated code, and iterated on it (for example, reworking the complaint field
after catching that its default naming implied a count semantic the data didn't actually have).</p>

</body></html>
"""
    return html


def run() -> None:
    html_path = ROOT_DIR / "reports" / "project_report.html"
    pdf_path = ROOT_DIR / "Project_Report.pdf"
    html_path.write_text(build_html())

    subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            f"--print-to-pdf={pdf_path}",
            "--print-to-pdf-no-header",
            "--no-pdf-header-footer",
            str(html_path),
        ],
        check=True,
    )
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    run()
