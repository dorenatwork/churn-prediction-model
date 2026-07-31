# Explainability Summary

## Top factors driving churn predictions (SHAP, held-out test set)

- `tenure_months` (mean impact 2.603)
- `complain` (mean impact 1.197)
- `number_of_address` (mean impact 0.864)
- `average_order_value` (mean impact 0.821)
- `days_since_last_order` (mean impact 0.611)

## In plain language, for a non-technical stakeholder

The model looks at each customer's recent behaviour and flags anyone whose pattern resembles customers who left before. The strongest signals it uses are:

- **How long they've been a customer (tenure)** — newer customers churn more; this is the single biggest driver in the model.
- **Whether they've raised a complaint / support ticket** — customers who have had a service problem are meaningfully more likely to leave.
- **How recently and how much they've ordered** — customers with fewer/lower-value orders and a longer gap since their last purchase are more likely to be flagged.
- **Number of registered delivery addresses** — an unexpectedly strong signal in this data; likely a proxy for account complexity or household/shared usage rather than a direct cause of churn, and worth validating with the business before acting on it.
- Their **preferred product category and device** also shift risk somewhat — e.g. customers primarily buying mobile-category products churn more in this data than grocery buyers.

**Caveat worth flagging to stakeholders:** in this dataset, 'days since last order' has a counter-intuitive relationship with churn — customers flagged as churned actually have *fewer* days since their last order on average than retained customers. This likely reflects how the source system defined/labelled churn (e.g. a snapshot taken at a fixed point rather than 'no purchase in N days'), not a general truth about e-commerce customers. This is exactly the kind of finding that needs validating against the business's actual churn definition before acting on it operationally.

**Limitations to communicate:** this is trained on a single historical snapshot of ~5,600 customers from one dataset; it will need retraining and re-validation once real production data with a business-agreed churn definition is available, and predictions should be treated as prioritisation signals for a retention team, not as an automatic cutoff for customer treatment.
