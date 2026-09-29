"""
Recommendation Engine (centerpiece: ties together segmentation,
market-basket rules, and purchase-propensity modeling)

Inputs:
  02_online_retail_transactions/processed/clean_transactions.parquet
  03_customer_segmentation/customer_segments.csv
  04_product_intelligence/association_rules.csv
  04_product_intelligence/top_products.csv

Outputs (in this folder):
  customer_propensity.csv   - per-customer P(alive), predicted 90-day
                               purchases, and CLV estimate (BG/NBD + Gamma-Gamma)
  customer_recommendations.csv - top-5 recommended products per customer
  sample_customer_cards.txt - a few formatted example profiles

Method:
  1. BG/NBD model estimates each customer's "alive" probability and
     expected future purchase count from their Recency/Frequency/T pattern.
  2. Gamma-Gamma model estimates expected monetary value per transaction
     (requires frequency > 0 and assumes monetary value is independent
     of purchase frequency - checked below).
  3. Product recommendations combine two sources:
       a) Association rules (from market_basket.py) triggered by items
          the customer has already bought, excluding items already owned.
       b) Fallback: top-selling products within the customer's RFM
          segment that they haven't bought yet (covers customers whose
          purchases don't trigger any association rule).
"""

import pandas as pd
import numpy as np
import ast
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

TXN_PATH = "../02_online_retail_transactions/processed/clean_transactions.parquet"
SEGMENTS_PATH = "../03_customer_segmentation/customer_segments.csv"
RULES_PATH = "../04_product_intelligence/association_rules.csv"
TOP_PRODUCTS_PATH = "../04_product_intelligence/top_products.csv"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
txn = pd.read_parquet(TXN_PATH)
txn["InvoiceDate"] = pd.to_datetime(txn["InvoiceDate"])
segments = pd.read_csv(SEGMENTS_PATH)
top_products = pd.read_csv(TOP_PRODUCTS_PATH)

snapshot_date = txn["InvoiceDate"].max() + pd.Timedelta(days=1)

# Winsorize extreme line-item revenue before it feeds the propensity/CLV
# models. One known outlier (StockCode 23843, "PAPER CRAFT, LITTLE
# BIRDIE") is a single £168K wholesale invoice for Customer 16446 -
# 99.9th percentile of line-item revenue is only ~£830, so this one row
# would otherwise single-handedly produce an unrealistic CLV for that
# customer. Capping (not deleting) preserves the transaction as a real
# purchase while preventing it from dominating the monetary model.
REVENUE_CAP = txn["Revenue"].quantile(0.995)
n_capped = (txn["Revenue"] > REVENUE_CAP).sum()
print(f"Capping {n_capped} line items above the 99.5th percentile (£{REVENUE_CAP:.2f}) "
      f"for propensity/CLV modeling only")
txn["Revenue_capped"] = txn["Revenue"].clip(upper=REVENUE_CAP)

# ---------------------------------------------------------------
# 2. BG/NBD: purchase propensity (P alive, expected future purchases)
# ---------------------------------------------------------------
summary = summary_data_from_transaction_data(
    txn, "CustomerID", "InvoiceDate",
    monetary_value_col="Revenue_capped", observation_period_end=snapshot_date
)
# frequency here = repeat purchase count (0 for one-time buyers, by BG/NBD convention)
print(f"Customers in BG/NBD summary: {len(summary)}")
print(f"One-time buyers (frequency=0): {(summary['frequency']==0).sum()}")

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(summary["frequency"], summary["recency"], summary["T"])

summary["prob_alive"] = bgf.conditional_probability_alive(
    summary["frequency"], summary["recency"], summary["T"]
)
summary["predicted_purchases_90d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
    90, summary["frequency"], summary["recency"], summary["T"]
)

# ---------------------------------------------------------------
# 3. Gamma-Gamma: expected monetary value -> CLV
# ---------------------------------------------------------------
repeat_buyers = summary[summary["frequency"] > 0]
corr = repeat_buyers[["frequency", "monetary_value"]].corr().iloc[0, 1]
print(f"\nFrequency/monetary correlation among repeat buyers: {corr:.3f} "
      f"(Gamma-Gamma assumes ~independence; a strong correlation would bias CLV)")

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(repeat_buyers["frequency"], repeat_buyers["monetary_value"])

summary["predicted_clv_12m"] = np.nan
summary.loc[repeat_buyers.index, "predicted_clv_12m"] = ggf.customer_lifetime_value(
    bgf,
    repeat_buyers["frequency"], repeat_buyers["recency"], repeat_buyers["T"],
    repeat_buyers["monetary_value"],
    time=12, freq="D", discount_rate=0.01,
)

summary = summary.reset_index().rename(columns={"index": "CustomerID"})
summary["CustomerID"] = summary["CustomerID"].astype(int)

# Merge in RFM segment
propensity = summary.merge(
    segments[["CustomerID", "Segment"]], on="CustomerID", how="left"
)
propensity.to_csv("customer_propensity.csv", index=False)

# Sanity check: confirm the known outlier (Customer 16446, the £168K bulk
# invoice) no longer distorts the model
outlier_row = propensity[propensity["CustomerID"] == 16446]
if len(outlier_row):
    r = outlier_row.iloc[0]
    print(f"\nOutlier check - Customer 16446 (previously £386K CLV pre-cap): "
          f"monetary_value=£{r['monetary_value']:.2f}, predicted_clv_12m="
          f"{'£%.2f' % r['predicted_clv_12m'] if pd.notna(r['predicted_clv_12m']) else 'N/A'}")

print("\n--- Purchase propensity summary by segment ---")
print(propensity.groupby("Segment").agg(
    Customers=("CustomerID", "count"),
    Avg_ProbAlive=("prob_alive", "mean"),
    Avg_Predicted90dPurchases=("predicted_purchases_90d", "mean"),
    Avg_CLV_12m=("predicted_clv_12m", "mean"),
).round(3))

# ---------------------------------------------------------------
# 4. Product recommendation: association rules + segment fallback
# ---------------------------------------------------------------
rules = pd.read_csv(RULES_PATH)
# Rules were saved with human-readable descriptions already; rebuild a
# lookup from single-item antecedent description -> ranked consequents
single_item_rules = rules[~rules["antecedents_desc"].str.contains(r"\+")].copy()
rule_lookup = (
    single_item_rules.sort_values("lift", ascending=False)
    .groupby("antecedents_desc")["consequents_desc"]
    .apply(list)
    .to_dict()
)

# Per-customer purchased product descriptions (map StockCode -> canonical description)
top_products_full = pd.read_csv(TOP_PRODUCTS_PATH, index_col="StockCode")
stockcode_to_desc = top_products_full["Description"].to_dict()

txn["Description_clean"] = txn["StockCode"].map(stockcode_to_desc).fillna(txn["Description"])

cust_purchases = txn.groupby("CustomerID")["Description_clean"].apply(set).to_dict()

# Segment-level top sellers (by revenue) for fallback recommendations
seg_top_products = {}
for seg in segments["Segment"].unique():
    seg_customers = segments.loc[segments["Segment"] == seg, "CustomerID"]
    seg_txn = txn[txn["CustomerID"].isin(seg_customers)]
    top_in_seg = (
        seg_txn.groupby("Description_clean")["Revenue"].sum()
        .sort_values(ascending=False).head(20).index.tolist()
    )
    seg_top_products[seg] = top_in_seg

def recommend_for_customer(customer_id, n=5):
    owned = cust_purchases.get(customer_id, set())
    scored = {}
    # Association-rule-based candidates
    for item in owned:
        for rec in rule_lookup.get(item, []):
            if rec not in owned:
                scored[rec] = scored.get(rec, 0) + 1  # simple vote count across triggering items
    ranked = sorted(scored.items(), key=lambda x: -x[1])
    recs = [r for r, _ in ranked][:n]
    # Fallback: fill remaining slots with top sellers from customer's segment
    if len(recs) < n:
        seg = segments.loc[segments["CustomerID"] == customer_id, "Segment"]
        seg = seg.iloc[0] if len(seg) else None
        if seg is not None:
            for p in seg_top_products.get(seg, []):
                if p not in owned and p not in recs:
                    recs.append(p)
                if len(recs) >= n:
                    break
    return recs[:n]

# ---------------------------------------------------------------
# 5. Generate recommendations for all segmented customers
# ---------------------------------------------------------------
rec_rows = []
for cust_id in segments["CustomerID"]:
    recs = recommend_for_customer(cust_id, n=5)
    rec_rows.append({"CustomerID": cust_id, **{f"Rec_{i+1}": r for i, r in enumerate(recs)}})

rec_df = pd.DataFrame(rec_rows)
rec_df.to_csv("customer_recommendations.csv", index=False)
print(f"\nGenerated recommendations for {len(rec_df)} customers -> customer_recommendations.csv")

# ---------------------------------------------------------------
# 6. Sample customer cards (demo output, matches the project brief's format)
# ---------------------------------------------------------------
sample_ids = (
    propensity.sort_values("predicted_clv_12m", ascending=False)
    .head(3)["CustomerID"].tolist()
)
lines = []
for cid in sample_ids:
    row = propensity[propensity["CustomerID"] == cid].iloc[0]
    recs = recommend_for_customer(cid, n=3)
    days_since_last_purchase = row["T"] - row["recency"]
    lines.append(f"CUSTOMER #{cid}")
    lines.append(f"Segment: {row['Segment']}")
    lines.append(f"Customer tenure: {row['T']:.0f} days since first purchase")
    lines.append(f"Days since last purchase: {days_since_last_purchase:.0f}")
    lines.append(f"Repeat purchases: {row['frequency']:.0f} (spanned {row['recency']:.0f} days between first and last)")
    lines.append(f"Monetary value (avg per transaction, capped): £{row['monetary_value']:.2f}")
    lines.append(f"Predicted 12-month CLV: £{row['predicted_clv_12m']:.2f}" if pd.notna(row['predicted_clv_12m']) else "Predicted 12-month CLV: N/A (one-time buyer)")
    lines.append(f"Purchase propensity (P alive): {row['prob_alive']*100:.1f}%")
    lines.append(f"Expected purchases in next 90 days: {row['predicted_purchases_90d']:.2f}")
    lines.append("Recommended products:")
    for i, r in enumerate(recs, 1):
        lines.append(f"  {i}. {r}")
    lines.append("")

with open("sample_customer_cards.txt", "w") as f:
    f.write("\n".join(lines))

print("\n" + "\n".join(lines))
