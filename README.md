# E-Commerce Intelligence: From Clicks to Customers

An end-to-end analytics platform connecting browsing behavior, customer purchasing patterns, segmentation, product recommendations, and revenue forecasting for an online retailer.

Built on three public datasets treated as **complementary layers, not a single joined table**:

| Dataset | Role | Population |
|---|---|---|
| [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) | Website behavior & conversion | 12,330 anonymous sessions |
| [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | Transactions, customers, products | 5,942 identified customers, Dec 2009 \u2013 Dec 2011 |

> **Note on the original three files:** a third file, `Online_Retail.xlsx`, was provided alongside Retail II. Inspection showed it duplicates Retail II's second year almost exactly (541,909 vs 541,910 rows, identical date range and customer count), so it was **not used as an independent source** \u2014 the full two-year transaction history already lives in `online_retail_II.xlsx`.

---

## Architecture

```
WEBSITE VISITOR
      \u2193
Online Shoppers dataset \u2192 Purchase Intent Model (conversion prediction)
      \u2193
   PURCHASE
      \u2193
Online Retail II \u2192 RFM Segmentation \u2192 Product Intelligence (market basket)
                              \u2193                    \u2193
                    Purchase Propensity & CLV \u2190\u2192 Recommendation Engine
                              \u2193
                    Demand & Revenue Forecasting
```

The shopper-intention and transaction datasets are **never joined** \u2014 they describe different, non-overlapping populations (anonymous sessions vs. identified customers). Findings from each are integrated at the business/insight level instead, as documented in each component below.

## Repository Structure

```
01_online_shoppers/          Session-level browsing/conversion data
02_online_retail_transactions/  Consolidated, cleaned 2009-2011 transactions
03_customer_segmentation/    RFM feature engineering + K-Means clustering
04_product_intelligence/     Market basket analysis (Apriori / association rules)
05_recommendation_engine/    BG/NBD + Gamma-Gamma propensity & CLV, hybrid recommendations
06_purchase_prediction/      Conversion prediction (Logistic Regression / RF / XGBoost)
07_demand_forecasting/       Revenue forecasting (Seasonal Naive / SARIMA / Holt-Winters)
08_dashboard/                (planned) interactive dashboard
assets/                      Chart images used in this README + generation script
```

Each folder contains its own script(s) and output CSVs so any stage can be re-run independently once the previous stage's outputs exist.

---

## 1. Data Cleaning

Raw transactions: 1,067,371 line items \u2192 **802,932 clean rows retained (75.2%)** after removing:
- Cancelled orders (invoices starting with "C") \u2014 1.8% of rows
- Non-product stock codes (postage, bank fees, manual adjustments, etc.)
- Rows with zero/negative quantity or price, or missing Customer ID

**5,853 customers, 4,625 products, \u00a317.4M total revenue** across the clean dataset.

One data quality issue worth flagging explicitly: a single invoice for **80,995 units of "PAPER CRAFT, LITTLE BIRDIE"** (\u00a3168,470, a wholesale-scale one-off) skews any average-based metric it touches. It's excluded from market-basket analysis (fails the minimum-invoice-count threshold) and its monetary value is capped at the 99.5th percentile before feeding the propensity/CLV model in Section 4 \u2014 uncapped, it inflated one customer's predicted CLV to an implausible \u00a3386,819.

![Monthly Revenue Trend](assets/01_monthly_revenue_trend.png)

Revenue climbs every September\u2013November (holiday build-up) in both years, peaking in November \u2014 a pattern that recurs independently in the shopper-intention conversion data (see Section 3).

---

## 2. Customer Segmentation (RFM + K-Means)

Recency, Frequency, and Monetary value computed per customer, log-transformed and standardized, then clustered with K-Means (**k=4**, chosen by silhouette score = 0.37). Clusters were labeled using explicit rules against the population median on each dimension \u2014 not by arbitrary rank order, which would have mislabeled at least two segments in early iterations of this analysis.

![Segment Revenue Share](assets/02_segment_revenue_share.png)

| Segment | Customers | Avg Recency | Avg Frequency | Avg Monetary | Revenue Share |
|---|---|---|---|---|---|
| **Champions** | 1,114 (19%) | 26 days | 19.9 orders | \u00a311,307 | **72%** |
| At-Risk (High Value) | 1,459 (25%) | 208 days | 5.4 orders | \u00a32,129 | 18% |
| Loyal Customers | 1,235 (21%) | 28 days | 3.0 orders | \u00a3838 | 6% |
| Lost / Dormant | 2,045 (35%) | 393 days | 1.4 orders | \u00a3348 | 4% |

**Key insight:** 19% of customers generate 72% of revenue \u2014 a very steep concentration even by retail standards. "At-Risk (High Value)" is the most actionable segment: customers who behaved like Champions historically but haven't purchased in ~7 months, making them a natural win-back target.

---

## 3. Conversion Intelligence (Purchase Intent Prediction)

Session-level model on the Online Shoppers dataset (15.5% base conversion rate; class-imbalance handled via balanced class weights).

![Model Comparison](assets/05_conversion_model_comparison.png)

| Model | ROC-AUC | PR-AUC | F1 (Purchase class) |
|---|---|---|---|
| **XGBoost** | **0.930** | **0.742** | **0.643** |
| Random Forest | 0.911 | 0.686 | 0.636 |
| Logistic Regression | 0.893 | 0.622 | 0.592 |

![Conversion Drivers](assets/06_conversion_drivers.png)

**PageValues dominates** as a predictor (24% of total feature importance) \u2014 far ahead of any other signal. November is the next-strongest driver, mirroring the same holiday seasonality seen independently in the transaction data above, even though these are two unrelated customer populations. Returning visitors convert *worse* than new visitors (13.9% vs 24.9%), and traffic source matters enormously (TrafficType 8 converts at 27.7% vs 5.8% for TrafficType 13) \u2014 directly actionable for ad spend allocation.

---

## 4. Product Intelligence (Market Basket Analysis)

Apriori algorithm on UK transactions (83.8% of revenue), restricted to products appearing in \u226560 invoices to keep the itemset space tractable. 427 frequent itemsets \u2192 142 association rules (confidence \u226530%, lift \u22651.5).

![Top Market Basket Rules](assets/04_market_basket_top_rules.png)

Top rules show lift of 17\u201333\u00d7 \u2014 almost entirely "buys one color/variant of a themed set, buys the matching ones too" (teacup sets, cutlery sets, signage pairs, hand-warmer designs, alarm clocks). This pattern feeds directly into the recommendation engine below.

---

## 5. Recommendation Engine & Customer Lifetime Value

The centerpiece, combining outputs from Sections 2 and 4 with a purchase-propensity model:

- **BG/NBD model** estimates each customer's probability of still being "alive" (active) and expected purchases in the next 90 days, from their recency/frequency/tenure pattern.
- **Gamma-Gamma model** estimates expected monetary value per transaction \u2192 combined with BG/NBD for a **12-month CLV estimate**.
- **Hybrid recommendations**: association-rule matches from a customer's own purchase history, falling back to top sellers within their RFM segment when no rule fires.

![CLV by Segment](assets/03_clv_by_segment.png)

| Segment | Avg 12-Month CLV |
|---|---|
| Champions | \u00a34,571 |
| Loyal Customers | \u00a3988 |
| At-Risk (High Value) | \u00a3835 |
| Lost/Dormant | \u00a3187 |

CLV ranks in exactly the order the RFM segmentation predicts \u2014 a useful independent validation of the clustering. Sample customer profile card:

```
CUSTOMER #14646
Segment: Champions
Days since last purchase: 2
Repeat purchases: 90 (spanned 736 days)
Predicted 12-month CLV: \u00a3193,694
Purchase propensity (P alive): 99.9%
Expected purchases in next 90 days: 10.13
Recommended products:
  1. JUMBO BAG RED RETROSPOT
  2. PACK OF 72 RETROSPOT CAKE CASES
  3. LUNCH BAG SUKI DESIGN
```

**Known limitation:** top-tier Champions receive very similar recommendations, since they've already purchased most items that would trigger a distinct association rule, and the segment-fallback converges on the same bestsellers. A production version would benefit from collaborative filtering or item embeddings for this high-value, low-novelty segment.

---

## 6. Revenue & Demand Forecasting

Only 24 complete months of history \u2014 exactly 2 seasonal cycles, the bare minimum to detect annual seasonality at all. Evaluated on the last 4 complete months as holdout:

![Revenue Forecast](assets/07_revenue_forecast.png)

| Model | MAPE | RMSE |
|---|---|---|
| **Seasonal Naive** | **5.78%** | \u00a370,438 |
| SARIMA(1,1,1)(1,0,0,12) | 7.33% | \u00a375,491 |
| Holt-Winters (trend-only) | 33.81% | \u00a3392,032 |

**The simplest model won.** A more complex SARIMA spec with full seasonal differencing initially looked competitive on paper but produced a runaway extrapolation that contradicted every year of actual seasonality in the data \u2014 caught only by sanity-checking the forecast shape against history, not by trusting the error metric alone. With just two years of data there's no reliable way to separate trend from noise, so "expect what happened this month last year" is the most defensible forecast available. This should be revisited with 36+ months of history.

---

## Reproducing This Project

Each numbered folder's script can be run independently once its inputs exist:

```bash
# 1. Build the clean transaction layer (from raw Retail II sheets)
# 2. Run RFM segmentation
cd 03_customer_segmentation && python3 rfm_segmentation.py

# 3. Run market basket analysis
cd 04_product_intelligence && python3 market_basket.py

# 4. Run the recommendation engine (depends on 2 and 3's outputs)
cd 05_recommendation_engine && python3 recommendation_engine.py

# 5. Run conversion prediction (independent of the retail pipeline)
cd 06_purchase_prediction && python3 conversion_model.py

# 6. Run revenue forecasting
cd 07_demand_forecasting && python3 revenue_forecasting.py

# Regenerate README chart assets
python3 assets/make_charts.py
```

**Dependencies:** pandas, numpy, scikit-learn, xgboost, mlxtend, statsmodels, lifetimes, matplotlib, pyarrow.

## Key Takeaways

1. **Revenue is extremely concentrated**: 19% of customers (Champions) drive 72% of revenue \u2014 retention of this group matters more than acquisition.
2. **PageValues is the single best conversion signal** available pre-purchase; it dwarfs every other behavioral feature.
3. **Holiday seasonality (Sep\u2013Nov) shows up independently** in both the anonymous session data and the identified transaction data \u2014 a genuine cross-validation between two unrelated populations, despite never joining them.
4. **Simpler models sometimes win**, and forecast outputs need sanity-checking against domain knowledge, not just holdout error metrics \u2014 a lesson from the forecasting stage that generalizes to the whole project.
5. **Outliers can silently distort downstream models**: a single wholesale invoice inflated one customer's CLV by >500\u00d7 until it was explicitly capped.
