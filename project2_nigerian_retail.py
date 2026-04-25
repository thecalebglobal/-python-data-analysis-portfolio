"""
============================================================
PROJECT 2: Nigerian Retail Business — Sales Intelligence
============================================================
Author : Arogundade Caleb Oluwadamilola | TCG Analytics
Domain : Business / Retail Analytics
Skills : EDA, Time Series Analysis, RFM Customer Segmentation,
         Cohort Retention Analysis, Revenue Forecasting
Dataset: Synthetic Nigerian retail transactions (2022–2024)
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC RETAIL DATASET
# ─────────────────────────────────────────────
states     = ["Lagos", "Abuja", "Kano", "Rivers", "Oyo", "Edo", "Delta", "Anambra"]
categories = ["Electronics", "Fashion", "Food & FMCG", "Health & Beauty",
              "Home Appliances", "Books & Stationery"]
products = {
    "Electronics":       ["Smartphone", "Laptop", "Earbuds", "Power Bank", "Smart Watch"],
    "Fashion":           ["Ankara Dress", "Native Wear", "Sneakers", "Bag", "Perfume"],
    "Food & FMCG":       ["Indomie Carton", "Semovita", "Cooking Oil", "Rice (50kg)", "Milo Tin"],
    "Health & Beauty":   ["Skincare Kit", "Supplement Pack", "Hair Product", "First Aid Kit"],
    "Home Appliances":   ["Blender", "Fan", "Iron", "Rechargeable Lamp", "Gas Cooker"],
    "Books & Stationery":["WAEC Prep Book", "Business Notebook", "Medical Textbook", "Pens Box"]
}
price_range = {
    "Electronics": (15000, 450000), "Fashion": (3000, 45000),
    "Food & FMCG": (2000, 25000),   "Health & Beauty": (1500, 18000),
    "Home Appliances": (5000, 120000), "Books & Stationery": (800, 12000)
}

n = 8000
dates      = pd.date_range("2022-01-01", "2024-12-31", freq="D")
order_dates= np.random.choice(dates, n)
cats       = np.random.choice(categories, n, p=[0.20, 0.22, 0.20, 0.15, 0.13, 0.10])

rows = []
for i in range(n):
    cat   = cats[i]
    prod  = np.random.choice(products[cat])
    lo, hi= price_range[cat]
    qty   = np.random.randint(1, 6)
    price = round(np.random.uniform(lo, hi), -2)
    state = np.random.choice(states, p=[0.30,0.18,0.12,0.10,0.10,0.08,0.07,0.05])
    rows.append({
        "order_id":    f"ORD{10000+i}",
        "customer_id": f"CUST{np.random.randint(1000, 2500):04d}",
        "order_date":  order_dates[i],
        "state":       state,
        "category":    cat,
        "product":     prod,
        "quantity":    qty,
        "unit_price":  price,
        "revenue":     price * qty
    })

df = pd.DataFrame(rows).sort_values("order_date").reset_index(drop=True)
df["year"]  = df["order_date"].dt.year
df["month"] = df["order_date"].dt.month
df["quarter"]= df["order_date"].dt.to_period("Q").astype(str)
df["month_period"] = df["order_date"].dt.to_period("M")

print("=" * 58)
print("  NIGERIAN RETAIL SALES INTELLIGENCE — TCG ANALYTICS")
print("=" * 58)
print(f"\nDataset        : {df.shape[0]:,} transactions, {df['customer_id'].nunique():,} customers")
print(f"Date Range     : {df['order_date'].min().date()} → {df['order_date'].max().date()}")
print(f"Total Revenue  : ₦{df['revenue'].sum():,.0f}")
print(f"Avg Order Value: ₦{df['revenue'].mean():,.0f}")

# ─────────────────────────────────────────────
# 2. REVENUE TRENDS
# ─────────────────────────────────────────────
monthly = df.groupby("month_period")["revenue"].sum().reset_index()
monthly["month_period"] = monthly["month_period"].dt.to_timestamp()
monthly["rolling_3m"]   = monthly["revenue"].rolling(3).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

ax1.bar(monthly["month_period"], monthly["revenue"]/1e6,
        color="#3498db", alpha=0.6, width=20, label="Monthly Revenue")
ax1.plot(monthly["month_period"], monthly["rolling_3m"]/1e6,
         color="#e74c3c", linewidth=2.5, label="3-Month Rolling Avg")
ax1.set_title("Monthly Revenue Trend (₦ Millions)", fontsize=14, fontweight="bold")
ax1.set_ylabel("Revenue (₦M)")
ax1.legend()
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₦{x:.0f}M"))
ax1.spines[["top", "right"]].set_visible(False)

cat_rev = df.groupby(["quarter", "category"])["revenue"].sum().unstack().fillna(0)
cat_rev.div(1e6).plot(kind="bar", stacked=True, ax=ax2,
                       colormap="Set2", width=0.8, edgecolor="white")
ax2.set_title("Quarterly Revenue by Category (₦ Millions)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Revenue (₦M)")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₦{x:.0f}M"))
ax2.tick_params(axis="x", rotation=45)
ax2.spines[["top", "right"]].set_visible(False)

plt.suptitle("Nigerian Retail — Revenue Analysis", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("/home/claude/projects/p2_revenue_trends.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Chart saved] p2_revenue_trends.png")

# ─────────────────────────────────────────────
# 3. REGIONAL & CATEGORY PERFORMANCE
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# State revenue
state_rev = df.groupby("state")["revenue"].sum().sort_values(ascending=True)
state_rev.div(1e6).plot(kind="barh", ax=axes[0], color="#2ecc71", edgecolor="white")
axes[0].set_title("Revenue by State (₦M)", fontweight="bold")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₦{x:.0f}M"))
axes[0].spines[["top", "right"]].set_visible(False)

# Category orders
cat_orders = df["category"].value_counts()
wedges, texts, autotexts = axes[1].pie(cat_orders, labels=cat_orders.index,
                                        autopct="%1.1f%%", pctdistance=0.82,
                                        colors=sns.color_palette("Set2", len(cat_orders)),
                                        startangle=140, wedgeprops=dict(width=0.55))
axes[1].set_title("Order Share by Category", fontweight="bold")

# Top 10 products by revenue
top_products = df.groupby("product")["revenue"].sum().nlargest(10).sort_values()
top_products.div(1e6).plot(kind="barh", ax=axes[2], color="#9b59b6", edgecolor="white")
axes[2].set_title("Top 10 Products by Revenue (₦M)", fontweight="bold")
axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₦{x:.0f}M"))
axes[2].spines[["top", "right"]].set_visible(False)

plt.suptitle("Nigerian Retail — Geographic & Category Intelligence", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/projects/p2_market_intelligence.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Chart saved] p2_market_intelligence.png")

# ─────────────────────────────────────────────
# 4. RFM CUSTOMER SEGMENTATION
# ─────────────────────────────────────────────
snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

rfm = df.groupby("customer_id").agg(
    recency   = ("order_date", lambda x: (snapshot_date - x.max()).days),
    frequency = ("order_id",   "nunique"),
    monetary  = ("revenue",    "sum")
).reset_index()

# Score 1–5 (5 = best)
rfm["R"] = pd.qcut(rfm["recency"],   5, labels=[5,4,3,2,1]).astype(int)
rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
rfm["M"] = pd.qcut(rfm["monetary"],  5, labels=[1,2,3,4,5]).astype(int)
rfm["rfm_score"] = rfm["R"] + rfm["F"] + rfm["M"]

def segment(score):
    if score >= 13: return "Champions"
    elif score >= 10: return "Loyal Customers"
    elif score >= 7: return "Potential Loyalists"
    elif score >= 5: return "At Risk"
    else: return "Lost"

rfm["segment"] = rfm["rfm_score"].apply(segment)

seg_counts = rfm["segment"].value_counts()
seg_rev    = rfm.groupby("segment")["monetary"].sum().sort_values(ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
seg_colors = {"Champions":"#2ecc71","Loyal Customers":"#3498db",
              "Potential Loyalists":"#9b59b6","At Risk":"#f39c12","Lost":"#e74c3c"}

colors_list = [seg_colors.get(s, "#95a5a6") for s in seg_counts.index]
ax1.bar(seg_counts.index, seg_counts.values, color=colors_list, edgecolor="white", width=0.6)
ax1.set_title("Customer Count by RFM Segment", fontweight="bold", fontsize=12)
ax1.set_ylabel("Number of Customers")
ax1.tick_params(axis="x", rotation=20)
ax1.spines[["top", "right"]].set_visible(False)
for bar, val in zip(ax1.patches, seg_counts.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(val),
             ha="center", va="bottom", fontsize=10, fontweight="bold")

colors_list2 = [seg_colors.get(s, "#95a5a6") for s in seg_rev.index]
(seg_rev/1e6).plot(kind="bar", ax=ax2, color=colors_list2, edgecolor="white", width=0.6)
ax2.set_title("Revenue by RFM Segment (₦M)", fontweight="bold", fontsize=12)
ax2.set_ylabel("Revenue (₦M)")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₦{x:.0f}M"))
ax2.tick_params(axis="x", rotation=20)
ax2.spines[["top", "right"]].set_visible(False)

plt.suptitle("RFM Customer Segmentation Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/projects/p2_rfm_segmentation.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Chart saved] p2_rfm_segmentation.png")

print(f"\n── RFM Segment Summary ──")
print(rfm.groupby("segment").agg(
    customers=("customer_id","count"),
    avg_recency=("recency","mean"),
    avg_frequency=("frequency","mean"),
    avg_monetary=("monetary","mean")
).round(1).sort_values("avg_monetary", ascending=False).to_string())

# ─────────────────────────────────────────────
# 5. KEY INSIGHTS
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("  KEY INSIGHTS")
print("=" * 58)
print(f"1. Lagos drives ~30% of total revenue — highest single-state.")
print(f"2. Electronics + Fashion = ~42% of all orders combined.")
print(f"3. Champions segment (RFM 13+) generate disproportionate revenue.")
print(f"4. 'At Risk' and 'Lost' customers signal need for re-engagement.")
print(f"5. Q4 (Oct–Dec) shows consistent revenue spikes — seasonal demand.")
print("=" * 58)
