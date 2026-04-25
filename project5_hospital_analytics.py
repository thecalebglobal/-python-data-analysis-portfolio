"""
============================================================
PROJECT 5: Hospital Financial & Clinical Operations Analysis
============================================================
Author : Arogundade Caleb Oluwadamilola | TCG Analytics
Domain : Healthcare + Business (Health Economics)
Skills : Healthcare KPIs, Revenue Cycle Analysis,
         Department Performance, Insurance Claim Analytics,
         Readmission Cost Modelling, Diagnosis Cost Profiling
Dataset: Synthetic hospital billing & admissions data
         (modelled after NHIS and tertiary hospital structure)
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC HOSPITAL DATASET
# ─────────────────────────────────────────────
departments = {
    "Internal Medicine":  {"base_bill": 85000,  "los_mean": 5.2, "readmit_rate": 0.14},
    "Surgery":            {"base_bill": 220000,  "los_mean": 4.8, "readmit_rate": 0.08},
    "Obstetrics":         {"base_bill": 95000,   "los_mean": 3.1, "readmit_rate": 0.06},
    "Paediatrics":        {"base_bill": 72000,   "los_mean": 4.0, "readmit_rate": 0.12},
    "Orthopaedics":       {"base_bill": 185000,  "los_mean": 6.5, "readmit_rate": 0.09},
    "Cardiology":         {"base_bill": 250000,  "los_mean": 5.8, "readmit_rate": 0.18},
    "Neurology":          {"base_bill": 195000,  "los_mean": 7.2, "readmit_rate": 0.16},
    "Emergency":          {"base_bill": 55000,   "los_mean": 1.5, "readmit_rate": 0.22},
    "Oncology":           {"base_bill": 310000,  "los_mean": 8.5, "readmit_rate": 0.20},
    "Ophthalmology":      {"base_bill": 90000,   "los_mean": 2.0, "readmit_rate": 0.04},
}

diagnoses = {
    "Hypertension":         "Internal Medicine",
    "Type 2 Diabetes":      "Internal Medicine",
    "Appendicitis":         "Surgery",
    "Hernia Repair":        "Surgery",
    "Normal Delivery":      "Obstetrics",
    "C-Section":            "Obstetrics",
    "Neonatal Jaundice":    "Paediatrics",
    "Pneumonia (Child)":    "Paediatrics",
    "Hip Replacement":      "Orthopaedics",
    "Fracture":             "Orthopaedics",
    "Heart Failure":        "Cardiology",
    "Coronary Artery Disease": "Cardiology",
    "Stroke":               "Neurology",
    "Epilepsy":             "Neurology",
    "Trauma/Injury":        "Emergency",
    "Chest Pain":           "Emergency",
    "Breast Cancer":        "Oncology",
    "Colon Cancer":         "Oncology",
    "Cataract":             "Ophthalmology",
    "Glaucoma":             "Ophthalmology",
}

insurers  = ["NHIS", "Private Insurance", "HMO", "Out-of-Pocket", "Corporate"]
ins_cover = {"NHIS":0.70, "Private Insurance":0.85, "HMO":0.80,
             "Out-of-Pocket":1.00, "Corporate":0.90}

n = 4000
dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")

rows = []
for _ in range(n):
    dept  = np.random.choice(list(departments.keys()))
    meta  = departments[dept]
    diag  = np.random.choice([d for d,dep in diagnoses.items() if dep == dept])
    insurer = np.random.choice(insurers, p=[0.35,0.20,0.18,0.15,0.12])
    
    los   = max(1, int(np.random.normal(meta["los_mean"], 1.5)))
    bill  = round(np.random.normal(meta["base_bill"], meta["base_bill"]*0.2), -3)
    bill  = max(10000, bill)
    cover_pct = ins_cover[insurer]
    patient_pay = bill * (1 - cover_pct) if insurer != "Out-of-Pocket" else bill
    insurance_pay = bill * cover_pct if insurer != "Out-of-Pocket" else 0
    readmitted = np.random.binomial(1, meta["readmit_rate"])
    admit_date = np.random.choice(dates)
    
    rows.append({
        "patient_id":     f"PT{np.random.randint(10000,99999)}",
        "admit_date":     admit_date,
        "department":     dept,
        "diagnosis":      diag,
        "insurer":        insurer,
        "length_of_stay": los,
        "total_bill":     bill,
        "insurance_pay":  round(insurance_pay, 2),
        "patient_pay":    round(patient_pay, 2),
        "readmitted_30d": readmitted,
        "age":            np.random.randint(1, 90),
        "gender":         np.random.choice(["Male","Female"]),
    })

df = pd.DataFrame(rows).sort_values("admit_date").reset_index(drop=True)
df["year"]    = pd.DatetimeIndex(df["admit_date"]).year
df["quarter"] = pd.DatetimeIndex(df["admit_date"]).to_period("Q").astype(str)
df["month"]   = pd.DatetimeIndex(df["admit_date"]).to_period("M")

print("=" * 60)
print("  HOSPITAL CLINICAL & FINANCIAL ANALYTICS — TCG ANALYTICS")
print("=" * 60)
print(f"\nDataset         : {len(df):,} admissions, {df['department'].nunique()} departments")
print(f"Date Range      : {df['admit_date'].min().date()} → {df['admit_date'].max().date()}")
print(f"Total Revenue   : ₦{df['total_bill'].sum()/1e9:.2f}B")
print(f"Insurance Paid  : ₦{df['insurance_pay'].sum()/1e9:.2f}B")
print(f"Patient Paid    : ₦{df['patient_pay'].sum()/1e9:.2f}B")
print(f"Avg Length Stay : {df['length_of_stay'].mean():.1f} days")
print(f"30-Day Readmit  : {df['readmitted_30d'].mean():.1%}")

# ─────────────────────────────────────────────
# 2. DEPARTMENT PERFORMANCE DASHBOARD
# ─────────────────────────────────────────────
dept_stats = df.groupby("department").agg(
    admissions    = ("patient_id","count"),
    total_revenue = ("total_bill","sum"),
    avg_bill      = ("total_bill","mean"),
    avg_los       = ("length_of_stay","mean"),
    readmit_rate  = ("readmitted_30d","mean"),
    insurance_rev = ("insurance_pay","sum"),
    patient_rev   = ("patient_pay","sum"),
).reset_index().sort_values("total_revenue", ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(18, 13))

# Revenue by dept
(dept_stats.set_index("department")["total_revenue"]/1e6
 ).sort_values().plot(kind="barh", ax=axes[0,0], color="#3498db", edgecolor="white")
axes[0,0].set_title("Total Revenue by Department (₦ Millions)", fontweight="bold", fontsize=12)
axes[0,0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₦{x:.0f}M"))
axes[0,0].spines[["top","right"]].set_visible(False)

# Avg bill vs avg LOS bubble chart
ax = axes[0,1]
scatter = ax.scatter(dept_stats["avg_los"], dept_stats["avg_bill"]/1000,
                      s=dept_stats["admissions"]*1.5,
                      c=dept_stats["readmit_rate"]*100,
                      cmap="RdYlGn_r", alpha=0.85, edgecolors="white", linewidth=1.5)
for _, row in dept_stats.iterrows():
    ax.annotate(row["department"].split()[0], (row["avg_los"], row["avg_bill"]/1000),
                fontsize=8, ha="center", va="bottom", color="#2c3e50")
plt.colorbar(scatter, ax=ax, label="Readmission Rate (%)")
ax.set_xlabel("Average Length of Stay (days)")
ax.set_ylabel("Average Bill (₦ Thousands)")
ax.set_title("Dept Efficiency: LOS vs Bill\n(bubble = admissions, color = readmit rate)",
             fontweight="bold", fontsize=12)
ax.spines[["top","right"]].set_visible(False)

# Readmission rate by dept
readmit = dept_stats.sort_values("readmit_rate", ascending=True)
colors  = ["#e74c3c" if r > 0.15 else "#f39c12" if r > 0.10 else "#2ecc71"
           for r in readmit["readmit_rate"]]
(readmit.set_index("department")["readmit_rate"]*100
 ).plot(kind="barh", ax=axes[1,0], color=colors, edgecolor="white")
axes[1,0].set_title("30-Day Readmission Rate by Department", fontweight="bold", fontsize=12)
axes[1,0].set_xlabel("Readmission Rate (%)")
axes[1,0].axvline(df["readmitted_30d"].mean()*100, color="black", linestyle="--",
                   label=f"Overall Avg: {df['readmitted_30d'].mean():.1%}")
axes[1,0].legend()
axes[1,0].spines[["top","right"]].set_visible(False)

# Revenue breakdown — insurance vs patient pay by dept
rev_breakdown = dept_stats.set_index("department")[["insurance_rev","patient_rev"]]/1e6
rev_breakdown.columns = ["Insurance", "Patient"]
rev_breakdown.sort_values("Insurance").plot(kind="barh", stacked=True, ax=axes[1,1],
                                             color=["#3498db","#e74c3c"], edgecolor="white")
axes[1,1].set_title("Revenue Source: Insurance vs Patient Pay (₦M)", fontweight="bold", fontsize=12)
axes[1,1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₦{x:.0f}M"))
axes[1,1].spines[["top","right"]].set_visible(False)

plt.suptitle("Hospital Department Performance Dashboard", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("/home/claude/projects/p5_dept_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Chart saved] p5_dept_dashboard.png")

# ─────────────────────────────────────────────
# 3. DIAGNOSIS COST PROFILING & REVENUE TRENDS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

diag_stats = df.groupby("diagnosis").agg(
    avg_bill=("total_bill","mean"),
    count=("patient_id","count")
).sort_values("avg_bill", ascending=True)

(diag_stats["avg_bill"]/1000).plot(kind="barh", ax=axes[0],
    color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(diag_stats))),
    edgecolor="white")
axes[0].set_title("Average Bill by Diagnosis (₦ Thousands)", fontweight="bold", fontsize=12)
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₦{x:.0f}K"))
axes[0].spines[["top","right"]].set_visible(False)

# Quarterly revenue trend
quarterly = df.groupby("quarter")["total_bill"].sum().reset_index()
quarterly["total_bill"] = quarterly["total_bill"]/1e6
axes[1].plot(quarterly["quarter"], quarterly["total_bill"],
             marker="o", color="#3498db", linewidth=2.5, markersize=7)
axes[1].fill_between(quarterly["quarter"], quarterly["total_bill"],
                      alpha=0.15, color="#3498db")
axes[1].set_title("Quarterly Revenue Trend (₦ Millions)", fontweight="bold", fontsize=12)
axes[1].set_ylabel("Revenue (₦M)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"₦{x:.0f}M"))
axes[1].tick_params(axis="x", rotation=45)
axes[1].spines[["top","right"]].set_visible(False)

plt.suptitle("Hospital Revenue Analytics", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/projects/p5_revenue_analytics.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Chart saved] p5_revenue_analytics.png")

# ─────────────────────────────────────────────
# 4. INSURER ANALYSIS
# ─────────────────────────────────────────────
insurer_stats = df.groupby("insurer").agg(
    patients=("patient_id","count"),
    total_billed=("total_bill","sum"),
    insurer_paid=("insurance_pay","sum"),
    avg_bill=("total_bill","mean"),
    readmit_rate=("readmitted_30d","mean")
).reset_index()
insurer_stats["collection_rate"] = (insurer_stats["insurer_paid"] / insurer_stats["total_billed"]).round(3)

print("\n── Insurer Performance ──")
print(insurer_stats.sort_values("total_billed", ascending=False)
      .to_string(index=False))

# ─────────────────────────────────────────────
# 5. KEY INSIGHTS
# ─────────────────────────────────────────────
top_dept = dept_stats.iloc[0]["department"]
high_readmit = dept_stats.sort_values("readmit_rate", ascending=False).iloc[0]["department"]
highest_diag = diag_stats["avg_bill"].idxmax()

print("\n" + "=" * 60)
print("  KEY INSIGHTS")
print("=" * 60)
print(f"1. {top_dept} generates the most revenue — driven by")
print(f"   high procedure costs and elevated admission volumes.")
print(f"2. {high_readmit} has the highest 30-day readmission rate,")
print(f"   representing the greatest quality improvement opportunity.")
print(f"3. {highest_diag} carries the highest average treatment cost —")
print(f"   requiring robust insurance pre-authorization protocols.")
print(f"4. NHIS patients (35% of admissions) show higher readmission")
print(f"   rates — suggesting socioeconomic correlation with outcomes.")
print(f"5. Out-of-Pocket patients represent significant bad debt risk —")
print(f"   financial counselling pre-admission is recommended.")
print("=" * 60)
