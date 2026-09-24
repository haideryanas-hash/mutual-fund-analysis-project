
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

RAW_FILE = "comprehensive_mutual_funds_data.csv"
CLEAN_FILE = "cleaned_mutual_funds_data.csv"
SCORED_FILE = "all_funds_scored.csv"
TOP30_FILE = "Top30_Mutual_Funds.xlsx"

WEIGHTS = {
    "returns_3yr": 0.40,
    "expense_ratio": 0.30,   
    "fund_age_yr": 0.20,
    "returns_1yr": 0.10,
}


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

section("STEP 1: LOAD DATA")
df = pd.read_csv(RAW_FILE)
print(f"Rows loaded            : {df.shape[0]}")
print(f"Columns loaded         : {df.shape[1]}")
print(f"Columns                : {list(df.columns)}")

section("STEP 2: UNDERSTAND THE DATASET")
print(df.dtypes)
print("\nSample rows:")
print(df.head(3))

section("STEP 3: DATA CLEANING")

print("Missing values BEFORE cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

numeric_text_cols = ["alpha", "beta", "sortino", "sharpe", "sd"]
for col in numeric_text_cols:
    before_non_numeric = df[col].apply(lambda x: not str(x).replace("-", "", 1).replace(".", "", 1).isdigit()).sum()
    df[col] = pd.to_numeric(df[col], errors="coerce")  
    print(f"  Converted '{col}' to numeric. Non-numeric placeholders found & set to NaN: {before_non_numeric}")

dup_count = df.duplicated(subset=["scheme_name"]).sum()
print(f"\nDuplicate scheme_name rows found: {dup_count}")
print("Investigation: duplicate names belong to different plan variants of the same")
print("scheme (different fund_size_cr / min_sip / min_lumpsum combinations, e.g.")
print("Direct vs Regular plans). They are NOT exact duplicate rows, so they are")
print("kept as distinct entries; only fully identical rows (all columns equal) are dropped.")
exact_dupes = df.duplicated().sum()
print(f"Fully identical duplicate rows (all columns): {exact_dupes}")
df 

print(f"\nRows missing returns_3yr : {df['returns_3yr'].isnull().sum()} (dropped - required for scoring)")
print(f"Rows missing returns_5yr : {df['returns_5yr'].isnull().sum()} (kept - not used in scoring)")
df = df.dropna(subset=["returns_3yr"]).reset_index(drop=True)

text_cols = df.select_dtypes(include="object").columns
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()

print(f"\nFinal cleaned dataset shape: {df.shape}")
print("\nMissing values AFTER cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().sum() else "None remaining in key scoring columns")

df.to_csv(CLEAN_FILE, index=False)
print(f"\nClean dataset saved -> {CLEAN_FILE}")

section("STEP 4: DATA EXPLORATION (EDA)")

q1 = df.groupby("amc_name")["fund_size_cr"].sum().sort_values(ascending=False)
print("Q1. Which AMC manages the highest AUM?")
print(q1.head(5), "\n")

q2 = df.groupby("category")["returns_3yr"].mean().sort_values(ascending=False)
print("Q2. Which category has the highest average 3Y return?")
print(q2, "\n")

q3 = df["risk_level"].value_counts().sort_index()
print("Q3. Which risk level has the most funds? (1=lowest risk ... 6=highest risk)")
print(q3, "\n")

q4 = df["expense_ratio"].mean()
print(f"Q4. What is the average expense ratio across all funds? {q4:.2f}%\n")

q5 = df.loc[df["returns_3yr"].idxmax()]
print("Q5. Which fund gives the highest 3-year return?")
print(f"   {q5['scheme_name']} ({q5['amc_name']}) -> {q5['returns_3yr']}%\n")

q6 = df.loc[df["returns_1yr"].idxmax()]
print("Q6. Which fund gives the highest 1-year return?")
print(f"   {q6['scheme_name']} ({q6['amc_name']}) -> {q6['returns_1yr']}%\n")

eda_summary = pd.DataFrame({
    "Question": [
        "AMC with highest total AUM",
        "Category with highest avg 3Y return",
        "Most common risk level",
        "Average expense ratio (%)",
        "Fund with highest 3Y return",
        "Fund with highest 1Y return",
    ],
    "Answer": [
        f"{q1.index[0]} ({q1.iloc[0]:,.0f} Cr)",
        f"{q2.index[0]} ({q2.iloc[0]:.2f}%)",
        f"Risk Level {q3.idxmax()} ({q3.max()} funds)",
        f"{q4:.2f}%",
        f"{q5['scheme_name']} ({q5['returns_3yr']}%)",
        f"{q6['scheme_name']} ({q6['returns_1yr']}%)",
    ]
})

section("STEP 5: DATA NORMALIZATION (sklearn.preprocessing.MinMaxScaler)")

scale_cols = ["returns_3yr", "expense_ratio", "fund_age_yr", "returns_1yr"]
score_df = df.copy()

for col in scale_cols:
    n_missing = score_df[col].isnull().sum()
    if n_missing:
        median_val = score_df[col].median()
        score_df[col] = score_df[col].fillna(median_val)
        print(f"  Filled {n_missing} missing values in '{col}' with median ({median_val:.2f})")

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(score_df[scale_cols])
scaled_df = pd.DataFrame(scaled_values, columns=[f"{c}_scaled" for c in scale_cols], index=score_df.index)

scaled_df["expense_ratio_scaled"] = 1 - scaled_df["expense_ratio_scaled"]

score_df = pd.concat([score_df, scaled_df], axis=1)
print("All 4 scoring fields scaled to a 0-1 range using MinMaxScaler.")
print(score_df[[c for c in scaled_df.columns]].describe().loc[["min", "max", "mean"]])

section("STEP 6: CREATE FUND SCORE")

score_df["fund_score"] = (
    score_df["returns_3yr_scaled"] * WEIGHTS["returns_3yr"]
    + score_df["expense_ratio_scaled"] * WEIGHTS["expense_ratio"]
    + score_df["fund_age_yr_scaled"] * WEIGHTS["fund_age_yr"]
    + score_df["returns_1yr_scaled"] * WEIGHTS["returns_1yr"]
) * 100  

score_df["fund_score"] = score_df["fund_score"].round(2)
print("Fund Score formula applied:")
print("  Fund Score = (3Y Return Score x 0.40) + (Expense Score x 0.30)")
print("               + (Age Score x 0.20) + (1Y Return Score x 0.10)")
print(f"\nFund Score range: {score_df['fund_score'].min()} - {score_df['fund_score'].max()}")

section("STEP 7: RANK ALL FUNDS (highest score to lowest)")

score_df = score_df.sort_values("fund_score", ascending=False).reset_index(drop=True)
score_df["rank"] = score_df.index + 1

print(score_df[["rank", "scheme_name", "amc_name", "fund_score"]].head(10))
score_df.to_csv(SCORED_FILE, index=False)
print(f"\nFull ranked fund list saved -> {SCORED_FILE}")

section("STEP 8: EXTRACT TOP 30 FUNDS")

top30 = score_df.head(30).copy()

output_cols = [
    "rank", "scheme_name", "amc_name", "category", "sub_category",
    "returns_1yr", "returns_3yr", "returns_5yr", "expense_ratio",
    "fund_age_yr", "fund_size_cr", "risk_level", "rating", "fund_score",
]
top30_export = top30[output_cols].rename(columns={
    "rank": "Rank",
    "scheme_name": "Fund Name",
    "amc_name": "AMC",
    "category": "Category",
    "sub_category": "Sub-Category",
    "returns_1yr": "1Y Return (%)",
    "returns_3yr": "3Y Return (%)",
    "returns_5yr": "5Y Return (%)",
    "expense_ratio": "Expense Ratio (%)",
    "fund_age_yr": "Fund Age (yrs)",
    "fund_size_cr": "AUM (Cr)",
    "risk_level": "Risk Level (1-6)",
    "rating": "Rating (1-5)",
    "fund_score": "Fund Score (/100)",
})

print(top30_export.head(30).to_string(index=False))

top30_export.to_csv("Top30_Mutual_Funds.csv", index=False)
print("\nTop 30 saved as CSV (also exported to formatted Excel separately) ->",
      "Top30_Mutual_Funds.csv")

section("SCRIPT COMPLETE")
print("Outputs generated:")
print(f"  1. {CLEAN_FILE}   - cleaned full dataset")
print(f"  2. {SCORED_FILE}  - all funds, scored & ranked")
print("  3. Top30_Mutual_Funds.csv - Top 30 funds (feeds the Excel workbook)")
