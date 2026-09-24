# 📊 Mutual Fund Overview & Insights — Finding High Return & Low Risk Funds

A complete data analytics project that cleans, explores, and scores 800+ Indian mutual funds using **Python (pandas, sklearn)** and visualizes the results in an interactive **Power BI dashboard**.

---

## 🎯 Objective

Investors face hundreds of mutual fund options, making it hard to compare funds fairly — since return, cost, and fund age are all measured on different scales. This project builds a single, objective **Fund Score** using MinMax-normalized weighted scoring, ranks all funds, and extracts the **Top 30 best funds** for investors.

---

## 📁 Repository Contents

| File | Description |
|---|---|
| `comprehensive_mutual_funds_data.csv` | Raw, original dataset — 814 mutual fund schemes with 20 attributes each (returns, expense ratio, fund age, AMC, risk metrics, ratings, etc.) |
| `mutual_fund_scoring.py` | Python script that cleans the raw data, performs EDA, applies `sklearn.preprocessing.MinMaxScaler` for normalization, computes the weighted Fund Score, and ranks all funds |
| `Top30_Mutual_Funds.csv` | Final output — Top 30 funds ranked by Fund Score, ready for analysis or dashboarding |
| `Cleaned Data & Top 30 Funds Overview.pbix` | Power BI dashboard file — 2 pages with 12 interactive visualizations covering the cleaned dataset and the Top 30 funds |
| `Cleaned Data & Top 30 Funds Overview.pdf` | Static PDF export of the Power BI dashboard, for quick viewing without Power BI Desktop |
| `Executive_Summary.docx` | Executive summary covering project objective, business problem, challenges faced & solutions, methodology, key findings, and recommendations |

---

## 🧮 Methodology — Fund Scoring Formula

All four scoring fields are normalized to a 0–1 range using **MinMaxScaler**, then combined using the following weights:

| Field | Weight |
|---|---|
| 3-Year Return | 40% |
| Expense Ratio (inverted — lower is better) | 30% |
| Fund Age | 20% |
| 1-Year Return | 10% |
