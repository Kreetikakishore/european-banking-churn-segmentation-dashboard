# =============================================
# PHASE 2 - Understanding the Dataset
# =============================================

import pandas as pd
import numpy as np

# ── Load the dataset ──────────────────────────
df = pd.read_csv('data/European_Bank.csv')

# ── Basic overview ────────────────────────────
print("=" * 50)
print("SHAPE OF DATASET")
print("=" * 50)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\n" + "=" * 50)
print("COLUMN NAMES & DATA TYPES")
print("=" * 50)
print(df.dtypes)

print("\n" + "=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("BASIC STATISTICS")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("CHURN DISTRIBUTION (Target Column)")
print("=" * 50)
churn_counts = df['Exited'].value_counts()
churn_pct = df['Exited'].value_counts(normalize=True) * 100
print(f"Stayed  (0) : {churn_counts[0]}  ({churn_pct[0]:.2f}%)")
print(f"Churned (1) : {churn_counts[1]}  ({churn_pct[1]:.2f}%)")

print("\n" + "=" * 50)
print("UNIQUE VALUES IN KEY COLUMNS")
print("=" * 50)
print(f"Geography : {df['Geography'].unique()}")
print(f"Gender    : {df['Gender'].unique()}")
print(f"Products  : {df['NumOfProducts'].unique()}")


df = df.drop(columns=['Year', 'CustomerId', 'Surname'])

print("=" * 50)
print("COLUMNS AFTER CLEANING")
print("=" * 50)
print(df.columns.tolist())

# ── Step 2: Fix data types ────────────────────
df['HasCrCard']      = df['HasCrCard'].astype('category')
df['IsActiveMember'] = df['IsActiveMember'].astype('category')
df['Exited']         = df['Exited'].astype('category')

print("\n" + "=" * 50)
print("UPDATED DATA TYPES")
print("=" * 50)
print(df.dtypes)

# ── Step 3: Create Age Groups ─────────────────
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0, 29, 45, 60, 100],
    labels=['Under 30', '30-45', '46-60', '60+']
)

# ── Step 4: Create Credit Score Bands ─────────
df['CreditBand'] = pd.cut(
    df['CreditScore'],
    bins=[0, 579, 669, 850],
    labels=['Low', 'Medium', 'High']
)

# ── Step 5: Create Tenure Groups ──────────────
df['TenureGroup'] = pd.cut(
    df['Tenure'],
    bins=[-1, 2, 6, 10],
    labels=['New (0-2 yrs)', 'Mid (3-6 yrs)', 'Long (7+ yrs)']
)

# ── Step 6: Create Balance Segments ───────────
def balance_segment(balance):
    if balance == 0:
        return 'Zero Balance'
    elif balance < 100000:
        return 'Low Balance'
    else:
        return 'High Balance'

df['BalanceSegment'] = df['Balance'].apply(balance_segment)

# ── Step 7: Verify all new columns ────────────
print("\n" + "=" * 50)
print("NEW SEGMENT COLUMNS CREATED")
print("=" * 50)
print(f"AgeGroup      : {df['AgeGroup'].unique().tolist()}")
print(f"CreditBand    : {df['CreditBand'].unique().tolist()}")
print(f"TenureGroup   : {df['TenureGroup'].unique().tolist()}")
print(f"BalanceSegment: {df['BalanceSegment'].unique().tolist()}")

print("\n" + "=" * 50)
print("SEGMENT VALUE COUNTS")
print("=" * 50)
print("\nAge Group Distribution:")
print(df['AgeGroup'].value_counts().sort_index())

print("\nCredit Band Distribution:")
print(df['CreditBand'].value_counts().sort_index())

print("\nTenure Group Distribution:")
print(df['TenureGroup'].value_counts().sort_index())

print("\nBalance Segment Distribution:")
print(df['BalanceSegment'].value_counts())

# ── Step 8: Save cleaned dataset ──────────────
df.to_csv('data/European_Bank_Cleaned.csv', index=False)

print("\n" + "=" * 50)
print("CLEANED DATASET SAVED")
print("=" * 50)
print("File saved to: data/European_Bank_Cleaned.csv")
print(f"Final shape  : {df.shape[0]} rows x {df.shape[1]} columns")


import pandas as pd
import numpy as np
df = pd.read_csv('data/European_Bank_Cleaned.csv')

# =============================================
# PHASE 4 - Exploratory Data Analysis (EDA)
# =============================================

import pandas as pd
import numpy as np

# ── Load the cleaned dataset ──────────────────
df = pd.read_csv('data/European_Bank_Cleaned.csv')
df['Exited'] = df['Exited'].astype(int)

def churn_rate(series):
    return round(series.mean() * 100, 2)

print("=" * 50)
print("1. OVERALL CHURN RATE")
print("=" * 50)
overall = churn_rate(df['Exited'])
print(f"Overall Churn Rate : {overall}%")
print(f"Total Customers    : {len(df)}")
print(f"Total Churned      : {df['Exited'].sum()}")
print(f"Total Retained     : {len(df) - df['Exited'].sum()}")

print("\n" + "=" * 50)
print("2. CHURN BY GEOGRAPHY")
print("=" * 50)
geo_churn = df.groupby('Geography')['Exited'].agg(['mean', 'sum', 'count'])
geo_churn.columns = ['Churn Rate', 'Churned', 'Total']
geo_churn['Churn Rate'] = (geo_churn['Churn Rate'] * 100).round(2)
print(geo_churn)

print("\n" + "=" * 50)
print("3. CHURN BY GENDER")
print("=" * 50)
gender_churn = df.groupby('Gender')['Exited'].agg(['mean', 'sum', 'count'])
gender_churn.columns = ['Churn Rate', 'Churned', 'Total']
gender_churn['Churn Rate'] = (gender_churn['Churn Rate'] * 100).round(2)
print(gender_churn)

print("\n" + "=" * 50)
print("4. CHURN BY AGE GROUP")
print("=" * 50)
age_churn = df.groupby('AgeGroup')['Exited'].agg(['mean', 'sum', 'count'])
age_churn.columns = ['Churn Rate', 'Churned', 'Total']
age_churn['Churn Rate'] = (age_churn['Churn Rate'] * 100).round(2)
print(age_churn)

print("\n" + "=" * 50)
print("5. CHURN BY CREDIT BAND")
print("=" * 50)
credit_churn = df.groupby('CreditBand')['Exited'].agg(['mean', 'sum', 'count'])
credit_churn.columns = ['Churn Rate', 'Churned', 'Total']
credit_churn['Churn Rate'] = (credit_churn['Churn Rate'] * 100).round(2)
print(credit_churn)

print("\n" + "=" * 50)
print("6. CHURN BY TENURE GROUP")
print("=" * 50)
tenure_churn = df.groupby('TenureGroup')['Exited'].agg(['mean', 'sum', 'count'])
tenure_churn.columns = ['Churn Rate', 'Churned', 'Total']
tenure_churn['Churn Rate'] = (tenure_churn['Churn Rate'] * 100).round(2)
print(tenure_churn)

print("\n" + "=" * 50)
print("7. CHURN BY BALANCE SEGMENT")
print("=" * 50)
balance_churn = df.groupby('BalanceSegment')['Exited'].agg(['mean', 'sum', 'count'])
balance_churn.columns = ['Churn Rate', 'Churned', 'Total']
balance_churn['Churn Rate'] = (balance_churn['Churn Rate'] * 100).round(2)
print(balance_churn)

print("\n" + "=" * 50)
print("8. CHURN BY ACTIVE MEMBERSHIP")
print("=" * 50)
active_churn = df.groupby('IsActiveMember')['Exited'].agg(['mean', 'sum', 'count'])
active_churn.columns = ['Churn Rate', 'Churned', 'Total']
active_churn['Churn Rate'] = (active_churn['Churn Rate'] * 100).round(2)
active_churn.index = ['Inactive', 'Active']
print(active_churn)

print("\n" + "=" * 50)
print("9. CHURN BY NUMBER OF PRODUCTS")
print("=" * 50)
product_churn = df.groupby('NumOfProducts')['Exited'].agg(['mean', 'sum', 'count'])
product_churn.columns = ['Churn Rate', 'Churned', 'Total']
product_churn['Churn Rate'] = (product_churn['Churn Rate'] * 100).round(2)
print(product_churn)

print("\n" + "=" * 50)
print("10. HIGH VALUE CUSTOMER CHURN")
print("=" * 50)
high_value = df[df['Balance'] >= 100000]
hv_churn   = churn_rate(high_value['Exited'])
hv_churned = high_value['Exited'].sum()
print(f"High Value Customers : {len(high_value)}")
print(f"High Value Churned   : {hv_churned}")
print(f"High Value Churn Rate: {hv_churn}%")

print("\n" + "=" * 50)
print("11. CHURN BY GEOGRAPHY x GENDER")
print("=" * 50)
geo_gender = df.groupby(['Geography', 'Gender'])['Exited'].agg(['mean', 'sum', 'count'])
geo_gender.columns = ['Churn Rate', 'Churned', 'Total']
geo_gender['Churn Rate'] = (geo_gender['Churn Rate'] * 100).round(2)
print(geo_gender)

print("\n" + "=" * 50)
print("12. CHURN BY GEOGRAPHY x AGE GROUP")
print("=" * 50)
geo_age = df.groupby(['Geography', 'AgeGroup'])['Exited'].agg(['mean', 'count'])
geo_age.columns = ['Churn Rate', 'Total']
geo_age['Churn Rate'] = (geo_age['Churn Rate'] * 100).round(2)
print(geo_age)

print("\n" + "=" * 50)
print("13. AVERAGE PROFILE: CHURNED vs RETAINED")
print("=" * 50)
profile = df.groupby('Exited')[['CreditScore', 'Age',
                                 'Tenure', 'Balance',
                                 'EstimatedSalary']].mean().round(2)
profile.index = ['Retained', 'Churned']
print(profile)


# =============================================
# PHASE 5 - KPI Calculations
# =============================================

import pandas as pd
import numpy as np

# ── Load cleaned dataset ──────────────────────
df = pd.read_csv('data/European_Bank_Cleaned.csv')
df['Exited'] = df['Exited'].astype(int)

def churn_rate(series):
    return round(series.mean() * 100, 2)

print("=" * 50)
print("KPI 1 — OVERALL CHURN RATE")
print("=" * 50)
overall_churn = churn_rate(df['Exited'])
total_customers = len(df)
total_churned   = df['Exited'].sum()
total_retained  = total_customers - total_churned
print(f"Overall Churn Rate : {overall_churn}%")
print(f"Total Customers    : {total_customers}")
print(f"Total Churned      : {total_churned}")
print(f"Total Retained     : {total_retained}")

print("\n" + "=" * 50)
print("KPI 2 — SEGMENT CHURN RATES")
print("=" * 50)
segments = {
    'Geography'     : 'Geography',
    'Gender'        : 'Gender',
    'AgeGroup'      : 'Age Group',
    'CreditBand'    : 'Credit Band',
    'TenureGroup'   : 'Tenure Group',
    'BalanceSegment': 'Balance Segment',
}
for col, label in segments.items():
    seg = df.groupby(col)['Exited'].mean() * 100
    print(f"\n{label}:")
    for k, v in seg.items():
        print(f"   {k:<20} : {v:.2f}%")

print("\n" + "=" * 50)
print("KPI 3 — HIGH VALUE CHURN RATIO")
print("=" * 50)
high_value        = df[df['Balance'] >= 100000]
hv_total          = len(high_value)
hv_churned        = high_value['Exited'].sum()
hv_churn_rate     = churn_rate(high_value['Exited'])
hv_churn_ratio    = round((hv_churned / total_churned) * 100, 2)
hv_revenue_risk   = round(high_value[high_value['Exited'] == 1]['Balance'].sum(), 2)
print(f"High Value Customers      : {hv_total}")
print(f"High Value Churned        : {hv_churned}")
print(f"High Value Churn Rate     : {hv_churn_rate}%")
print(f"% of Total Churn          : {hv_churn_ratio}%")
print(f"Total Balance at Risk (€) : {hv_revenue_risk:,.2f}")

print("\n" + "=" * 50)
print("KPI 4 — GEOGRAPHIC RISK INDEX")
print("=" * 50)
geo_risk = df.groupby('Geography')['Exited'].agg(['mean', 'sum', 'count'])
geo_risk.columns = ['Churn Rate', 'Churned', 'Total']
geo_risk['Churn Rate']   = (geo_risk['Churn Rate'] * 100).round(2)
geo_risk['Risk Index']   = (geo_risk['Churn Rate'] / overall_churn).round(2)
geo_risk['Revenue Risk'] = df[df['Exited'] == 1].groupby(
    'Geography')['Balance'].sum().round(2)
print(geo_risk)
print("\nRisk Index > 1.0 means higher than average churn risk")

print("\n" + "=" * 50)
print("KPI 5 — ENGAGEMENT DROP INDICATOR")
print("=" * 50)
active_churn   = churn_rate(df[df['IsActiveMember'] == 1]['Exited'])
inactive_churn = churn_rate(df[df['IsActiveMember'] == 0]['Exited'])
engagement_gap = round(inactive_churn - active_churn, 2)
inactive_total = len(df[df['IsActiveMember'] == 0])
inactive_pct   = round((inactive_total / total_customers) * 100, 2)
print(f"Active Member Churn Rate   : {active_churn}%")
print(f"Inactive Member Churn Rate : {inactive_churn}%")
print(f"Engagement Gap             : {engagement_gap}%")
print(f"Total Inactive Customers   : {inactive_total} ({inactive_pct}%)")
print(f"\nConclusion: Inactive customers churn {round(inactive_churn/active_churn, 1)}x more than active ones")

print("\n" + "=" * 50)
print("KPI SUMMARY DASHBOARD")
print("=" * 50)
print(f"""
┌─────────────────────────────────────────────┐
│           BANK CHURN KPI SUMMARY            │
├─────────────────────────────────────────────┤
│  Overall Churn Rate       : {overall_churn}%           │
│  Total Churned Customers  : {total_churned}              │
│  High Value Churn Rate    : {hv_churn_rate}%           │
│  High Value Balance Risk  : €{hv_revenue_risk:,.0f}   │
│  Highest Risk Country     : Germany (32.44%) │
│  Highest Risk Age Group   : 46-60 (51.12%)  │
│  Engagement Gap           : {engagement_gap}%           │
│  Most Loyal Segment       : 2 Products (7.58%)│
└─────────────────────────────────────────────┘
""")

# ── Save all KPIs to a summary file ───────────
kpi_summary = {
    'KPI'  : [
        'Overall Churn Rate',
        'Total Customers',
        'Total Churned',
        'Total Retained',
        'High Value Churn Rate',
        'High Value Customers',
        'High Value Churned',
        'Balance at Risk (EUR)',
        'Germany Churn Rate',
        'France Churn Rate',
        'Spain Churn Rate',
        'Active Member Churn Rate',
        'Inactive Member Churn Rate',
        'Engagement Gap',
        'Age 46-60 Churn Rate',
    ],
    'Value': [
        f"{overall_churn}%",
        total_customers,
        total_churned,
        total_retained,
        f"{hv_churn_rate}%",
        hv_total,
        hv_churned,
        f"€{hv_revenue_risk:,.2f}",
        "32.44%",
        "16.15%",
        "16.67%",
        f"{active_churn}%",
        f"{inactive_churn}%",
        f"{engagement_gap}%",
        "51.12%",
    ]
}

kpi_df = pd.DataFrame(kpi_summary)
kpi_df.to_csv('data/KPI_Summary.csv', index=False)
print("KPI Summary saved to: data/KPI_Summary.csv")