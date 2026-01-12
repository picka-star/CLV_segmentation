import pandas as pd

def compute_lrfm(df):
    snapshot_date = df["Transaction_Date"].max() + pd.Timedelta(days=1)

    lrfm = df.groupby("CustomerID").agg({
        "Tenure_Months": "max",
        "Transaction_Date": lambda x: (snapshot_date - x.max()).days,
        "Transaction_ID": "count",
        "Total_Amount": "sum"
    }).reset_index()

    lrfm.columns = ["CustomerID", "Length", "Recency", "Frequency", "Monetary"]
    return lrfm
