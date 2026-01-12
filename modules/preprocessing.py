import pandas as pd

def preprocess_data(df):
    df = df.dropna(subset=["CustomerID", "Transaction_Date"])
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])
    df["Total_Amount"] = df["Quantity"] * df["Avg_Price"]
    return df
