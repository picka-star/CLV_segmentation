import pandas as pd

def build_category_proportions(df):
    df_cat = df.groupby(["CustomerID", "Product_Category"])["Quantity"].sum().reset_index()

    total_per_customer = df_cat.groupby("CustomerID")["Quantity"].sum().reset_index()
    total_per_customer.columns = ["CustomerID", "TotalQty"]

    df_cat = df_cat.merge(total_per_customer, on="CustomerID")
    df_cat["Prop"] = df_cat["Quantity"] / df_cat["TotalQty"]

    pivot = df_cat.pivot_table(
        index="CustomerID",
        columns="Product_Category",
        values="Prop",
        fill_value=0
    )

    pivot.columns = [f"prop_{c.lower().replace(' ', '_')}" for c in pivot.columns]
    pivot.reset_index(inplace=True)

    return pivot
