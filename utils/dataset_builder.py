import pandas as pd

def build_customer_dataset(rfm, category_props):
    """
    Menggabungkan skor RFM dengan proporsi kategori menjadi dataset pelanggan final.
    Sesuai subbab 3.8.5 Penyusunan Dataset Pelanggan pada proposal.
    """
    df = rfm.merge(category_props, on="CustomerID", how="left")
    df = df.fillna(0)
    return df
