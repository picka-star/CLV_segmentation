"""
Script untuk menganalisis dataset dan memberikan rekomendasi parameter optimal
Jalankan: python analyze_dataset.py

GUNAKAN SCRIPT INI UNTUK:
1. Memahami karakteristik dataset Anda
2. Mendapatkan rekomendasi parameter K-Means
3. Mendapatkan rekomendasi parameter Apriori
4. Melihat preview patterns yang mungkin ditemukan
"""

import pandas as pd
import numpy as np
from itertools import combinations
import sys

def analyze_dataset(filepath):
    """
    Analisis lengkap dataset untuk menentukan parameter optimal
    """
    print("="*80)
    print("ANALISIS DATASET ONLINE STORE")
    print("="*80)
    print()
    
    # Load data
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        print(f"✅ Dataset loaded: {filepath}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    print()
    print("-"*80)
    print("1. STATISTIK DASAR")
    print("-"*80)
    
    # Basic stats
    total_rows = len(df)
    total_cols = len(df.columns)
    
    print(f"Total Rows     : {total_rows:,}")
    print(f"Total Columns  : {total_cols}")
    print()
    
    # Check required columns
    required_cols = ['CustomerID', 'Transaction_ID', 'Transaction_Date', 
                     'Product_Category', 'Quantity', 'Avg_Price']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️ Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    else:
        print(f"✅ All required columns present")
    
    print()
    print("-"*80)
    print("2. ANALISIS TRANSAKSI")
    print("-"*80)
    
    # Transaction analysis
    df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
    df['Transaction_ID'] = pd.to_numeric(df['Transaction_ID'], errors='coerce')
    df = df.dropna(subset=['CustomerID', 'Transaction_ID'])
    
    total_customers = df['CustomerID'].nunique()
    total_transactions = df['Transaction_ID'].nunique()
    total_categories = df['Product_Category'].nunique()
    
    print(f"Unique Customers   : {total_customers:,}")
    print(f"Unique Transactions: {total_transactions:,}")
    print(f"Unique Categories  : {total_categories}")
    print()
    
    # Items per transaction
    items_per_trans = df.groupby('Transaction_ID')['Product_Category'].nunique()
    avg_items = items_per_trans.mean()
    
    print(f"Avg Items/Transaction: {avg_items:.2f}")
    print()
    
    # Multi-item transactions (CRITICAL for Apriori)
    single_item = (items_per_trans == 1).sum()
    multi_item = (items_per_trans >= 2).sum()
    multi_item_pct = multi_item / total_transactions * 100
    
    print(f"Single-Item Transactions: {single_item:,} ({single_item/total_transactions*100:.1f}%)")
    print(f"Multi-Item Transactions : {multi_item:,} ({multi_item_pct:.1f}%)")
    print()
    
    # Distribution
    print("Distribution of Items per Transaction:")
    for i in range(1, min(6, items_per_trans.max()+1)):
        count = (items_per_trans == i).sum()
        pct = count / total_transactions * 100
        print(f"  {i} items: {count:>6,} ({pct:>5.1f}%)")
    
    if items_per_trans.max() > 5:
        count = (items_per_trans > 5).sum()
        pct = count / total_transactions * 100
        print(f"  6+ items: {count:>5,} ({pct:>5.1f}%)")
    
    print()
    print("-"*80)
    print("3. ANALISIS KATEGORI PRODUK")
    print("-"*80)
    
    # Category analysis
    category_freq = df['Product_Category'].value_counts()
    
    print(f"\nTop 15 Categories:")
    for i, (cat, count) in enumerate(category_freq.head(15).items(), 1):
        pct = count / len(df) * 100
        print(f"  {i:2}. {cat:<30} : {count:>6,} ({pct:>5.1f}%)")
    
    print()
    print("-"*80)
    print("4. CO-OCCURRENCE ANALYSIS (Preview)")
    print("-"*80)
    
    # Co-occurrence
    transactions_grouped = (df.groupby('Transaction_ID')['Product_Category']
                           .apply(lambda x: list(set(x)))
                           .values)
    
    multi_item_trans = [t for t in transactions_grouped if len(t) >= 2]
    
    if len(multi_item_trans) > 0:
        cooccurrence = {}
        for transaction in multi_item_trans:
            for pair in combinations(sorted(transaction), 2):
                cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
        
        sorted_cooc = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 15 Product Pairs (Co-occurrence):")
        for i, (pair, count) in enumerate(sorted_cooc[:15], 1):
            pct = count / len(multi_item_trans) * 100
            cat_a = pair[0].replace('_', ' ').title()
            cat_b = pair[1].replace('_', ' ').title()
            print(f"  {i:2}. {cat_a:<25} + {cat_b:<25} : {count:>4}× ({pct:>5.1f}%)")
    else:
        print("\n⚠️ NO multi-item transactions found!")
        print("   Association rules CANNOT be generated from this data.")
    
    print()
    print("="*80)
    print("REKOMENDASI PARAMETER")
    print("="*80)
    print()
    
    # Recommendations for K-Means
    print("-"*80)
    print("K-MEANS CLUSTERING")
    print("-"*80)
    
    if total_customers < 100:
        rec_k = "2-3"
    elif total_customers < 300:
        rec_k = "3-4"
    elif total_customers < 600:
        rec_k = "4-5"
    else:
        rec_k = "4-6"
    
    print(f"Recommended n_clusters: {rec_k}")
    print(f"Reasoning: {total_customers} unique customers")
    print()
    
    # Recommendations for Apriori
    print("-"*80)
    print("APRIORI ASSOCIATION RULES")
    print("-"*80)
    
    if multi_item_pct < 5:
        status = "❌ SANGAT SULIT"
        rec_support = "0.001 (0.1%)"
        rec_confidence = "0.05 (5%)"
        rec_lift = "1.0"
        note = "Multi-item terlalu rendah. Gunakan Co-Occurrence Analysis sebagai alternatif."
    elif multi_item_pct < 10:
        status = "⚠️ SULIT"
        rec_support = "0.005 (0.5%)"
        rec_confidence = "0.1 (10%)"
        rec_lift = "1.0"
        note = "Turunkan threshold secara agresif. Expect sedikit rules."
    elif multi_item_pct < 20:
        status = "⚠️ CHALLENGING"
        rec_support = "0.01 (1%)"
        rec_confidence = "0.15 (15%)"
        rec_lift = "1.0"
        note = "Parameter rendah diperlukan. Moderate number of rules expected."
    elif multi_item_pct < 30:
        status = "✅ FEASIBLE"
        rec_support = "0.02 (2%)"
        rec_confidence = "0.2 (20%)"
        rec_lift = "1.0"
        note = "Good chance of finding meaningful rules."
    else:
        status = "✅✅ EXCELLENT"
        rec_support = "0.03 (3%)"
        rec_confidence = "0.25 (25%)"
        rec_lift = "1.2"
        note = "High multi-item percentage. Many rules expected!"
    
    print(f"Status: {status}")
    print(f"Multi-item %: {multi_item_pct:.1f}%")
    print()
    print(f"Recommended Parameters:")
    print(f"  min_support    : {rec_support}")
    print(f"  min_confidence : {rec_confidence}")
    print(f"  min_lift       : {rec_lift}")
    print()
    print(f"Note: {note}")
    print()
    
    # Expected results
    if multi_item_pct >= 5:
        if multi_item_pct < 10:
            exp_rules = "5-20 rules"
        elif multi_item_pct < 20:
            exp_rules = "20-60 rules"
        elif multi_item_pct < 30:
            exp_rules = "50-150 rules"
        else:
            exp_rules = "100-300 rules"
        
        print(f"Expected Results: {exp_rules}")
    
    print()
    print("="*80)
    print("SUMMARY & ACTION PLAN")
    print("="*80)
    print()
    
    print("1. Upload dataset ke Streamlit app")
    print()
    print("2. Set parameters di sidebar:")
    print(f"   - Jumlah Cluster: {rec_k.split('-')[0]}")
    print(f"   - Min Support: {rec_support.split()[0]}")
    print(f"   - Min Confidence: {rec_confidence.split()[0]}")
    print(f"   - Min Lift: {rec_lift}")
    print()
    print("3. Jalankan analisis:")
    print("   - Tab 1: Dashboard (overview)")
    print("   - Tab 2: Preprocessing (data cleaning)")
    print("   - Tab 3: RFM Analysis")
    print("   - Tab 4: Clustering (segmentation)")
    print("   - Tab 5: Apriori (association rules)")
    print()
    
    if multi_item_pct < 10:
        print("⚠️ IMPORTANT:")
        print("   Karena multi-item% rendah, fokuskan analisis pada:")
        print("   - Segmentasi RFM (Tab 4) - ini akan tetap valuable")
        print("   - Co-Occurrence Analysis (Tab 5) - alternatif dari Apriori")
        print("   - Strategi promosi berdasarkan karakteristik cluster")
        print()
    
    print("="*80)
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_dataset.py <filepath>")
        print()
        print("Example:")
        print("  python analyze_dataset.py data/dataonlineshop2024kaggle.csv")
        print("  python analyze_dataset.py data/tabledataonlineshop2024kaggle.xlsx")
    else:
        filepath = sys.argv[1]
        analyze_dataset(filepath)