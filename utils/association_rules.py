import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

class AssociationAnalyzer:
    """
    Kelas untuk analisis pola asosiasi menggunakan algoritma Apriori
    Subbab 3.8.8: Analisis Pola Asosiasi Menggunakan Apriori
    """
    
    def __init__(self, transaction_data, clustered_data):
        self.transaction_data = transaction_data.copy()
        self.clustered_data = clustered_data.copy()
        self.rules_by_cluster = {}
        
    def prepare_transactions_by_cluster(self, cluster_id):
        """
        Menyiapkan data transaksi untuk cluster tertentu
        dalam format basket untuk Apriori
        
        Format: Setiap baris = 1 transaksi, kolom = kategori produk (binary)
        """
        # Filter pelanggan di cluster ini
        cluster_customers = self.clustered_data[
            self.clustered_data['Cluster'] == cluster_id
        ]['CustomerID'].unique()
        
        # Filter transaksi untuk pelanggan di cluster ini
        cluster_transactions = self.transaction_data[
            self.transaction_data['CustomerID'].isin(cluster_customers)
        ]
        
        if len(cluster_transactions) == 0:
            print(f"    ⚠ Tidak ada transaksi untuk cluster {cluster_id}")
            return None, 0
        
        # Buat basket: group by Transaction_ID, ambil unique Product_Category
        baskets = (cluster_transactions
                   .groupby('Transaction_ID')['Product_Category']
                   .apply(lambda x: list(set(x)))  # Unique categories per transaction
                   .values)
        
        # Filter basket yang memiliki minimal 2 item (untuk asosiasi)
        baskets = [basket for basket in baskets if len(basket) >= 2]
        
        if len(baskets) < 5:
            print(f"    ⚠ Transaksi multi-item terlalu sedikit: {len(baskets)} basket")
            return None, len(baskets)
        
        # Encode ke format one-hot (binary matrix)
        te = TransactionEncoder()
        te_ary = te.fit(baskets).transform(baskets)
        df_basket = pd.DataFrame(te_ary, columns=te.columns_)
        
        return df_basket, len(baskets)
    
    def run_apriori(self, cluster_id, min_support=0.01, min_confidence=0.2, min_lift=1.0):
        """
        Analisis Pola Asosiasi Menggunakan Apriori (Subbab 3.8.8)
        
        PARAMETER DISESUAIKAN UNTUK MENEMUKAN RULES:
        - min_support: default 0.01 (1%) - lebih rendah dari sebelumnya
        - min_confidence: default 0.2 (20%) - lebih rendah
        - min_lift: default 1.0 - semua asosiasi positif
        """
        # Siapkan data transaksi
        df_basket, n_baskets = self.prepare_transactions_by_cluster(cluster_id)
        
        if df_basket is None:
            return None
        
        # Tampilkan info basket
        print(f"    📦 {n_baskets} transaksi multi-item, {len(df_basket.columns)} kategori unik")
        
        # Cek frequency setiap item
        item_freq = df_basket.sum() / len(df_basket)
        print(f"    📊 Top 5 kategori:")
        for cat, freq in item_freq.nlargest(5).items():
            print(f"       • {cat}: {freq:.2%}")
        
        # Temukan frequent itemsets dengan support lebih rendah
        try:
            frequent_itemsets = apriori(
                df_basket, 
                min_support=min_support, 
                use_colnames=True,
                low_memory=True
            )
        except Exception as e:
            print(f"    ❌ Error pada apriori: {str(e)}")
            return None
        
        if len(frequent_itemsets) == 0:
            print(f"    ⚠ Tidak ada frequent itemset dengan support ≥ {min_support}")
            print(f"    💡 Coba turunkan min_support ke {min_support/2:.4f}")
            return None
        
        print(f"    ✓ {len(frequent_itemsets)} frequent itemsets ditemukan")
        
        # Filter itemsets dengan minimal 2 items (untuk rules)
        frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) >= 2)]
        
        if len(frequent_itemsets) == 0:
            print(f"    ⚠ Tidak ada itemset dengan ≥2 items")
            return None
        
        print(f"    ✓ {len(frequent_itemsets)} itemsets dengan ≥2 items")
        
        # Generate association rules
        try:
            rules = association_rules(
                frequent_itemsets, 
                metric="confidence", 
                min_threshold=min_confidence
            )
        except Exception as e:
            print(f"    ❌ Error pada association_rules: {str(e)}")
            return None
        
        if len(rules) == 0:
            print(f"    ⚠ Tidak ada rules dengan confidence ≥ {min_confidence}")
            print(f"    💡 Coba turunkan min_confidence ke {min_confidence - 0.1:.2f}")
            return None
        
        print(f"    ✓ {len(rules)} rules dengan confidence ≥ {min_confidence}")
        
        # Filter berdasarkan lift
        rules = rules[rules['lift'] >= min_lift]
        
        if len(rules) == 0:
            print(f"    ⚠ Tidak ada rules dengan lift ≥ {min_lift}")
            return None
        
        # Tambahkan kolom cluster
        rules['Cluster'] = cluster_id
        
        # Konversi frozenset ke string untuk visualisasi
        rules['antecedents_str'] = rules['antecedents'].apply(
            lambda x: ', '.join([item.replace('_', ' ').title() for item in sorted(list(x))])
        )
        rules['consequents_str'] = rules['consequents'].apply(
            lambda x: ', '.join([item.replace('_', ' ').title() for item in sorted(list(x))])
        )
        
        # Tambahkan kolom untuk interpretasi lift
        rules['lift_interpretation'] = rules['lift'].apply(self._interpret_lift)
        
        # Urutkan berdasarkan lift (descending)
        rules = rules.sort_values('lift', ascending=False)
        
        print(f"    ✅ {len(rules)} association rules ditemukan (lift ≥ {min_lift})")
        
        return rules
    
    def _interpret_lift(self, lift_value):
        """Interpretasi nilai lift"""
        if lift_value >= 3:
            return "Sangat Kuat"
        elif lift_value >= 2:
            return "Kuat"
        elif lift_value >= 1.5:
            return "Sedang"
        elif lift_value > 1:
            return "Lemah"
        else:
            return "Negatif"
    
    def analyze_all_clusters(self, min_support=0.01, min_confidence=0.2, min_lift=1.0):
        """
        Menjalankan analisis Apriori untuk semua cluster
        
        DEFAULT PARAMETERS DISESUAIKAN:
        - min_support: 0.01 (1%) - lebih rendah untuk menemukan patterns
        - min_confidence: 0.2 (20%) - lebih lenient
        - min_lift: 1.0 - semua asosiasi positif
        """
        print("\n" + "="*70)
        print("ANALISIS POLA ASOSIASI APRIORI (Subbab 3.8.8)")
        print("="*70 + "\n")
        
        print(f"⚙️ Parameter:")
        print(f"   Minimum Support    : {min_support} ({min_support*100:.1f}%)")
        print(f"   Minimum Confidence : {min_confidence} ({min_confidence*100:.1f}%)")
        print(f"   Minimum Lift       : {min_lift}")
        print()
        
        # Analisis data transaksi terlebih dahulu
        print("📊 Analisis Data Transaksi:")
        total_transactions = self.transaction_data['Transaction_ID'].nunique()
        total_customers = self.transaction_data['CustomerID'].nunique()
        total_categories = self.transaction_data['Product_Category'].nunique()
        
        # Hitung transaksi multi-item
        items_per_transaction = (self.transaction_data
                                 .groupby('Transaction_ID')['Product_Category']
                                 .nunique())
        multi_item = (items_per_transaction >= 2).sum()
        multi_item_pct = multi_item / total_transactions * 100
        
        print(f"   Total Transaksi       : {total_transactions:,}")
        print(f"   Total Pelanggan       : {total_customers:,}")
        print(f"   Total Kategori        : {total_categories}")
        print(f"   Transaksi Multi-Item  : {multi_item:,} ({multi_item_pct:.1f}%)")
        print()
        
        if multi_item_pct < 10:
            print("   ⚠️ WARNING: Transaksi multi-item < 10%")
            print("   💡 Ini akan membatasi jumlah association rules yang ditemukan")
            print()
        
        all_rules = []
        unique_clusters = sorted(self.clustered_data['Cluster'].unique())
        
        print(f"🔍 Menganalisis {len(unique_clusters)} cluster...\n")
        
        total_rules_found = 0
        
        for cluster_id in unique_clusters:
            cluster_size = (self.clustered_data['Cluster'] == cluster_id).sum()
            cluster_transactions = self.transaction_data[
                self.transaction_data['CustomerID'].isin(
                    self.clustered_data[self.clustered_data['Cluster'] == cluster_id]['CustomerID']
                )
            ]['Transaction_ID'].nunique()
            
            print(f"📦 Cluster {cluster_id}:")
            print(f"   Pelanggan : {cluster_size}")
            print(f"   Transaksi : {cluster_transactions}")
            
            rules = self.run_apriori(
                cluster_id,
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift
            )
            
            if rules is not None and len(rules) > 0:
                self.rules_by_cluster[cluster_id] = rules
                all_rules.append(rules)
                total_rules_found += len(rules)
                
                # Tampilkan top 3 rules untuk cluster ini
                top_3 = rules.nlargest(min(3, len(rules)), 'lift')
                print(f"   🏆 Top {len(top_3)} Rules (berdasarkan Lift):")
                for idx, (_, row) in enumerate(top_3.iterrows(), 1):
                    print(f"      {idx}. {row['antecedents_str']} → {row['consequents_str']}")
                    print(f"         Support: {row['support']:.2%}, Confidence: {row['confidence']:.2%}, Lift: {row['lift']:.2f} ({row['lift_interpretation']})")
            
            print()
        
        # Gabungkan semua rules
        if all_rules:
            combined_rules = pd.concat(all_rules, ignore_index=True)
            
            print("="*70)
            print(f"✅ RINGKASAN HASIL APRIORI")
            print("="*70)
            print(f"Total Rules Ditemukan : {len(combined_rules):,}")
            print(f"Cluster dengan Rules  : {len(self.rules_by_cluster)} dari {len(unique_clusters)}")
            print(f"Avg Support           : {combined_rules['support'].mean():.2%}")
            print(f"Avg Confidence        : {combined_rules['confidence'].mean():.2%}")
            print(f"Avg Lift              : {combined_rules['lift'].mean():.2f}")
            print(f"Max Lift              : {combined_rules['lift'].max():.2f}")
            print()
            
            # Distribusi rules per cluster
            rules_per_cluster = combined_rules.groupby('Cluster').size()
            print(f"Rules per Cluster:")
            for cluster_id, count in rules_per_cluster.items():
                pct = count / len(combined_rules) * 100
                print(f"   Cluster {cluster_id}: {count:>3} rules ({pct:>5.1f}%)")
            
            print("="*70 + "\n")
            
            return combined_rules
        else:
            print("="*70)
            print("❌ TIDAK ADA ASSOCIATION RULES YANG DITEMUKAN")
            print("="*70)
            print()
            print("🔍 DIAGNOSIS:")
            print(f"   1. Transaksi multi-item: {multi_item_pct:.1f}%")
            if multi_item_pct < 20:
                print("      ⚠️ Terlalu rendah! Sebagian besar transaksi hanya 1 item.")
            print()
            print("💡 SOLUSI:")
            print("   1. Turunkan min_support lebih lanjut:")
            print(f"      min_support = {min_support/2:.4f} atau {min_support/5:.4f}")
            print()
            print("   2. Turunkan min_confidence:")
            print(f"      min_confidence = 0.1 (10%)")
            print()
            print("   3. Analisis per kategori daripada per cluster:")
            print("      • Lihat pola pembelian global")
            print("      • Fokus pada top categories")
            print()
            print("   4. Alternatif: Gunakan analisis frekuensi co-occurrence")
            print("      • Hitung berapa sering kategori muncul bersama")
            print("      • Tidak perlu threshold minimum")
            print("="*70 + "\n")
            
            # Tampilkan co-occurrence matrix sebagai alternatif
            print("📊 ALTERNATIF: Co-Occurrence Matrix")
            print("="*70)
            self._show_cooccurrence_analysis()
            
            return None
    
    def _show_cooccurrence_analysis(self):
        """
        Analisis alternatif: co-occurrence tanpa threshold
        """
        print("\nMenghitung co-occurrence antar kategori...\n")
        
        # Buat co-occurrence matrix
        from itertools import combinations
        
        # Group by transaction
        transactions_grouped = (self.transaction_data
                               .groupby('Transaction_ID')['Product_Category']
                               .apply(lambda x: list(set(x)))
                               .values)
        
        # Filter multi-item
        multi_item_trans = [t for t in transactions_grouped if len(t) >= 2]
        
        if len(multi_item_trans) == 0:
            print("❌ Tidak ada transaksi dengan 2+ items")
            return
        
        # Count co-occurrences
        cooccurrence = {}
        for transaction in multi_item_trans:
            for pair in combinations(sorted(transaction), 2):
                cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
        
        # Convert to DataFrame
        cooc_df = pd.DataFrame([
            {
                'Category_A': pair[0],
                'Category_B': pair[1],
                'Count': count,
                'Pct': count / len(multi_item_trans) * 100
            }
            for pair, count in cooccurrence.items()
        ]).sort_values('Count', ascending=False)
        
        print(f"Total {len(multi_item_trans)} transaksi multi-item")
        print(f"Ditemukan {len(cooc_df)} pasangan kategori yang muncul bersama\n")
        
        print("Top 10 Pasangan Kategori (Co-occurrence):")
        print("-" * 70)
        for idx, row in cooc_df.head(10).iterrows():
            cat_a = row['Category_A'].replace('_', ' ').title()
            cat_b = row['Category_B'].replace('_', ' ').title()
            print(f"{cat_a:<25} + {cat_b:<25} : {row['Count']:>3}× ({row['Pct']:>5.1f}%)")
        
        print("\n💡 Gunakan pasangan di atas untuk strategi bundling!")
        print("="*70 + "\n")
    
    def get_top_rules(self, cluster_id=None, n=10, metric='lift'):
        """
        Mendapatkan top-N rules berdasarkan metrik tertentu
        """
        if cluster_id is not None:
            if cluster_id in self.rules_by_cluster:
                rules = self.rules_by_cluster[cluster_id]
                return rules.nlargest(min(n, len(rules)), metric)
            else:
                return None
        else:
            # Gabungkan semua cluster
            if self.rules_by_cluster:
                all_rules = pd.concat(self.rules_by_cluster.values(), ignore_index=True)
                return all_rules.nlargest(min(n, len(all_rules)), metric)
            return None
    
    def generate_recommendations(self, cluster_id):
        """
        Generate rekomendasi strategi promosi berdasarkan association rules
        """
        # Ambil profil cluster
        cluster_data = self.clustered_data[self.clustered_data['Cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        
        avg_r = cluster_data['R_score'].mean()
        avg_f = cluster_data['F_score'].mean()
        avg_m = cluster_data['M_score'].mean()
        
        recommendations = {
            'cluster_id': cluster_id,
            'customer_count': cluster_size,
            'avg_rfm': {'R': avg_r, 'F': avg_f, 'M': avg_m},
            'avg_rfm_str': f"R={avg_r:.1f}, F={avg_f:.1f}, M={avg_m:.1f}",
            'strategies': []
        }
        
        # Cek apakah ada rules untuk cluster ini
        if cluster_id in self.rules_by_cluster:
            rules = self.rules_by_cluster[cluster_id]
            
            # Strategi 1: Bundling (rules dengan lift tinggi)
            top_bundling = rules.nlargest(min(3, len(rules)), 'lift')
            for _, rule in top_bundling.iterrows():
                recommendations['strategies'].append({
                    'type': 'Bundling',
                    'description': f"Paket Bundle: {rule['antecedents_str']} + {rule['consequents_str']}",
                    'rationale': f"Pelanggan yang beli {rule['antecedents_str']} sangat mungkin beli {rule['consequents_str']}",
                    'metrics': f"Support: {rule['support']:.2%}, Confidence: {rule['confidence']:.2%}, Lift: {rule['lift']:.2f}",
                    'lift': rule['lift'],
                    'confidence': rule['confidence']
                })
            
            # Strategi 2: Cross-selling (rules dengan confidence tinggi)
            top_cross_sell = rules.nlargest(min(3, len(rules)), 'confidence')
            for _, rule in top_cross_sell.iterrows():
                desc = f"Jika pelanggan beli {rule['antecedents_str']}, rekomendasikan {rule['consequents_str']}"
                if not any(s['description'] == desc for s in recommendations['strategies']):
                    recommendations['strategies'].append({
                        'type': 'Cross-selling',
                        'description': desc,
                        'rationale': f"{rule['confidence']:.0%} pelanggan yang beli {rule['antecedents_str']} juga beli {rule['consequents_str']}",
                        'metrics': f"Confidence: {rule['confidence']:.2%}, Lift: {rule['lift']:.2f}",
                        'lift': rule['lift'],
                        'confidence': rule['confidence']
                    })
        
        # Strategi 3: Berdasarkan karakteristik RFM cluster (SELALU ADA)
        if avg_r >= 4 and avg_f >= 4:
            recommendations['strategies'].append({
                'type': 'Loyalty Program',
                'description': 'Program VIP untuk pelanggan high-value',
                'rationale': 'Cluster ini adalah pelanggan loyal dengan transaksi baru dan sering',
                'metrics': 'Early access, exclusive offers, personal shopper',
                'lift': None,
                'confidence': None
            })
        elif avg_r <= 2:
            recommendations['strategies'].append({
                'type': 'Re-engagement',
                'description': 'Kampanye win-back dengan diskon khusus',
                'rationale': 'Pelanggan sudah lama tidak transaksi, perlu reaktivasi',
                'metrics': 'Email campaign dengan 20-30% discount + free shipping',
                'lift': None,
                'confidence': None
            })
        elif avg_f <= 2:
            recommendations['strategies'].append({
                'type': 'Retention',
                'description': 'Program peningkatan frekuensi pembelian',
                'rationale': 'Pelanggan jarang transaksi, perlu dorongan',
                'metrics': 'Reminder emails, new product announcements',
                'lift': None,
                'confidence': None
            })
        else:
            recommendations['strategies'].append({
                'type': 'Engagement',
                'description': 'Program peningkatan engagement',
                'rationale': 'Cluster dengan potensi pertumbuhan',
                'metrics': 'Personalized recommendations, loyalty points',
                'lift': None,
                'confidence': None
            })
        
        return recommendations
    
    def get_all_recommendations(self):
        """
        Generate rekomendasi untuk semua cluster
        """
        all_recommendations = []
        
        for cluster_id in sorted(self.clustered_data['Cluster'].unique()):
            rec = self.generate_recommendations(cluster_id)
            if rec:
                all_recommendations.append(rec)
        
        return all_recommendations