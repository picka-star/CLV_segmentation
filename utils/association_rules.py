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
            return None
        
        # Buat basket: group by Transaction_ID, ambil list Product_Category
        baskets = (cluster_transactions
                   .groupby('Transaction_ID')['Product_Category']
                   .apply(list)
                   .values)
        
        # Encode ke format one-hot (binary matrix)
        te = TransactionEncoder()
        te_ary = te.fit(baskets).transform(baskets)
        df_basket = pd.DataFrame(te_ary, columns=te.columns_)
        
        return df_basket
    
    def run_apriori(self, cluster_id, min_support=0.01, min_confidence=0.3, min_lift=1.0):
        """
        Analisis Pola Asosiasi Menggunakan Apriori (Subbab 3.8.8)
        
        Algoritma Apriori untuk menemukan:
        1. Frequent itemsets: kombinasi kategori yang sering muncul bersama
        2. Association rules: pola hubungan A → B
        
        Metrik:
        - Support(A→B) = P(A ∩ B) = frekuensi kemunculan bersama
        - Confidence(A→B) = P(B|A) = probabilitas B dibeli jika A dibeli
        - Lift(A→B) = Confidence(A→B) / Support(B) = kekuatan asosiasi
        
        Parameters:
        - min_support: minimum frekuensi kemunculan (default: 0.01 = 1%)
        - min_confidence: minimum tingkat kepercayaan (default: 0.3 = 30%)
        - min_lift: minimum kekuatan asosiasi (default: 1.0)
        
        Interpretasi Lift:
        - Lift > 1: Asosiasi positif (B lebih mungkin dibeli jika A dibeli)
        - Lift = 1: Independen (tidak ada hubungan)
        - Lift < 1: Asosiasi negatif (B lebih jarang dibeli jika A dibeli)
        """
        # Siapkan data transaksi
        df_basket = self.prepare_transactions_by_cluster(cluster_id)
        
        if df_basket is None or len(df_basket) < 10:
            print(f"  ⚠ Cluster {cluster_id}: Data transaksi terlalu sedikit (< 10)")
            return None
        
        # Temukan frequent itemsets
        try:
            frequent_itemsets = apriori(df_basket, min_support=min_support, use_colnames=True)
        except Exception as e:
            print(f"  ⚠ Cluster {cluster_id}: Error pada apriori - {str(e)}")
            return None
        
        if len(frequent_itemsets) == 0:
            print(f"  ⚠ Cluster {cluster_id}: Tidak ada frequent itemset dengan support ≥ {min_support}")
            return None
        
        # Generate association rules
        try:
            rules = association_rules(
                frequent_itemsets, 
                metric="confidence", 
                min_threshold=min_confidence,
                num_itemsets=len(frequent_itemsets)
            )
        except Exception as e:
            print(f"  ⚠ Cluster {cluster_id}: Error pada association_rules - {str(e)}")
            return None
        
        if len(rules) == 0:
            print(f"  ⚠ Cluster {cluster_id}: Tidak ada rules dengan confidence ≥ {min_confidence}")
            return None
        
        # Filter berdasarkan lift
        rules = rules[rules['lift'] >= min_lift]
        
        if len(rules) == 0:
            print(f"  ⚠ Cluster {cluster_id}: Tidak ada rules dengan lift ≥ {min_lift}")
            return None
        
        # Tambahkan kolom cluster
        rules['Cluster'] = cluster_id
        
        # Konversi frozenset ke string untuk visualisasi
        rules['antecedents_str'] = rules['antecedents'].apply(
            lambda x: ', '.join([item.replace('_', ' ').title() for item in list(x)])
        )
        rules['consequents_str'] = rules['consequents'].apply(
            lambda x: ', '.join([item.replace('_', ' ').title() for item in list(x)])
        )
        
        # Tambahkan kolom untuk interpretasi lift
        rules['lift_interpretation'] = rules['lift'].apply(self._interpret_lift)
        
        # Urutkan berdasarkan lift (descending)
        rules = rules.sort_values('lift', ascending=False)
        
        print(f"  ✓ Cluster {cluster_id}: {len(rules)} association rules ditemukan")
        
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
    
    def analyze_all_clusters(self, min_support=0.01, min_confidence=0.3, min_lift=1.0):
        """
        Menjalankan analisis Apriori untuk semua cluster
        """
        print("\n" + "="*70)
        print("ANALISIS POLA ASOSIASI APRIORI (Subbab 3.8.8)")
        print("="*70 + "\n")
        
        print(f"Parameter:")
        print(f"  Minimum Support    : {min_support} ({min_support*100:.1f}%)")
        print(f"  Minimum Confidence : {min_confidence} ({min_confidence*100:.1f}%)")
        print(f"  Minimum Lift       : {min_lift}")
        print()
        
        all_rules = []
        unique_clusters = sorted(self.clustered_data['Cluster'].unique())
        
        print(f"Menganalisis {len(unique_clusters)} cluster...\n")
        
        for cluster_id in unique_clusters:
            cluster_size = (self.clustered_data['Cluster'] == cluster_id).sum()
            print(f"Cluster {cluster_id} ({cluster_size} pelanggan):")
            
            rules = self.run_apriori(
                cluster_id,
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift
            )
            
            if rules is not None and len(rules) > 0:
                self.rules_by_cluster[cluster_id] = rules
                all_rules.append(rules)
                
                # Tampilkan top 3 rules untuk cluster ini
                top_3 = rules.nlargest(3, 'lift')
                print(f"  Top 3 Rules (berdasarkan Lift):")
                for idx, row in top_3.iterrows():
                    print(f"    • {row['antecedents_str']} → {row['consequents_str']}")
                    print(f"      Support: {row['support']:.2%}, Confidence: {row['confidence']:.2%}, Lift: {row['lift']:.2f} ({row['lift_interpretation']})")
            
            print()
        
        # Gabungkan semua rules
        if all_rules:
            combined_rules = pd.concat(all_rules, ignore_index=True)
            
            print("="*70)
            print(f"RINGKASAN HASIL APRIORI")
            print("="*70)
            print(f"Total Rules      : {len(combined_rules):,}")
            print(f"Cluster dengan Rules: {len(self.rules_by_cluster)}")
            print(f"Avg Confidence   : {combined_rules['confidence'].mean():.2%}")
            print(f"Avg Lift         : {combined_rules['lift'].mean():.2f}")
            print(f"Max Lift         : {combined_rules['lift'].max():.2f}")
            print("="*70 + "\n")
            
            return combined_rules
        else:
            print("="*70)
            print("⚠ TIDAK ADA ASSOCIATION RULES YANG DITEMUKAN")
            print("="*70)
            print("Saran:")
            print("  1. Turunkan min_support (coba 0.01 atau 0.005)")
            print("  2. Turunkan min_confidence (coba 0.2)")
            print("  3. Turunkan min_lift (coba 1.0)")
            print("  4. Periksa distribusi transaksi per cluster")
            print("="*70 + "\n")
            return None
    
    def get_top_rules(self, cluster_id=None, n=10, metric='lift'):
        """
        Mendapatkan top-N rules berdasarkan metrik tertentu
        """
        if cluster_id is not None:
            if cluster_id in self.rules_by_cluster:
                return self.rules_by_cluster[cluster_id].nlargest(n, metric)
            else:
                return None
        else:
            # Gabungkan semua cluster
            if self.rules_by_cluster:
                all_rules = pd.concat(self.rules_by_cluster.values(), ignore_index=True)
                return all_rules.nlargest(n, metric)
            return None
    
    def generate_recommendations(self, cluster_id):
        """
        Generate rekomendasi strategi promosi berdasarkan association rules
        
        Strategi yang dihasilkan:
        1. Bundling: Paket produk berdasarkan lift tinggi
        2. Cross-selling: Rekomendasi produk berdasarkan confidence tinggi
        3. Upselling: Produk dengan monetary value lebih tinggi
        """
        if cluster_id not in self.rules_by_cluster:
            return None
        
        rules = self.rules_by_cluster[cluster_id]
        
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
        
        # Strategi 1: Bundling (rules dengan lift tinggi)
        top_bundling = rules.nlargest(min(5, len(rules)), 'lift')
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
        top_cross_sell = rules.nlargest(min(5, len(rules)), 'confidence')
        for _, rule in top_cross_sell.iterrows():
            # Hindari duplikasi dengan bundling
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
        
        # Strategi 3: Berdasarkan karakteristik RFM cluster
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
        
        return recommendations
    
    def get_all_recommendations(self):
        """
        Generate rekomendasi untuk semua cluster
        """
        all_recommendations = []
        
        for cluster_id in sorted(self.rules_by_cluster.keys()):
            rec = self.generate_recommendations(cluster_id)
            if rec:
                all_recommendations.append(rec)
        
        return all_recommendations