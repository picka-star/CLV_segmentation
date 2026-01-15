import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class CustomerClustering:
    """
    Kelas untuk melakukan klasterisasi pelanggan menggunakan K-Means
    Subbab 3.8.6: Normalisasi dan Penskalaan Data
    Subbab 3.8.7: Klasterisasi Pelanggan Menggunakan K-Means
    """
    
    def __init__(self, customer_dataset):
        self.customer_dataset = customer_dataset.copy()
        self.scaled_data = None
        self.scaler = None
        self.optimal_k = None
        self.kmeans_model = None
        self.clustered_data = None
        self.feature_names = None
        
    def normalize_data(self):
        """
        Normalisasi dan Penskalaan Data (Subbab 3.8.6)
        Menggunakan StandardScaler
        
        z = (x - μ) / σ  (Persamaan 3.7)
        
        di mana:
        - x = nilai asli fitur
        - μ = rata-rata fitur
        - σ = standar deviasi fitur
        """
        print("\n" + "="*70)
        print("NORMALISASI DAN PENSKALAAN DATA (Subbab 3.8.6)")
        print("="*70 + "\n")
        
        # Pilih fitur untuk scaling: R_score, F_score, M_score + proporsi kategori
        self.feature_names = ['R_score', 'F_score', 'M_score'] + \
                            [col for col in self.customer_dataset.columns if col.startswith('prop_')]
        
        X = self.customer_dataset[self.feature_names]
        
        print(f"Fitur yang dinormalisasi:")
        print(f"  - Skor RFM: R_score, F_score, M_score")
        print(f"  - Proporsi kategori: {len([c for c in self.feature_names if c.startswith('prop_')])} variabel")
        print(f"  Total: {len(self.feature_names)} fitur\n")
        
        # Standarisasi menggunakan Persamaan 3.7
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(X)
        
        # Verifikasi: mean ≈ 0, std ≈ 1
        mean_check = np.abs(self.scaled_data.mean()) < 0.01
        std_check = np.abs(self.scaled_data.std() - 1.0) < 0.1
        
        print(f"✓ Data dinormalisasi menggunakan StandardScaler")
        print(f"  Mean: {self.scaled_data.mean():.6f} {'✓' if mean_check else '⚠'} (target ≈ 0)")
        print(f"  Std : {self.scaled_data.std():.6f} {'✓' if std_check else '⚠'} (target ≈ 1)")
        
        if not (mean_check and std_check):
            print("  ⚠ Perhatian: Normalisasi mungkin tidak sempurna")
        
        return self
    
    def determine_optimal_k(self, k_range=(2, 11)):
        """
        Menentukan jumlah klaster optimal menggunakan:
        1. Elbow Method (WCSS - Within-Cluster Sum of Squares)
        2. Silhouette Coefficient
        3. Davies-Bouldin Index
        4. Calinski-Harabasz Index
        """
        print("\n" + "="*70)
        print("PENENTUAN JUMLAH KLASTER OPTIMAL")
        print("="*70 + "\n")
        
        wcss = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        k_values = range(k_range[0], k_range[1])
        
        print("Menghitung metrik evaluasi untuk setiap k...")
        for k in k_values:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(self.scaled_data)
            
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, labels))
            davies_bouldin_scores.append(davies_bouldin_score(self.scaled_data, labels))
            calinski_harabasz_scores.append(calinski_harabasz_score(self.scaled_data, labels))
            
            print(f"  k={k}: Silhouette={silhouette_scores[-1]:.4f}, DB={davies_bouldin_scores[-1]:.4f}")
        
        # Pilih k berdasarkan Silhouette Score tertinggi
        best_silhouette_idx = np.argmax(silhouette_scores)
        self.optimal_k = k_values[best_silhouette_idx]
        
        print(f"\n✓ Jumlah klaster optimal (berdasarkan Silhouette): {self.optimal_k}")
        print(f"  Silhouette Score: {silhouette_scores[best_silhouette_idx]:.4f}")
        print(f"  Davies-Bouldin Index: {davies_bouldin_scores[best_silhouette_idx]:.4f} (lebih rendah = lebih baik)")
        
        # Rekomendasi alternatif berdasarkan Davies-Bouldin (lebih rendah lebih baik)
        best_db_idx = np.argmin(davies_bouldin_scores)
        if best_db_idx != best_silhouette_idx:
            print(f"\n  💡 Alternatif berdasarkan Davies-Bouldin: k={k_values[best_db_idx]}")
        
        return {
            'k_values': list(k_values),
            'wcss': wcss,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'calinski_harabasz_scores': calinski_harabasz_scores,
            'optimal_k': self.optimal_k
        }
    
    def fit_kmeans(self, n_clusters=None):
        """
        Klasterisasi Pelanggan Menggunakan K-Means (Subbab 3.8.7)
        
        Algoritma K-Means dengan:
        - Inisialisasi: k-means++ (untuk centroid awal yang lebih baik)
        - Metrik jarak: Euclidean Distance
        
        d(p,q) = √(Σ(qi - pi)²)  (Persamaan 2.1 dari proposal)
        
        Proses iteratif:
        1. Assignment: Tempatkan setiap pelanggan ke centroid terdekat
        2. Update: Hitung ulang posisi centroid sebagai rata-rata anggota
        3. Ulangi hingga konvergen
        """
        print("\n" + "="*70)
        print("KLASTERISASI K-MEANS (Subbab 3.8.7)")
        print("="*70 + "\n")
        
        if n_clusters is None:
            n_clusters = self.optimal_k if self.optimal_k else 4
        
        print(f"Menjalankan K-Means clustering...")
        print(f"  Jumlah cluster: {n_clusters}")
        print(f"  Inisialisasi: k-means++")
        print(f"  Metrik jarak: Euclidean (Persamaan 2.1)")
        print(f"  Max iterasi: 300\n")
        
        # Inisialisasi dan fit K-Means
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            random_state=42,
            n_init=10,
            max_iter=300,
            verbose=0
        )
        
        cluster_labels = self.kmeans_model.fit_predict(self.scaled_data)
        
        # Tambahkan label klaster ke dataset
        self.clustered_data = self.customer_dataset.copy()
        self.clustered_data['Cluster'] = cluster_labels
        
        # Evaluasi kualitas klaster
        silhouette = silhouette_score(self.scaled_data, cluster_labels)
        davies_bouldin = davies_bouldin_score(self.scaled_data, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(self.scaled_data, cluster_labels)
        
        print(f"✓ K-Means clustering selesai dengan {n_clusters} klaster")
        print(f"\nMetrik Evaluasi Klaster:")
        print(f"  Silhouette Score      : {silhouette:.4f} (range: -1 to 1, >0.5 = baik)")
        print(f"  Davies-Bouldin Index  : {davies_bouldin:.4f} (lebih rendah = lebih baik)")
        print(f"  Calinski-Harabasz     : {calinski_harabasz:.2f} (lebih tinggi = lebih baik)")
        print(f"  Iterasi konvergensi   : {self.kmeans_model.n_iter_}")
        
        # Interpretasi kualitas
        if silhouette > 0.5:
            quality = "Baik ✓"
        elif silhouette > 0.3:
            quality = "Cukup"
        else:
            quality = "Perlu perbaikan ⚠"
        print(f"\n  Kualitas klasterisasi: {quality}")
        
        # Distribusi pelanggan per cluster
        cluster_dist = pd.Series(cluster_labels).value_counts().sort_index()
        print(f"\nDistribusi Pelanggan per Cluster:")
        for cluster_id, count in cluster_dist.items():
            pct = count / len(cluster_labels) * 100
            print(f"  Cluster {cluster_id}: {count:>5} pelanggan ({pct:>5.1f}%)")
        
        return self
    
    def get_cluster_profiles(self):
        """
        Membuat profil karakteristik setiap klaster
        Analisis untuk setiap cluster:
        1. Statistik RFM (mean, median)
        2. Top 3 kategori produk yang paling sering dibeli
        3. Karakteristik perilaku
        """
        if self.clustered_data is None:
            raise ValueError("Run fit_kmeans() terlebih dahulu")
        
        print("\n" + "="*70)
        print("PROFIL KARAKTERISTIK CLUSTER")
        print("="*70 + "\n")
        
        # Statistik RFM per klaster
        rfm_profile = self.clustered_data.groupby('Cluster').agg({
            'Recency': ['mean', 'median', 'min', 'max'],
            'Frequency': ['mean', 'median', 'min', 'max'],
            'Monetary': ['mean', 'median', 'min', 'max'],
            'R_score': 'mean',
            'F_score': 'mean',
            'M_score': 'mean',
            'CustomerID': 'count'
        }).round(2)
        
        rfm_profile.columns = ['_'.join(col).strip() for col in rfm_profile.columns.values]
        rfm_profile = rfm_profile.rename(columns={'CustomerID_count': 'Customer_Count'})
        
        # Top 3 kategori per klaster
        prop_cols = [col for col in self.clustered_data.columns if col.startswith('prop_')]
        category_profile = self.clustered_data.groupby('Cluster')[prop_cols].mean()
        
        top_categories = {}
        cluster_characteristics = {}
        
        for cluster in sorted(self.clustered_data['Cluster'].unique()):
            cluster_data = self.clustered_data[self.clustered_data['Cluster'] == cluster]
            
            # Top 3 kategori
            top_3 = category_profile.loc[cluster].nlargest(3)
            top_categories[cluster] = [
                (cat.replace('prop_', ''), f"{val:.2%}") 
                for cat, val in top_3.items()
            ]
            
            # Karakteristik cluster
            avg_r = cluster_data['R_score'].mean()
            avg_f = cluster_data['F_score'].mean()
            avg_m = cluster_data['M_score'].mean()
            
            # Klasifikasi karakteristik
            if avg_r >= 4 and avg_f >= 4 and avg_m >= 4:
                char = "High-Value Champions"
                desc = "Pelanggan terbaik dengan transaksi baru, sering, dan nilai tinggi"
            elif avg_f >= 4 and avg_r >= 3:
                char = "Loyal Customers"
                desc = "Pelanggan setia dengan frekuensi tinggi"
            elif avg_r >= 4:
                char = "Potential Loyalists"
                desc = "Pelanggan baru dengan potensi tinggi"
            elif avg_r <= 2 and avg_f >= 3:
                char = "At Risk"
                desc = "Pelanggan lama yang perlu re-engagement"
            elif avg_r <= 2 and avg_f <= 2:
                char = "Hibernating/Lost"
                desc = "Pelanggan yang sudah lama tidak aktif"
            else:
                char = "Need Attention"
                desc = "Pelanggan yang memerlukan perhatian khusus"
            
            cluster_characteristics[cluster] = {
                'name': char,
                'description': desc,
                'avg_rfm': f"R={avg_r:.1f}, F={avg_f:.1f}, M={avg_m:.1f}"
            }
        
        # Tampilkan profil
        for cluster in sorted(self.clustered_data['Cluster'].unique()):
            char = cluster_characteristics[cluster]
            count = (self.clustered_data['Cluster'] == cluster).sum()
            pct = count / len(self.clustered_data) * 100
            
            print(f"Cluster {cluster}: {char['name']}")
            print(f"  Jumlah: {count:,} pelanggan ({pct:.1f}%)")
            print(f"  Karakteristik: {char['description']}")
            print(f"  Skor RFM rata-rata: {char['avg_rfm']}")
            print(f"  Top 3 Kategori:")
            for i, (cat, prop) in enumerate(top_categories[cluster], 1):
                print(f"    {i}. {cat.replace('_', ' ').title()} ({prop})")
            print()
        
        return {
            'rfm_profile': rfm_profile,
            'top_categories': top_categories,
            'category_profile': category_profile,
            'cluster_characteristics': cluster_characteristics
        }
    
    def get_clustered_data(self):
        """Mengembalikan data dengan label klaster"""
        return self.clustered_data
    
    def run_all(self, n_clusters=None):
        """
        Menjalankan seluruh proses klasterisasi
        """
        self.normalize_data()
        elbow_data = self.determine_optimal_k()
        self.fit_kmeans(n_clusters)
        profiles = self.get_cluster_profiles()
        
        print("="*70)
        print("KLASTERISASI SELESAI")
        print("="*70 + "\n")
        
        return {
            'clustered_data': self.clustered_data,
            'elbow_data': elbow_data,
            'profiles': profiles
        }