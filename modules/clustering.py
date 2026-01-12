from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def normalize_lrfm(lrfm_df):
    scaler = MinMaxScaler()
    lrfm_scaled = lrfm_df.copy()
    lrfm_scaled[["Length", "Recency", "Frequency", "Monetary"]] = scaler.fit_transform(
        lrfm_df[["Length", "Recency", "Frequency", "Monetary"]]
    )
    return lrfm_scaled

def elbow_method(lrfm_scaled):
    X = lrfm_scaled[["Length", "Recency", "Frequency", "Monetary"]]
    distortions = []
    K = range(2, 11)

    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        distortions.append(km.inertia_)

    return K, distortions

def run_kmeans(lrfm_scaled, k):
    X = lrfm_scaled[["Length", "Recency", "Frequency", "Monetary"]]
    kmeans = KMeans(n_clusters=k, random_state=42)
    lrfm_scaled["Cluster"] = kmeans.fit_predict(X)
    return lrfm_scaled
