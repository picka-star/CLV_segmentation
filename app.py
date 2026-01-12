import streamlit as st
import pandas as pd
from modules.preprocessing import preprocess_data
from modules.lrfm import compute_lrfm
from modules.ahp import get_ahp_weights
from modules.clustering import normalize_lrfm, elbow_method, run_kmeans
from modules.clv import compute_clv
from modules.visualization import plot_elbow, plot_clv_per_cluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="CLV Segmentation Platform", layout="wide", page_icon="📊")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 📊 Customer Segmentation Platform")
    #t.markdown("LRFM + AHP + K-Means + CLV")
    #st.divider()
    uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx", "csv"])
    #st.divider()
    st.markdown("### AHP Weights")
    wL = st.slider("Length", 0.0, 1.0, 0.25)
    wR = st.slider("Recency", 0.0, 1.0, 0.25)
    wF = st.slider("Frequency", 0.0, 1.0, 0.25)
    wM = st.slider("Monetary", 0.0, 1.0, 0.25)
    weights = get_ahp_weights(wL, wR, wF, wM)
    st.caption("Total weight will be normalized automatically")

# =========================
# HEADER
# =========================
st.markdown("# Customer Segmentation Dashboard")
#st.markdown("**LRFM + AHP + K-Means + CLV**")
#st.divider()

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Data Overview",
        "📐 LRFM Analysis",
        "🧮 Clustering",
        "💰 CLV & Insights"
    ])

    # =========================
    # TAB 1 - DATA OVERVIEW
    # =========================
    with tab1:
        st.subheader("Dataset Preview")
        colA, colB = st.columns([2,1])
        with colA:
            st.dataframe(df.head(50), use_container_width=True)
        with colB:
            st.markdown("### Dataset Info")
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            st.write("Columns:")
            st.write(list(df.columns))

    # =========================
    # PREPROCESS
    # =========================
    df_clean = preprocess_data(df)
    lrfm = compute_lrfm(df_clean)
    lrfm_scaled = normalize_lrfm(lrfm)

    # =========================
    # TAB 2 - LRFM
    # =========================
    with tab2:
        st.subheader("LRFM Table")
        st.dataframe(lrfm.head(50), use_container_width=True)

        st.markdown("### Statistical Summary")
        st.dataframe(lrfm[["Length","Recency","Frequency","Monetary"]].describe(), use_container_width=True)

    # =========================
    # TAB 3 - CLUSTERING
    # =========================
    with tab3:
        st.subheader("Elbow Method")
        K, distortions = elbow_method(lrfm_scaled)
        plot_elbow(K, distortions)

        k_selected = st.slider("Select number of clusters (k)", 2, 10, 3)

        lrfm_clustered = run_kmeans(lrfm_scaled, k_selected)

        st.subheader("Clustered Data")
        st.dataframe(lrfm_clustered.head(50), use_container_width=True)

        # Validation metrics
        X = lrfm_scaled[["Length","Recency","Frequency","Monetary"]]
        sil = silhouette_score(X, lrfm_clustered["Cluster"])
        db = davies_bouldin_score(X, lrfm_clustered["Cluster"])
        ch = calinski_harabasz_score(X, lrfm_clustered["Cluster"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette", f"{sil:.3f}")
        col2.metric("Davies-Bouldin", f"{db:.3f}")
        col3.metric("Calinski-Harabasz", f"{ch:.1f}")

    # =========================
    # TAB 4 - CLV & INSIGHTS
    # =========================
    with tab4:
        lrfm_clv = compute_clv(lrfm_clustered, weights)
        final_df = pd.merge(lrfm, lrfm_clv[["CustomerID","Cluster","CLV"]], on="CustomerID")

        st.subheader("CLV Result")
        st.dataframe(final_df.head(50), use_container_width=True)

        st.subheader("Average CLV per Cluster")
        plot_clv_per_cluster(final_df)

        st.subheader("Download Results")
        csv = final_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CLV Result (CSV)",
            data=csv,
            file_name="hasil_clv_segmentation.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a dataset from the sidebar to start analysis.")