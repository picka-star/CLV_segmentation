import matplotlib.pyplot as plt
import streamlit as st

def plot_elbow(K, distortions):
    fig, ax = plt.subplots()
    ax.plot(K, distortions, marker='o')
    ax.set_xlabel("Jumlah Cluster (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

def plot_clv_per_cluster(df):
    clv_cluster = df.groupby("Cluster")["CLV"].mean().reset_index()

    fig, ax = plt.subplots()
    ax.bar(clv_cluster["Cluster"].astype(str), clv_cluster["CLV"])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Rata-rata CLV")
    ax.set_title("Rata-rata CLV per Cluster")
    st.pyplot(fig)
