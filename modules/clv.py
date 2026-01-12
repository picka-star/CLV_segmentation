def compute_clv(lrfm_scaled, weights):
    wL, wR, wF, wM = weights

    lrfm_scaled["CLV"] = (
        wL * lrfm_scaled["Length"] +
        wR * (1 / (1 + lrfm_scaled["Recency"])) +
        wF * lrfm_scaled["Frequency"] +
        wM * lrfm_scaled["Monetary"]
    )

    return lrfm_scaled
