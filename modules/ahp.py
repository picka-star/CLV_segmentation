def get_ahp_weights(wL, wR, wF, wM):
    total = wL + wR + wF + wM
    if total == 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [wL/total, wR/total, wF/total, wM/total]
