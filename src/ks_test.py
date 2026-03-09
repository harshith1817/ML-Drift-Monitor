from scipy.stats import ks_2samp

def ks_drift_test(old_col, new_col, threshold=0.05):
    
    statistic, p_value=ks_2samp(old_col, new_col)
    return{
        "ks_statistic": float(statistic),
        "p_value": round(float(p_value),20),
        "drift_detected": bool(p_value<threshold)
    }