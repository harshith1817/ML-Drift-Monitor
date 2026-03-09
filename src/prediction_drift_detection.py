import joblib

from src.ks_test import ks_drift_test
from src.psi_test import calculate_psi

def detect_prediction_drift(x_old, x_new, model_path):
    
    model=joblib.load(model_path)
    
    old_preds=model.predict_proba(x_old)[:,1]
    new_preds=model.predict_proba(x_new)[:,1]
    
    ks_result=ks_drift_test(old_preds, new_preds)
    psi_value=calculate_psi(old_preds, new_preds)
    
    drift_detected=ks_result["p_value"]<0.05 or psi_value>0.2
    
    if(drift_detected):
        print("\nPrediction drift detected.")
    else:
        print("\nNo prediction drift detected.")