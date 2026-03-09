import joblib
import json

from sklearn.metrics import accuracy_score

def detect_concept_drift(x_new, y_new, model_path, baseline_metrics_path):
    
    model=joblib.load(model_path)
    
    preds=model.predict(x_new)
    
    new_accuracy=accuracy_score(y_new, preds)
    
    with open(baseline_metrics_path) as f:
        baseline=json.load(f)
    
    baseline_accuracy=baseline["baseline_accuracy"]
    
    drift_detected=new_accuracy < baseline_accuracy*0.9
    
    if drift_detected:
        print("\nConcept drift detected.")
    else:
        print("\nNo concept drift detected.")