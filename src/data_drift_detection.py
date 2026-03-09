import json

def detect_data_drift(report_path, ks_threshold=0.05, psi_threshold=0.2):
    
    with open(report_path, "r") as f:
        report=json.load(f)
    
    drift_features=[]
    
    for feature, stats in report.items():
        ks_p_value=stats["ks_test"]["p_value"]
        psi_value=stats["psi"]
        
        if(ks_p_value<ks_threshold or psi_value>psi_threshold):
            drift_features.append(feature)
        
    if len(drift_features)==0:
        print("\nNo data drift detected.")
    else:
        print("\nData drift is detected in the following features:")
        for i in drift_features:
            print("                                                 ",i)
    return drift_features