import json
from src.ks_test import ks_drift_test
from src.psi_test import calculate_psi

def load_baseline(path):
    with open(path, "r") as f:
        baseline=json.load(f)
    return baseline


def generate_data_drift_report(old_df, new_df, baseline_report_path):
    
    # old_df=pd.read_csv(old_path)
    # new_df=pd.read_csv(new_path)
    
    baseline=load_baseline(baseline_report_path)
    
    report={}
    
    # numeric_cols=old_df.select_dtypes(include=["number"]).columns
    
    for col in old_df.columns:
        if col not in new_df.columns:
            continue
        
        old_values=old_df[col]
        new_values=new_df[col]
        
        ks_result=ks_drift_test(old_values, new_values)
        
        psi_result=calculate_psi(old_values, new_values)
        
        report[col]={
            "baseline_mean": baseline[col]["mean"],
            "production mean": float(new_values.mean()),
            "ks_test": ks_result,
            "psi": psi_result
        }
        
    with open("reports/drift_report.json","w") as f:
        json.dump(report, f, indent=4)
    print("\nData drift report generated.")