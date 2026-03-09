from src.preprocessing import dynamic_preprocessor
from src.model import train_model
from src.utils import save_baseline_stats
from src.data_drift_report_generator import generate_data_drift_report
from src.data_drift_detection import detect_data_drift
from src.prediction_drift_detection import detect_prediction_drift
from src.concept_drift_detection import detect_concept_drift

def main():
    old_df="data/adult_income.csv"
    new_df="data/adult_income_new.csv"
    target_col="income"
    x_old,y_old=dynamic_preprocessor(old_df,target_col)
    print("Old data preprocesing is done.")
    x_new,y_new=dynamic_preprocessor(new_df,target_col)
    print("\nNew data preprocesing is done.")
    model, train_inputs, acc=train_model(x_old,y_old)
    save_baseline_stats(train_inputs)
    generate_data_drift_report(x_old, x_new, "reports/baseline_stats.json")
    detect_data_drift("reports/data_drift.json")
    detect_prediction_drift(x_old, x_new, "models/model.pkl")
    detect_concept_drift(x_new, y_new, "models/model.pkl", "reports/baseline_metrics.json")

if __name__=="__main__":
    main()