from src.preprocessing import dynamic_preprocessor
from src.model import train_model
from src.utils import save_baseline_stats
from src.drift_detector import detect_drift

def main():
    x_old,y_old=dynamic_preprocessor("data/teleco_churn.csv","Churn")
    print("Training data preprocesing is done")
    x_new,y_new=dynamic_preprocessor("data/teleco_churn_new.csv","Churn")
    print("New data preprocesing is done")
    model, train_inputs, acc=train_model(x_old,y_old)
    save_baseline_stats(train_inputs)
    detect_drift(x_old, x_new, "reports/baseline_stats.json")

if __name__=="__main__":
    main()