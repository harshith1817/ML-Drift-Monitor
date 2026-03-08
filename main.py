from src.preprocessing import dynamic_preprocessor
from src.model import train_model
from src.utils import save_baseline_stats

def main():
    x,y=dynamic_preprocessor("data/adult_income.csv","income")
    model, train_inputs, acc=train_model(x,y)
    save_baseline_stats(train_inputs)

if __name__=="__main__":
    main()