import numpy as np
import pandas as pd

# df=pd.read_csv("data/teleco_churn.csv")
# df_new=df.copy()

# df_new["tenure"]=df_new["tenure"] + np.random.randint(0, 12, size=len(df_new))
# a=np.round(np.random.uniform(0, 21, size=len(df_new)),1)
# df_new["MonthlyCharges"]=np.round(df_new["MonthlyCharges"] + a,2)
# df_new["TotalCharges"] = pd.to_numeric(df_new["TotalCharges"], errors="coerce")
# df_new["TotalCharges"]=np.round(df_new["TotalCharges"] + a*np.round(np.random.uniform(12,14, size=len(df_new)),1),2)

# df_new.to_csv("data/teleco_churn_new.csv", index=False)

# print("Drift dataset created.")





# df=pd.read_csv("data/adult_income.csv")
# df_new=df.copy()

# df_new["age"]=df_new["age"] + np.random.randint(1, 6, size=len(df_new))
# scale = np.random.uniform(0.9, 1.1, size=len(df_new))
# df_new["fnlwgt"] = (df_new["fnlwgt"] * scale).astype(int)
# mask = df_new.sample(frac=0.15).index
# df_new.loc[mask, "capital.gain"] += np.random.randint(0, 2000, size=len(mask))
# mask = df_new.sample(frac=0.1).index
# df_new.loc[mask, "capital.loss"] += np.random.randint(0, 200, size=len(mask))
# df_new["hours.per.week"] = df_new["hours.per.week"] + np.random.randint(-3, 6, size=len(df_new))

# df_new.to_csv("data/adult_income_new.csv", index=False)

# print("Drift dataset created.")




df=pd.read_csv("data/creditcard.csv")
df_new=df.copy()

df_new["Time"] = df_new["Time"] + np.random.randint(1000, 10000, size=len(df_new))

pca_cols = [f"V{i}" for i in range(1,29)]
noise = np.random.normal(0, 0.2, size=(len(df_new), len(pca_cols)))

df_new[pca_cols] = df_new[pca_cols] + noise

scale = np.random.uniform(1.05, 1.3, size=len(df_new))
df_new["Amount"] = np.round(df_new["Amount"] * scale, 2)

df_new.to_csv("data/creditcard_new.csv", index=False)

print("Drift dataset created.")