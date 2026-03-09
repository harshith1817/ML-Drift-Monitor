import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def dynamic_preprocessor(path, target_col):
    churn_df=pd.read_csv(path)
    churn_df = churn_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    for col in churn_df.columns:
        if churn_df[col].dtype == "object":
            converted = pd.to_numeric(churn_df[col], errors="coerce")
    
            # convert if most values are numeric
            if converted.notna().sum() / len(churn_df) > 0.8:
                churn_df[col] = converted
        
        missing_ratio = churn_df[col].isnull().mean()
        if missing_ratio < 0.025:
            churn_df = churn_df[churn_df[col].notnull()]
        else:
            if churn_df[col].dtype in ["int64","float64"]:
                churn_df[col] = churn_df[col].fillna(churn_df[col].median())
            else:
                churn_df[col] = churn_df[col].fillna(churn_df[col].mode()[0])
        
        cols_to_drop = []
        if col != target_col:
            unique_ratio = churn_df[col].nunique() / len(churn_df)
            
            if unique_ratio==1:
                cols_to_drop.append(col)
        churn_df = churn_df.drop(columns=cols_to_drop)
    inputs = churn_df.drop(columns=[target_col])
    target = churn_df[target_col]
    numeric_cols=inputs.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols=inputs.select_dtypes(include=["object"]).columns

    binary_cols = [
        col for col in categorical_cols
        if inputs[col].nunique() == 2
    ]
    for col in binary_cols:
        unique_vals = inputs[col].dropna().unique()
        inputs[col] = inputs[col].replace({unique_vals[0]:1, unique_vals[1]:0})
    
    multiple_cols = [
        col for col in categorical_cols
        if inputs[col].nunique() > 2
    ]
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_dfs = []
    for col in multiple_cols:
        encoded = encoder.fit_transform(inputs[[col]])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out([col]),
            index=inputs.index
        )
        encoded_dfs.append(encoded_df)
    
    if len(encoded_dfs)>0:
        encoded_all = pd.concat(encoded_dfs, axis=1)
        inputs = inputs.drop(columns=multiple_cols)
        inputs = pd.concat([inputs, encoded_all], axis=1)
    return inputs, target