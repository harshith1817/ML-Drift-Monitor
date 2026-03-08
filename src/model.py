import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)

def train_model(inputs,target):
    inputs_train, inputs_test, target_train, target_test=train_test_split(inputs, target, test_size=0.2, random_state=42)
    model=RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(inputs_train, target_train)
    accuracy=model.score(inputs_test, target_test)
    print("Model trained")
    print("Baseline Accuracy:", accuracy)
    joblib.dump(model, "models/model.pkl")
    return model, inputs_train, accuracy