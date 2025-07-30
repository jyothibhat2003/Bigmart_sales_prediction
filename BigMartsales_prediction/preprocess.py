import pandas as pd
import pickle
import json

# Load label encoders and feature columns globally once
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

def preprocess_input(input_data):
    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data])

    # Encode categorical columns with saved label encoders
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Reorder columns exactly as in training
    df = df[feature_columns]

    return df
