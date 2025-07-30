import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import json

# Load dataset
data = pd.read_csv("Train.csv")

# Handle missing values
data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)
data['Item_Visibility'].replace(0, np.nan, inplace=True)
data['Item_Visibility'].fillna(data['Item_Visibility'].mean(), inplace=True)

# Initialize label encoders dict
label_encoders = {}

# Encode categorical features and save encoders
for col in ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features and target
X = data.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'], axis=1)
y = data['Item_Outlet_Sales']

# Save feature columns order
feature_columns = list(X.columns)
with open("feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and label encoders
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Model, encoders and feature columns saved successfully!")
