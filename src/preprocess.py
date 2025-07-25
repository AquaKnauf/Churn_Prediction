import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.dropna()
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df = df.drop("customerID", axis=1)

    for col in df.select_dtypes(include=['object']).columns:
        if col != "Churn":
            df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})
    return train_test_split(X, y, test_size=0.2, random_state=42)
