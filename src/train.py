import pandas as pd
import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import preprocess_data
from mlflow.models import infer_signature
import json, os
import os

mlflow.set_experiment("churn_prediction")

def main():
    df = pd.read_csv("/home/qureshi/code/churn_pipeline_starter/data/raw/telco_churn.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Inside train.py (after train_test_split)
    feature_names = X_train.columns.tolist()

    # Save feature names to file
    os.makedirs("models", exist_ok=True)
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    with mlflow.start_run():
        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("accuracy", acc)
        signature = infer_signature(X_test, preds)
        input_example = X_test.iloc[:1]

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )


    os.makedirs("models", exist_ok=True)  
    joblib.dump(model, "models/model.pkl")

if __name__ == "__main__":
    main()
