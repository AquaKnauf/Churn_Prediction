### 🧠 Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using a Random Forest classifier, FastAPI for serving predictions, MLflow for tracking experiments, and Streamlit for interactive exploration.

#🚀 Live Demo & API
	•	📊 Streamlit UI: Hugging Face Spaces
	•	🔌 REST API: Hosted on Render


#📦 Features
	•	🔍 Preprocessing: Categorical encoding, numeric conversion, null handling
	•	🌲 Modeling: RandomForestClassifier with MLflow logging
	•	🌐 API: FastAPI endpoint serving real-time predictions
	•	🎛 Streamlit App: User-friendly frontend for testing predictions
	•	📦 Dockerized: Reproducible deployments via Docker
	•	🔁 ML Lifecycle: Uses mlflow to log metrics, parameters, and model versions


#🧪 Quick Start

#1. Clone the Repo

'''bash
git clone https://github.com/AquaKnauf/Churn_Prediction.git
cd Churn_Prediction
'''
#2. Install Requirements

'''bash
pip install -r requirements.txt
'''

#3. Train the Model

'''bash
python train.py
'''

This:
	•	Preprocesses Telco churn data
	•	Trains a Random Forest model
	•	Logs experiment to MLflow
	•	Saves the model and feature list to /models

#4. Run the FastAPI Server

'''bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
'''

#5. Test the API

'''bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.5,
  "TotalCharges": 375.3
}'
'''

#6. Launch Streamlit UI (Locally)
'''bash
streamlit run streamlit_app.py
'''

#🐳 Docker Deployment

To build and run with Docker:
'''bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
'''

#📁 Project Structure
'''
Churn_Prediction/
├── app/
│   └── main.py                # FastAPI backend
├── models/
│   ├── model.pkl              # Trained model
│   └── feature_names.json     # Feature list
├── data/
│   └── raw/telco_churn.csv    # Source dataset
├── preprocess.py              # Data transformation
├── train.py                   # Model training + MLflow logging
├── streamlit_app.py           # Streamlit UI
├── requirements.txt
└── Dockerfile
'''

#🔧 CI/CD Tips

To enable CI/CD:
	•	GitHub Actions: Automate model training, testing, and Docker builds.
	•	Render: Connect this repo, deploy Dockerfile or a FastAPI app from main.py
	•	MLflow Tracking Server (optional): Set up remote tracking for shared experiment logs


#📊 Dataset

Uses the Telco Customer Churn dataset, which includes customer account info, service usage, and churn labels.


#📜 License

MIT License. See LICENSE.





