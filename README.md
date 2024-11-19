# Loan Approval and Fraud Detection: Anomaly Detection and Classification
This project aims to improve the efficiency and security of loan processing systems by leveraging machine learning. It addresses two key objectives:
1. **Fraud Detection**: Identify potentially fraudulent applications using unsupervised learning (Isolation Forest).
2. **Loan Approval Prediction**: Predict loan approval outcomes using supervised learning (XGBoost).

The project includes:
- Data preprocessing and exploratory analysis.
- Building, evaluating, and comparing machine learning models.
- Deploying the models via a Flask-based API for real-time predictions.
- **Unsupervised Learning**:
  - Identified anomalies in loan applications using Isolation Forest.
  - Highlighted features contributing to anomalous behavior (e.g., high loan-to-income ratio, low credit scores).
  
- **Supervised Learning**:
  - Built an XGBoost model to predict loan approval with high accuracy and AUC-ROC.
  - Evaluated using precision, recall, F1-score, and feature importance.

- **Deployment**:
  - Developed a RESTful API using Flask to serve predictions.
  - Users can input loan details and receive predictions in real-time.

- **Scalability**:
  - The solution is ready for integration into loan management systems.
