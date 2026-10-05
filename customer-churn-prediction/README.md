# Customer Churn Prediction

Predicts whether a bank customer will churn, using `data/customer_data.csv`
(10,000 rows). The target column (`Exited`) and ID-like columns
(`RowNumber`, `CustomerId`, `Surname`) are detected automatically.

## Setup (VS Code terminal)
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Train
```bash
python training/train_model.py
```
Compares Logistic Regression, Random Forest and Gradient Boosting
(accuracy, precision, recall, F1, ROC-AUC, confusion matrix), picks the
best by ROC-AUC, and saves the full pipeline to `model/churn_model.pkl`.
Missing values are imputed; class imbalance (about 20% churn) is handled with balanced weights.

## Run the app
```bash
python app.py
```
Open http://127.0.0.1:5000

## API
`POST /predict` with JSON of the feature values returns
`{"prediction": "CHURN" | "NOT CHURN", "churn_probability": 56.9}`
