"""
Train and compare Logistic Regression, Random Forest and Gradient Boosting
on the churn dataset, then save the best full pipeline with Joblib.

Run from the project root:   python training/train_model.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customer_data.csv"
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"
INFO_PATH = BASE_DIR / "model" / "model_info.json"
RANDOM_STATE = 42


# ---------- 1. Helpers that detect columns automatically ----------
def find_target_column(df):
    """Find the churn column: a binary column whose name looks like churn."""
    keywords = ["churn", "exited", "attrition", "left", "cancel"]
    for col in df.columns:
        if df[col].nunique() == 2 and any(k in col.lower() for k in keywords):
            return col
    # fallback: last binary column
    binary_cols = [c for c in df.columns if df[c].nunique() == 2]
    if binary_cols:
        return binary_cols[-1]
    raise ValueError("Could not find a binary churn column.")


def find_id_columns(df, target):
    """ID-like columns (row numbers, customer ids, names) carry no signal."""
    ids = []
    for col in df.columns:
        if col == target:
            continue
        name = col.lower()
        unique_ratio = df[col].nunique() / len(df)
        is_text = not pd.api.types.is_numeric_dtype(df[col])
        if "rownumber" in name or name.endswith("id") or name == "id":
            ids.append(col)
        elif unique_ratio > 0.95 and not pd.api.types.is_float_dtype(df[col]):
            ids.append(col)  # (float columns like salary are real features)
        elif is_text and df[col].nunique() > 50:  # e.g. Surname
            ids.append(col)
    return ids


def clean_target(series):
    """Turn the target into 0/1 (handles Yes/No, True/False, 1/0)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0,
               "churn": 1, "not churn": 0}
    return series.astype(str).str.strip().str.lower().map(mapping)


# ---------- 2. Load and inspect data ----------
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

target = find_target_column(df)
id_cols = find_id_columns(df, target)
print("Target column  :", target)
print("Dropped columns:", id_cols)

df = df.drop_duplicates()
df[target] = clean_target(df[target])
df = df.dropna(subset=[target])

X = df.drop(columns=[target] + id_cols)
y = df[target].astype(int)

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]
print("Numeric features    :", numeric_cols)
print("Categorical features:", categorical_cols)
print("Missing values      :", int(X.isnull().sum().sum()))

class_share = y.value_counts(normalize=True).round(4).to_dict()
print("Class distribution  :", class_share)
imbalanced = y.value_counts(normalize=True).min() < 0.4
print("Class imbalance handled with balanced weights:", imbalanced)

# ---------- 3. Preprocessing ----------
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])

# ---------- 4. Models ----------
class_weight = "balanced" if imbalanced else None
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight=class_weight, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight=class_weight,
        random_state=RANDOM_STATE, n_jobs=-1),
    # Gradient Boosting has no class_weight option, so we pass sample weights
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
sample_weight = compute_sample_weight("balanced", y_train) if imbalanced else None

# ---------- 5. Train and evaluate ----------
results = {}
fitted = {}
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    if name == "Gradient Boosting" and sample_weight is not None:
        pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
    else:
        pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    fitted[name] = pipe

print("\n=== Model comparison (test set) ===")
table = pd.DataFrame(results).T.drop(columns="confusion_matrix").astype(float).round(4)
print(table.to_string())
for name, r in results.items():
    print(f"\nConfusion matrix - {name}  [[TN, FP], [FN, TP]]")
    print(np.array(r["confusion_matrix"]))

# ---------- 6. Select best model (highest ROC-AUC) and save ----------
best_name = max(results, key=lambda n: results[n]["roc_auc"])
print(f"\nBest model: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.4f})")
joblib.dump(fitted[best_name], MODEL_PATH)

# ---------- 7. Save feature info so the web form matches the dataset ----------
features = []
for col in X.columns:
    if col in categorical_cols:
        features.append({"name": col, "type": "category",
                         "options": sorted(X[col].dropna().unique().tolist())})
    elif set(X[col].dropna().unique()) <= {0, 1}:
        features.append({"name": col, "type": "binary"})
    else:
        is_int = bool((X[col].dropna() % 1 == 0).all())
        features.append({"name": col, "type": "number", "integer": is_int,
                         "min": float(X[col].min()), "max": float(X[col].max()),
                         "default": float(X[col].median())})

info = {"target": target, "best_model": best_name,
        "features": features, "results": results,
        "class_distribution": {str(k): v for k, v in class_share.items()}}
INFO_PATH.write_text(json.dumps(info, indent=2))
print("Saved:", MODEL_PATH)
print("Saved:", INFO_PATH)
