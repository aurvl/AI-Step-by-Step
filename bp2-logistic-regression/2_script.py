import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# 0) OUTILS
# ============================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def log_loss(y_true, y_proba, eps=1e-12):
    y_proba = np.clip(y_proba, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba)
    )

def standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

def standardize_transform(X, mean, std):
    return (X - mean) / std

def one_hot_fit_transform(train_df, test_df, categorical_cols):
    train_enc = pd.get_dummies(train_df, columns=categorical_cols, drop_first=False)
    test_enc = pd.get_dummies(test_df, columns=categorical_cols, drop_first=False)
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)
    return train_enc, test_enc

# ============================================================
# 1) NEURONE ARTIFICIEL BINAIRE
# ============================================================

class ArtificialNeuron:
    def __init__(self, learning_rate=0.05, epochs=3000, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y, verbose=False):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        self.w = rng.normal(0, 0.1, size=n_features)
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            z = X @ self.w + self.b
            y_proba = sigmoid(z)

            loss = log_loss(y, y_proba)
            self.loss_history.append(loss)

            # gradients
            dz = y_proba - y
            dw = (X.T @ dz) / n_samples
            db = np.mean(dz)

            # update
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch:04d} | Log-loss = {loss:.4f}")

        return self

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# ============================================================
# 2) IMPORTER LE DATASET
# ============================================================

df = pd.read_csv("dataset_streaming_churn.csv")

print("\nDataset chargé :")
print(df.head())
print("\nTaux de churn :", round(df["target"].mean(), 4))

# ============================================================
# 3) FEATURE ENGINEERING
# ============================================================

df["engagement_score"] = (
    0.4 * df["watch_hours_week"]
    + 0.4 * df["sessions_week"]
    + 20 * df["content_completion_rate"]
)

df["friction_score"] = (
    1.5 * df["support_tickets_90d"]
    + 2.5 * df["payment_failures_6m"]
    + 0.08 * df["days_since_last_login"]
)

df["price_per_hour"] = df["monthly_price"] / (df["watch_hours_week"] + 1)

# ============================================================
# 4) TRAIN / TEST SPLIT
# ============================================================

X = df.drop(columns=["target"])
y = df["target"].astype(int).to_numpy()

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTrain shape :", X_train_df.shape)
print("Test shape  :", X_test_df.shape)

print("\nRépartition de la cible (train) :")
vals, counts = np.unique(y_train, return_counts=True)
print(dict(zip(vals, counts / counts.sum())))

# ============================================================
# 5) PREPROCESSING
# ============================================================

numeric_cols = X_train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X_train_df.select_dtypes(include=["object"]).columns.tolist()

# imputation simple
for col in numeric_cols:
    med = X_train_df[col].median()
    X_train_df[col] = X_train_df[col].fillna(med)
    X_test_df[col] = X_test_df[col].fillna(med)

for col in categorical_cols:
    mode = X_train_df[col].mode()[0]
    X_train_df[col] = X_train_df[col].fillna(mode)
    X_test_df[col] = X_test_df[col].fillna(mode)

# one-hot encoding
X_train_df, X_test_df = one_hot_fit_transform(X_train_df, X_test_df, categorical_cols)

# standardisation numérique
mean, std = standardize_fit(X_train_df.to_numpy(dtype=float))
X_train = standardize_transform(X_train_df.to_numpy(dtype=float), mean, std)
X_test = standardize_transform(X_test_df.to_numpy(dtype=float), mean, std)

# ============================================================
# 6) CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model_cv = ArtificialNeuron(learning_rate=0.05, epochs=3000, random_state=42)
    model_cv.fit(X_tr, y_tr, verbose=False)

    y_val_pred = model_cv.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    cv_scores.append(acc)

print("\n==============================")
print("CROSS-VALIDATION (train set)")
print("==============================")
print("Scores par fold :", np.round(cv_scores, 4))
print(f"Accuracy moyenne : {np.mean(cv_scores):.4f}")
print(f"Ecart-type       : {np.std(cv_scores):.4f}")

# ============================================================
# 7) ENTRAINEMENT FINAL
# ============================================================

model = ArtificialNeuron(learning_rate=0.05, epochs=3000, random_state=42)
model.fit(X_train, y_train, verbose=True)

# ============================================================
# 8) PREDICTIONS + METRICS
# ============================================================

y_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

print("\n==============================")
print("EVALUATION NEURONE ARTIFICIEL")
print("==============================")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification report :")
print(classification_report(y_test, y_pred))
print("Confusion matrix :")
print(confusion_matrix(y_test, y_pred))