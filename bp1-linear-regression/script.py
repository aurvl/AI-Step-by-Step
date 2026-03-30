"""
Régression linéaire sur dataset_vols.csv

Usage:
    python script.py --data dataset_vols.csv --strategy proper
    python script.py --data dataset_vols.csv --strategy minimal
    python script.py --data dataset_vols.csv --strategy raw
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


TARGET_COL = "retard_arrivee_min"

CATEGORICAL_COLS = [
    "compagnie",
    "aeroport_depart",
    "aeroport_arrivee",
    "type_avion",
    "jour_semaine",
]

BINARY_COLS = [
    "vol_international",
    "hub_depart",
    "greve_sol",
    "changement_equipage",
]

NUMERIC_COLS = [
    "retard_depart_min",
    "meteo_score",
    "heure_depart",
    "nb_escales",
    "charge_reseau",
    "distance_km",
    "vent_kmh",
    "precipitation_mm",
    "temperature_c",
    "congestion_aeroport",
    "experience_captain_ans",
    "maintenance_score",
    "ponctualite_route_hist",
    "age_avion_ans",
    "rotations_24h",
    "passagers",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Régression linéaire sur dataset_vols.csv")
    parser.add_argument("--data", type=str, default="dataset_vols.csv", help="Chemin vers le CSV")
    parser.add_argument(
        "--strategy",
        type=str,
        default="proper",
        choices=["proper", "minimal", "raw"],
        help="Stratégie de préparation des données",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Part du test set")
    parser.add_argument("--random-state", type=int, default=42, help="Seed")
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"La cible '{TARGET_COL}' est absente du dataset.")
    return df


def build_experiment(
    df: pd.DataFrame,
    strategy: str = "proper",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    data = df.copy()
    all_features = [c for c in data.columns if c != TARGET_COL]
    decisions: List[str] = []

    if strategy == "proper":
        data = data.drop_duplicates().copy()
        decisions.extend([
            "Suppression des doublons",
            "Imputation des valeurs manquantes",
            "One-hot encoding des variables catégorielles",
            "Normalisation des variables numériques",
        ])
        selected_features = all_features

        X = data[selected_features]
        y = data[TARGET_COL]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    NUMERIC_COLS,
                ),
                (
                    "cat",
                    Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    CATEGORICAL_COLS,
                ),
                ("bin", "passthrough", BINARY_COLS),
            ]
        )

    elif strategy == "minimal":
        data = data.drop_duplicates().dropna().copy()
        decisions.extend([
            "Suppression des doublons",
            "Suppression des lignes contenant des NaN",
            "One-hot encoding des variables catégorielles",
            "Pas de normalisation",
        ])
        selected_features = all_features

        X = data[selected_features]
        y = data[TARGET_COL]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", NUMERIC_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
                ("bin", "passthrough", BINARY_COLS),
            ]
        )

    elif strategy == "raw":
        decisions.extend([
            "Conservation des doublons",
            "Imputation grossière des NaN (0 / 'Unknown')",
            "Encodage ordinal des catégories",
            "Suppression volontaire de plusieurs variables très informatives",
        ])

        dropped_features = [
            "retard_depart_min",
            "charge_reseau",
            "congestion_aeroport",
            "ponctualite_route_hist",
            "maintenance_score",
            "vent_kmh",
            "precipitation_mm",
            "temperature_c",
        ]

        selected_features = [c for c in all_features if c not in dropped_features]
        cat_raw = [c for c in CATEGORICAL_COLS if c in selected_features]
        bin_raw = [c for c in BINARY_COLS if c in selected_features]
        num_raw = [c for c in NUMERIC_COLS if c in selected_features]

        X = data[selected_features]
        y = data[TARGET_COL]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ]),
                    num_raw,
                ),
                (
                    "cat",
                    Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]),
                    cat_raw,
                ),
                ("bin", "passthrough", bin_raw),
            ]
        )

    else:
        raise ValueError("strategy doit être 'proper', 'minimal' ou 'raw'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("reg", Ridge(alpha=1.0)),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    return {
        "strategy": strategy,
        "n_rows_used": len(X),
        "n_features_used": len(selected_features),
        "decisions": decisions,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "model": model,
        "selected_features": selected_features,
    }


def print_results(res: Dict) -> None:
    labels = {
        "proper": "contexte riche et propre",
        "minimal": "nettoyage minimal",
        "raw": "contexte pauvre",
    }

    print(f"\n=== Approche : {labels[res['strategy']]} ===")
    print("Décisions de préparation :")
    for d in res["decisions"]:
        print(f"  - {d}")

    print(f"Lignes utilisées     : {res['n_rows_used']:,}")
    print(f"Variables utilisées  : {res['n_features_used']}")
    print(f"MAE                  : {res['mae']:.3f}")
    print(f"RMSE                 : {res['rmse']:.3f}")
    print(f"R²                   : {res['r2']:.4f}")


def main() -> None:
    args = parse_args()
    df = load_data(args.data)
    res = build_experiment(
        df=df,
        strategy=args.strategy,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print_results(res)


if __name__ == "__main__":
    main()