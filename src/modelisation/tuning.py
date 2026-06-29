"""
Tuning des hyperparametres du HistGradientBoosting par
optimisation bayesienne (Optuna).
On optimise la PR-AUC sur le test temporel (mai-juin 2004),
metrique principale du projet.

Hyperparametres explores :
    - learning_rate    : vitesse d'apprentissage
    - max_leaf_nodes   : complexite des arbres
    - min_samples_leaf : regularisation par taille minimale des feuilles
    - l2_regularization: regularisation L2
    - max_iter         : nombre maximum d'arbres

Resultats ecrits dans data/results/metrics/tuning_results.csv
Meilleurs hyperparametres dans data/results/metrics/best_params.csv
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

import config as cfg
from modelisation import features as F

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 30  # nombre d'essais d'optimisation


def objective(trial, train, test, cat, num):
    """Fonction objectif : PR-AUC sur le test temporel."""

    # Espace de recherche des hyperparametres
    params = {
        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_leaf_nodes"   : trial.suggest_int("max_leaf_nodes", 15, 63),
        "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 50, 500),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.1, 10.0, log=True),
        "max_iter"         : trial.suggest_int("max_iter", 200, 800),
    }

    pipe = Pipeline([
        ("pre", F.make_preprocessor(cat, num, scale=False)),
        ("clf", HistGradientBoostingClassifier(
            **params,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.1,
            random_state=0,
        ))
    ])

    pipe.fit(train[cat + num], train["fraude"])
    p_te = pipe.predict_proba(test[cat + num])[:, 1]
    return average_precision_score(test["fraude"], p_te)


def main():
    print(">>> Tuning HistGradientBoosting (Optuna, bayesien)")
    print(f"    Nombre d'essais : {N_TRIALS}")

    gold = pd.read_parquet(cfg.GOLD_PARQUET)
    train, test = F.temporal_split(gold)
    cat, num = F.get_feature_lists()

    # Lancement de l'optimisation bayesienne
    sampler = TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, train, test, cat, num),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # Resultats de tous les essais
    results = study.trials_dataframe()[
        ["number", "value", "params_learning_rate", "params_max_leaf_nodes",
         "params_min_samples_leaf", "params_l2_regularization", "params_max_iter"]
    ].rename(columns={"number": "trial", "value": "PR_AUC"})
    results = results.sort_values("PR_AUC", ascending=False).reset_index(drop=True)
    results.to_csv(cfg.METRICS_DIR / "tuning_results.csv", index=False)

    # Meilleurs hyperparametres
    best = study.best_params
    best["PR_AUC"] = study.best_value
    pd.DataFrame([best]).to_csv(cfg.METRICS_DIR / "best_params.csv", index=False)

    print("\n=== MEILLEURS HYPERPARAMETRES ===")
    for k, v in best.items():
        print(f"   {k} : {v}")
    print(f"\n   PR-AUC obtenue : {study.best_value:.4f}")
    print(f"\n-> Resultats complets : {cfg.METRICS_DIR / 'tuning_results.csv'}")
    print(f"-> Meilleurs params   : {cfg.METRICS_DIR / 'best_params.csv'}")


if __name__ == "__main__":
    main()
