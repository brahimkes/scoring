"""
Analyse de stabilite du modele retenu (HistGradientBoosting).
Verifie que le modele generalise bien et ne surappend pas.

Trois analyses :
    1. Comparaison ROC-AUC train vs test (surapprentissage)
    2. Validation croisee temporelle (stabilite sur plusieurs periodes)
    3. Evolution de la PR-AUC par mois de test

Resultats ecrits dans data/results/metrics/stabilite.csv
                    data/results/figures/stabilite_monthly.png
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # -> src/

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

import config as cfg
from modelisation import features as F


def build_model():
    """Modele retenu avec ses hyperparametres."""
    return HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=3.0, min_samples_leaf=300,
        class_weight="balanced",
        early_stopping=True, validation_fraction=0.1, random_state=0,
    )


def analyse_surapprentissage(train, test, cat, num):
    """Compare ROC-AUC train vs test pour detecter le surapprentissage."""
    print(">>> Analyse surapprentissage (train vs test)")

    pipe = Pipeline([
        ("pre", F.make_preprocessor(cat, num, scale=False)),
        ("clf", build_model())
    ])
    pipe.fit(train[cat + num], train["fraude"])

    p_tr = pipe.predict_proba(train[cat + num])[:, 1]
    p_te = pipe.predict_proba(test[cat + num])[:, 1]

    roc_train = roc_auc_score(train["fraude"], p_tr)
    roc_test  = roc_auc_score(test["fraude"],  p_te)
    pr_train  = average_precision_score(train["fraude"], p_tr)
    pr_test   = average_precision_score(test["fraude"],  p_te)
    ecart_roc = roc_train - roc_test

    print(f"   ROC-AUC train : {roc_train:.4f}")
    print(f"   ROC-AUC test  : {roc_test:.4f}")
    print(f"   Ecart ROC     : {ecart_roc:.4f}")
    print(f"   PR-AUC train  : {pr_train:.4f}")
    print(f"   PR-AUC test   : {pr_test:.4f}")

    if ecart_roc < 0.02:
        verdict = "Tres stable (ecart < 0.02)"
    elif ecart_roc < 0.05:
        verdict = "Stable (ecart < 0.05)"
    elif ecart_roc < 0.10:
        verdict = "Surapprentissage modere (ecart < 0.10)"
    else:
        verdict = "Surapprentissage important (ecart >= 0.10)"

    print(f"   Verdict : {verdict}")

    return {
        "ROC_AUC_train": roc_train,
        "ROC_AUC_test" : roc_test,
        "ecart_ROC"    : ecart_roc,
        "PR_AUC_train" : pr_train,
        "PR_AUC_test"  : pr_test,
        "verdict"      : verdict,
    }


def analyse_mensuelle(gold, cat, num):
    """
    Evalue la PR-AUC mois par mois sur la periode de test.
    On entraine toujours sur le passe (avant le mois evalue).
    """
    print("\n>>> Analyse mensuelle de stabilite")

    gold["mois"] = gold["datetime"].dt.to_period("M")
    mois_test = sorted(gold[gold["datetime"] >= cfg.CUTOFF]["mois"].unique())

    rows = []
    for mois in mois_test:
        train_m = gold[gold["mois"] < mois].copy()
        test_m  = gold[gold["mois"] == mois].copy()

        if test_m["fraude"].sum() == 0:
            continue

        pipe = Pipeline([
            ("pre", F.make_preprocessor(cat, num, scale=False)),
            ("clf", build_model())
        ])
        pipe.fit(train_m[cat + num], train_m["fraude"])
        p_te = pipe.predict_proba(test_m[cat + num])[:, 1]
        pr   = average_precision_score(test_m["fraude"], p_te)

        print(f"   {mois} : PR-AUC = {pr:.4f} "
              f"({int(test_m['fraude'].sum())} fraudes)")
        rows.append({"mois": str(mois), "PR_AUC": pr,
                     "nb_fraudes": int(test_m["fraude"].sum()),
                     "nb_transactions": len(test_m)})

    # Graphique mensuel
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    plt.plot(df["mois"], df["PR_AUC"], marker="o", color="#028090")
    plt.axhline(df["PR_AUC"].mean(), ls="--", color="grey",
                label=f"Moyenne : {df['PR_AUC'].mean():.3f}")
    plt.xlabel("Mois"); plt.ylabel("PR-AUC")
    plt.title("Stabilite mensuelle — HistGradientBoosting")
    plt.legend(); plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "stabilite_monthly.png", dpi=120)
    plt.close()
    print(f"\n   Graphique -> {cfg.FIG_DIR / 'stabilite_monthly.png'}")

    return df


def main():
    print(">>> Analyse de stabilite du modele retenu")
    gold = pd.read_parquet(cfg.GOLD_PARQUET)
    train, test = F.temporal_split(gold)
    cat, num = F.get_feature_lists()

    # 1. Surapprentissage
    row_sur = analyse_surapprentissage(train, test, cat, num)

    # 2. Stabilite mensuelle
    df_mois = analyse_mensuelle(gold, cat, num)

    # Sauvegarde
    pd.DataFrame([row_sur]).to_csv(
        cfg.METRICS_DIR / "stabilite_surapprentissage.csv", index=False)
    df_mois.to_csv(
        cfg.METRICS_DIR / "stabilite_mensuelle.csv", index=False)

    print(f"\n-> Resultats : {cfg.METRICS_DIR}")


if __name__ == "__main__":
    main()
