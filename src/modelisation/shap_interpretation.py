"""
Interpretation du modele retenu (HistGradientBoosting) par SHAP.

Deux niveaux d'interpretation :
    1. Global  : quelles variables comptent le plus en moyenne ?
                 -> shap_importance.csv + shap_beeswarm.png
    2. Local   : pourquoi cette transaction precise est-elle
                 classee frauduleuse ?
                 -> shap_local_top_fraude.png

On travaille sur un echantillon du test (rapidite).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

import config as cfg
from modelisation import features as F

SAMPLE_SIZE = 5000


def build_model():
    return HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=3.0, min_samples_leaf=300,
        class_weight="balanced",
        early_stopping=True, validation_fraction=0.1, random_state=0,
    )


def main():
    print(">>> Interpretation SHAP — HistGradientBoosting")

    gold = pd.read_parquet(cfg.GOLD_PARQUET)
    train, test = F.temporal_split(gold)
    cat, num = F.get_feature_lists()

    print("   Entrainement du modele...")
    pipe = Pipeline([
        ("pre", F.make_preprocessor(cat, num, scale=False)),
        ("clf", build_model())
    ])
    pipe.fit(train[cat + num], train["fraude"])

    sample = test.sample(min(SAMPLE_SIZE, len(test)), random_state=0)
    X_sample = sample[cat + num]
    X_transformed = pipe["pre"].transform(X_sample)
    feature_names = cat + num

    print("   Calcul des valeurs SHAP (explainer arbre)...")
    explainer = shap.TreeExplainer(pipe["clf"])
    shap_values = explainer.shap_values(X_transformed)

    # 1. IMPORTANCE GLOBALE
    print("   Generation graphique importance globale...")
    importance = pd.DataFrame({
        "variable": feature_names,
        "SHAP_mean_abs": np.abs(shap_values).mean(axis=0)
    }).sort_values("SHAP_mean_abs", ascending=False).reset_index(drop=True)

    importance.to_csv(cfg.METRICS_DIR / "shap_importance.csv", index=False)

    shap.summary_plot(
        shap_values, X_transformed,
        feature_names=feature_names,
        show=False, max_display=15
    )
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 8))
    top15 = importance.head(15)
    plt.barh(top15["variable"][::-1], top15["SHAP_mean_abs"][::-1], color="#B85042")
    plt.xlabel("SHAP mean |valeur|")
    plt.title("Importance globale des variables (SHAP)")
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "shap_importance.png", dpi=120)
    plt.close()

    print(f"   Top 5 variables les plus importantes :")
    print(importance.head(5).round(4).to_string(index=False))

    # 2. INTERPRETATION LOCALE
    print("\n   Generation graphique interpretation locale...")
    p_te = pipe.predict_proba(sample[cat + num])[:, 1]
    sample = sample.copy()
    sample["score"] = p_te
    fraudes = sample[sample["fraude"] == 1].sort_values("score", ascending=False)

    if len(fraudes) > 0:
        idx = fraudes.index[0]
        pos = sample.index.get_loc(idx)

        expected = explainer.expected_value
        if hasattr(expected, '__len__'):
            expected = float(expected[0])
        else:
            expected = float(expected)

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[pos],
                base_values=expected,
                data=X_transformed[pos],
                feature_names=feature_names,
            ),
            show=False, max_display=12
        )
        plt.tight_layout()
        plt.savefig(cfg.FIG_DIR / "shap_local_top_fraude.png", dpi=120, bbox_inches="tight")
        plt.close()
        print("   Graphique local -> shap_local_top_fraude.png")
    else:
        print("   Aucune fraude dans l'echantillon local.")

    print(f"\n-> Resultats : {cfg.METRICS_DIR}")
    print(f"-> Figures   : {cfg.FIG_DIR}")


if __name__ == "__main__":
    main()
