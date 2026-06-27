# Scoring de fraude bancaire — M2 TIDE

Détection de transactions frauduleuses par carte bancaire.

Le projet enchaîne trois étapes :

- un pipeline de préparation en architecture médaillon (raw vers clean vers gold)
- un screening des variables par Information Value (IV) sur le train seul
- une comparaison de quatre modèles de scoring, du plus interprétable au plus performant

## Données

- Une ligne = une transaction réalisée par une carte
- 1 151 432 transactions, de novembre 2003 à juin 2004
- Cible : fraude (0 ou 1), très déséquilibrée (environ 0,63 % de fraudes)
- Le fichier parquet de chaque couche fait foi (complet) ; les fichiers xlsx ne sont que des échantillons, car Excel plafonne à environ 1,05 M lignes alors qu'on en a 1,15 M

## Ce qui a été fait et pourquoi

Préparation des données (pipeline médaillon)

- raw : lecture de la source SAS et conversion en parquet, sans transformation
- clean : recombinaison de Date et Heure en datetime, typage et nettoyage
- gold : table finale prête pour la modélisation
- But : séparer clairement chargement, nettoyage et table d'analyse pour garder un pipeline lisible et reproductible

Screening des variables par Information Value

- L'IV mesure le pouvoir prédictif d'une variable vis-à-vis de la cible
- Il est calculé sur le train uniquement, pour éviter toute fuite d'information
- Résultat clé : les familles Sum et Velocity sont très prédictives (IV 0,47 à 0,79) alors que leur corrélation de Pearson est faible. La relation est non linéaire ou à seuil, ce qu'une simple corrélation ne voyait pas

Comparaison de modèles

- Quatre modèles entraînés sur le passé et testés sur le futur (mai-juin 2004)
- La métrique principale est la PR-AUC, adaptée aux classes très déséquilibrées ; l'accuracy n'a aucun sens ici
- Le lift et les tables de seuil servent à prioriser les dossiers à contrôler

Test d'ablation sur FM_Difference_Pays

- Cette variable a un IV proche de zéro : elle n'apporte rien
- En la retirant, la PR-AUC du modèle retenu passe de 0,178 à 0,219 (gain de 0,04)
- Elle est donc exclue par défaut des features ; les colonnes restent dans gold et peuvent être réintégrées via get_feature_lists(keep_diff_pays=True)

Scorecard WoE interprétable

- Encodage WoE appris sur le train, puis régression logistique
- Produit une grille de points (scorecard_points.csv), où un score élevé correspond à un risque élevé
- Détail notable : le code de refus reçoit des points négatifs car, à comportement égal, la majorité des fraudes passent en accepté

## Résultats sur le test temporel (mai-juin 2004)

- Logistique simple : ROC 0,827, PR-AUC 0,112, lift@1% 27, lift@10% 6,0
- Scorecard WoE (interprétable) : ROC 0,829, PR-AUC 0,120, lift@1% 26, lift@10% 6,2
- Random Forest : ROC 0,836, PR-AUC 0,181, lift@1% 31, lift@10% 6,5
- HistGradientBoosting (retenu) : ROC 0,831, PR-AUC 0,219, lift@1% 37, lift@10% 6,4

Le HistGradientBoosting est retenu pour sa PR-AUC nettement supérieure et son meilleur lift au sommet du score.

## Principes anti-fuite

- Split strictement temporel : on apprend sur le passé, on teste sur le futur
- Tout ce qui dérive de la cible (WoE, IV, TargetEncoder) est appris sur le train seul ; rien n'est figé dans la table gold
- Les codes 41 et 43 (carte perdue ou volée) sont quasi circulaires ; ils peuvent être retirés via get_feature_lists(drop_near_leak=True) pour mesurer une performance honnête

## Arborescence commentée

```
projet-scoring-fraude/
├── environment.yml                  dépendances conda
├── proxy_off.ps1                    désactive le proxy d'entreprise (utilitaire)
├── env/                             environnement conda local (non versionné)
├── README.md                        ce document
├── data/
│   ├── raw/                         couche brute (raw.parquet + échantillon xlsx)
│   ├── clean/                       couche nettoyée et typée (clean.parquet + xlsx)
│   ├── gold/                        table finale à modéliser (gold.parquet + xlsx)
│   └── results/
│       ├── figures/                 courbes ROC, PR, lift, IV, comparaison
│       │   └── eda/                 figures d'exploration
│       └── metrics/                 récap, tables de seuil, IV, scorecard, ablation
└── src/
    ├── config.py                    chemins, date de coupure, constantes
    ├── pipeline_preprocess/
    │   ├── raw.py                   SAS -> raw.parquet (copie fidèle)
    │   ├── clean.py                 typage + horodatage reconstruit
    │   └── gold.py                  feature engineering row-local
    ├── exploration/
    │   ├── viz.py                   distributions par cible (cat & continu)
    │   ├── information_value.py     IV train-only + classement
    │   └── run_eda.py               génère les figures EDA
    └── modelisation/
        ├── features.py             listes features, split temporel, préprocesseur
        ├── encoders.py             WoEEncoder (train-only)
        ├── evaluation.py           métriques, table de seuil, courbes
        ├── logistic_regression.py  baseline interprétable
        ├── logistic_scorecard.py   scorecard WoE + grille de points
        ├── random_forest.py        forêt aléatoire
        ├── hist_gradient_boosting.py  boosting (modèle retenu)
        ├── ablation_difference_pays.py  test d'une feature inutile
        └── run_all.py              4 modèles + comparaison
```

Détail par branche

Racine

- environment.yml : dépendances conda du projet
- proxy_off.ps1 : désactive le proxy d'entreprise (utilitaire local)
- env/ : environnement conda local, non versionné, à créer
- README.md : ce document

Données (data/)

- raw/ : couche brute, copie fidèle de la source SAS (raw.parquet plus échantillon xlsx)
- clean/ : couche nettoyée et typée (clean.parquet plus échantillon xlsx)
- gold/ : table finale prête à modéliser (gold.parquet plus échantillon xlsx)
- results/figures/ : courbes ROC, PR, lift par modèle, classement IV, comparaison et sous-dossier eda
- results/metrics/ : récapitulatif des modèles, tables de seuil, classement IV, grille de points, ablation

Code (src/)

config.py

- Point unique de configuration : tous les scripts s'y réfèrent
- Bloc chemins : racine, dossiers et fichiers parquet de chaque couche
- Bloc paramètres métier : date de coupure train/test (CUTOFF), codes carte perdue ou volée, familles de variables glissantes sur 4 fenêtres (3, 6, 12, 24 h)
- Fonction excel_sample : écrit un échantillon Excel stratifié (toutes les fraudes plus un tirage de non-fraudes)

pipeline_preprocess/ (architecture médaillon, un script par couche)

- raw.py : lit le fichier SAS et le matérialise en parquet, sans aucune transformation
- clean.py : recombine Date et Heure en un vrai datetime, type les colonnes (identifiants et catégorielles, montant en float), supprime les doublons et trie par temps ; aucune variable dérivée de la cible ici
- gold.py : feature engineering row-local (la table est identique en train et en test). Blocs : variables de temps (heure, jour, nuit, weekend), montant (log et arrondi), 16 variables glissantes brutes, ratios décorrélés 3h/24h, seuil de vélocité, flags de code réponse. Les encodages dérivés de la cible sont volontairement absents

exploration/ (EDA et Information Value)

- viz.py : fonctions de visualisation, distribution d'une variable par rapport à la cible, en catégoriel et en continu
- information_value.py : calcule l'IV de chaque variable sur le train seul (binning par quantiles pour les continues), produit le classement iv_ranking et son graphique
- run_eda.py : génère les figures EDA dans results/figures/eda (continues tracées sur un échantillon pour la rapidité)

modelisation/ (briques communes, modèles et orchestration)

- features.py : listes des variables, split strictement temporel, préprocesseur anti-fuite (TargetEncoder à cross-fitting plus scaling pour les modèles linéaires). Options pour retirer les variables quasi circulaires ou réintégrer FM_Difference_Pays
- encoders.py : WoEEncoder, encodage Weight of Evidence ajusté sur le train seul (brique du scorecard bancaire)
- evaluation.py : métriques adaptées à la fraude (ROC-AUC, PR-AUC, lift), table de seuil pour prioriser les dossiers, sauvegarde des courbes ROC, PR et lift
- logistic_regression.py : baseline interprétable sur le jeu de variables réduit et décorrélé, avec scaling
- logistic_scorecard.py : régression logistique sur variables WoE ; produit une grille de points (scorecard_points.csv) où un score élevé signifie un risque élevé
- random_forest.py : forêt aléatoire, robuste à la colinéarité, sans scaling, réglages maîtrisés pour rester rapide
- hist_gradient_boosting.py : boosting d'arbres, le modèle retenu (meilleure PR-AUC), avec early stopping et pondération des classes
- ablation_difference_pays.py : compare le modèle retenu avec et sans la famille FM_Difference_Pays pour mesurer son apport réel
- run_all.py : entraîne les quatre modèles, compile le récapitulatif et le graphique de comparaison PR-AUC

## Installation

- conda env create -f environment.yml -p ./env
- conda activate ./env

## Lancement

Les données pré-générées sont incluses. Pour tout régénérer depuis la source, placer le fichier SAS dans data/raw, puis exécuter dans l'ordre :

- python src/pipeline_preprocess/raw.py
- python src/pipeline_preprocess/clean.py
- python src/pipeline_preprocess/gold.py
- python src/exploration/information_value.py
- python src/exploration/run_eda.py
- python src/modelisation/run_all.py
- python src/modelisation/ablation_difference_pays.py

## Dictionnaire de données (couche raw et clean)

- Carte : numéro de carte, identifiant (environ 198k cartes)
- Pays : code pays de la transaction (167 modalités)
- Date et Heure : jour et heure, recombinés en datetime au clean
- CodeRep : code réponse de l'autorisation (00 = accepté, sinon refus)
- MCC : code commerçant (661 modalités)
- Montant : montant de la transaction en euros
- fraude : cible, 1 = frauduleuse, 0 = saine
- FM_Velocity_Condition_3/6/12/24 : nombre de transactions acceptées sur les X dernières heures
- FM_Sum_3/6/12/24 : montant cumulé sur les X dernières heures
- FM_Redondance_MCC_3/6/12/24 : nombre de transactions chez le même commerçant sur X heures
- FM_Difference_Pays_3/6/12/24 : nombre de pays différents sur X heures (exclue du modèle, IV proche de zéro)
