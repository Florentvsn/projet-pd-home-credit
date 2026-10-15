"""
04_calibration_stabilite.py : contrôles de calibration et de stabilité du modèle.

Entrée  : data/processed/ (produit par 02_modele.py et 03_classes_risques.py)
Sortie  : affichage des résultats (aucun fichier)

Utilisation (depuis la racine du projet) : python 04_calibration_stabilite.py

Contrôles :
  1. score de Brier, comparé à celui d'une PD constante égale au taux de défaut moyen ;
  2. test de calibration par classe (z-score) et intervalle de confiance à 95 % ;
  3. stabilité de la répartition des clients entre train et test (PSI).
"""

import pandas as pd
from sklearn.metrics import brier_score_loss
import numpy as np

# ----- Chargement -----

pd_test = pd.read_pickle("data/processed/pd_test.pkl")
y_test = pd.read_pickle("data/processed/y_test.pkl")
classes_test = pd.read_pickle("data/processed/classes_test.pkl")

# Vérification : même taille et même index pour les trois séries.
print(pd_test.shape, y_test.shape, classes_test.shape)
print((pd_test.index == y_test.index).all(), (pd_test.index == classes_test.index).all())
print(classes_test.isna().sum())

# ----- Score de Brier -----

# Moyenne des carrés des écarts entre la PD et le défaut réel (plus bas = mieux).
# Référence : une PD constante égale au taux de défaut moyen p, dont le score vaut p * (1 - p).
# Le modèle doit faire mieux que cette référence pour apporter de l'information.
p = y_test.mean()
print(p, p * (1 -p))

score_brier = ((pd_test - y_test)**2).mean()
print(score_brier)

# Vérification avec la fonction de scikit-learn.
print(brier_score_loss(y_test, pd_test))

# ----- Calibration par classe : z-score -----

# z = (taux observé - PD prédite) / erreur-type attendue si le modèle est juste.
# |z| < 1,96 : écart compatible avec le hasard (seuil de 5 %, loi normale).
df_classes = pd.DataFrame({
    "classe_test": classes_test,
    "pd_pred": pd_test,
    "defaut": y_test
})

tableau = df_classes.groupby("classe_test", observed=True).agg(
    effectif=("defaut", "count"),
    pd_moyenne=("pd_pred", "mean"),
    taux_defaut=("defaut", "mean")
)

tableau["z"] = (tableau["taux_defaut"] - tableau["pd_moyenne"]) / np.sqrt(tableau["pd_moyenne"] * (1 - tableau["pd_moyenne"]) / tableau["effectif"])

# ----- Intervalle de confiance à 95 % du taux de défaut observé -----

# Demi-largeur : 1,96 x erreur-type du taux observé.
tableau["demi_largeur"] = 1.96 * np.sqrt(tableau["taux_defaut"] * (1 - tableau["taux_defaut"]) / tableau["effectif"])
tableau["ic_bas"] = tableau["taux_defaut"] - tableau["demi_largeur"]
tableau["ic_haut"] = tableau["taux_defaut"] + tableau["demi_largeur"]

# La PD moyenne prédite est-elle dans l'intervalle du taux observé ?
tableau["pd_dans_ic"] = (tableau["pd_moyenne"] >= tableau["ic_bas"]) & (tableau["pd_moyenne"] <= tableau["ic_haut"])

print(tableau)

# ----- Stabilité : PSI (Population Stability Index) -----

# Les mêmes bornes sont appliquées au train et au test, puis on compare la part des clients de chaque classe.
# PSI = somme sur les 10 classes de (part_test - part_train) x ln(part_test / part_train)
# Seuils usuels : PSI < 0,10 population stable ; de 0,10 à 0,25 dérive modérée ; > 0,25 dérive forte.
# Limite : le découpage train/test est aléatoire, la stabilité est donc attendue (pas de test dans le temps).
pd_train = pd.read_pickle("data/processed/pd_train.pkl")
bornes = pd.read_pickle("data/processed/bornes.pkl").values
classes_train = pd.cut(pd_train, bins=bornes, labels=range(1, 11))

print(pd_train.shape, bornes.shape)
print(classes_train.isna().sum())
print(classes_train.value_counts().sort_index())

part_train = classes_train.value_counts(normalize=True).sort_index()
part_test = classes_test.value_counts(normalize=True).sort_index()
print(part_train, part_test)

psi = ((part_test - part_train) * np.log(part_test / part_train)).sum()
print(psi)
