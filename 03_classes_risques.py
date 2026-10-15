"""
03_classes_risques.py : regroupement des clients en 10 classes de risque (échelle de notation).

Entrée  : data/processed/ (produit par 02_modele.py)
Sortie  : data/processed/bornes.pkl et classes_test.pkl
          figures/calibration_classes.png

Utilisation (depuis la racine du projet) : python 03_classes_risques.py
La fenêtre du graphique doit être fermée pour que le script se termine.

Les bornes des classes sont définies sur le train (10 groupes de même effectif),
puis appliquées au test. Pour chaque classe, on compare la PD moyenne prédite au taux de défaut observé.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# ----- Chargement -----

pd_train = pd.read_pickle("data/processed/pd_train.pkl")
pd_test = pd.read_pickle("data/processed/pd_test.pkl")
y_test = pd.read_pickle("data/processed/y_test.pkl")

# Vérification : une PD par client, mêmes index pour pd_test et y_test.
print(pd_train.shape, pd_test.shape, y_test.shape)
print((pd_test.index == y_test.index).all())
print(pd_train.mean(), pd_test.mean())

# ----- Classes de risque : 10 groupes de même effectif sur le train -----

classes_train, bornes = pd.qcut(pd_train, q=10, retbins=True)
print(bornes)

# Première et dernière bornes élargies à l'infini : tout client du test reçoit une classe.
bornes[0] = -np.inf
bornes[-1] = np.inf

# Application au test : 1 = PD la plus basse, 10 = PD la plus haute.
classes_test = pd.cut(pd_test, bins=bornes, labels=range(1, 11))
print(classes_test.isna().sum(), classes_test.value_counts().sort_index())

# ----- Sauvegarde pour 04_calibration_stabilite.py -----

pd.Series(bornes).to_pickle("data/processed/bornes.pkl")
classes_test.to_pickle("data/processed/classes_test.pkl")

# ----- Tableau par classe (échantillon de test) -----

df_classes = pd.DataFrame({
    "classe": classes_test,
    "pd_pred": pd_test,
    "defaut": y_test
})

tableau = df_classes.groupby("classe", observed=True).agg(
    effectif=("defaut", "count"),
    pd_moyenne=("pd_pred", "mean"),
    taux_defaut=("defaut", "mean")
)

# Contrôles : tous les clients du test sont classés, et le taux de défaut augmente d'une classe à l'autre.
print(tableau)
print(tableau["effectif"].sum())
print(tableau["taux_defaut"].is_monotonic_increasing)

# ----- Graphique : PD prédite et taux de défaut observé par classe -----

classes = tableau.index.astype(int)
x = np.arange(len(classes))
largeur = 0.38

# Intervalle de confiance à 95 % du taux observé : 1,96 x erreur-type.
demi_largeur = 1.96 * np.sqrt(tableau["taux_defaut"] * (1 - tableau["taux_defaut"]) / tableau["effectif"])
fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(x - largeur / 2, tableau["pd_moyenne"] * 100, largeur, label="PD moyenne prédite", color="#2a78d6")
ax.bar(x + largeur / 2, tableau["taux_defaut"] * 100, largeur, yerr=demi_largeur * 100, capsize=3, label="Taux de défaut observé (IC à 95 %)", color="#eb6834")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_xlabel("Classe de risque (1 = la plus sûre, 10 = la plus risquée)")
ax.set_ylabel("Taux de défaut (%)")
ax.set_title("Calibration par classe de risque (échantillon de test)")
ax.legend(frameon=False)
ax.grid(axis="y", color="#e6e5e1", linewidth=1)
ax.set_axisbelow(True)
for cote in ["top", "right"]:
    ax.spines[cote].set_visible(False)

plt.tight_layout()
plt.savefig("figures/calibration_classes.png", dpi=150)
plt.show()
