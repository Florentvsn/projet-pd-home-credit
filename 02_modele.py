"""
02_modele.py : modèle de probabilité de défaut (PD) par régression logistique.

Entrée  : data/processed/ (produit par 01_preparation.py)
Sortie  : data/processed/pd_train.pkl et pd_test.pkl (PD de chaque client)
          figures/coefficients_top15.png

Utilisation (depuis la racine du projet) : python 02_modele.py

Contenu : variable CREDIT_SUR_PRIX, standardisation, régression logistique (AUC, Gini),
interprétation des coefficients, comparaison avec un gradient boosting.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

# ----- Chargement des données préparées -----

X_train_enc = pd.read_pickle("data/processed/X_train.pkl")
X_test_enc = pd.read_pickle("data/processed/X_test.pkl")
y_train = pd.read_pickle("data/processed/y_train.pkl")
y_test = pd.read_pickle("data/processed/y_test.pkl")

print(X_train_enc.shape, X_test_enc.shape)
print(y_train.mean(), y_test.mean())

# ----- Variable CREDIT_SUR_PRIX (remplace AMT_GOODS_PRICE) -----

# AMT_CREDIT et AMT_GOODS_PRICE sont quasi identiques (corrélation d'environ 0,987) : avec les deux,
# la régression logistique donne des coefficients opposés et très grands, qui sont un artefact et non un effet réel.
# On remplace AMT_GOODS_PRICE par le ratio crédit / prix du bien (part du crédit qui ne finance pas le bien).
# Calcul ligne par ligne : aucune statistique du train n'est utilisée, donc pas de fuite d'information.
# Comparaison des trois variantes : voir README.
print(X_train_enc["AMT_CREDIT"].corr(X_train_enc["AMT_GOODS_PRICE"]))

X_train_enc["CREDIT_SUR_PRIX"] = X_train_enc["AMT_CREDIT"] / X_train_enc["AMT_GOODS_PRICE"]
X_test_enc["CREDIT_SUR_PRIX"] = X_test_enc["AMT_CREDIT"] / X_test_enc["AMT_GOODS_PRICE"]

X_train_enc = X_train_enc.drop(columns=["AMT_GOODS_PRICE"])
X_test_enc = X_test_enc.drop(columns=["AMT_GOODS_PRICE"])

# ----- Standardisation (moyenne 0 et écart-type 1 par colonne) -----

# La régression logistique n'accepte pas de valeurs manquantes.
print(X_train_enc.isna().sum().sum(), X_test_enc.isna().sum().sum())

# fit_transform sur le train : mesure et applique. transform seul sur le test : applique les mesures du train.
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_enc), columns=X_train_enc.columns, index=X_train_enc.index)
X_test_sc = pd.DataFrame(scaler.transform(X_test_enc), columns=X_test_enc.columns, index=X_test_enc.index)

# Contrôles : moyennes du train proches de 0 et écarts-types égaux à 1 ; moyennes du test proches de 0 sans l'être exactement.
print(X_train_sc.mean().abs().max())
print(X_train_sc.std().round(2).value_counts())
print(X_test_sc.mean().abs().max())

# ----- Régression logistique : entraînement sur le train, PD sur le test -----

logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_sc, y_train)
pd_test = logreg.predict_proba(X_test_sc)[:, 1]

# La PD moyenne doit être proche du taux de défaut réel (environ 0,08).
print(pd_test.mean(), pd_test.min(), pd_test.max())

# ----- AUC et Gini (qualité du classement des clients) -----

roc_auc = roc_auc_score(y_test, pd_test)
Gini = 2*roc_auc - 1
print(roc_auc, Gini)

# ----- Interprétation : coefficients -----

# Signe positif : la variable augmente le risque de défaut. Signe négatif : elle le diminue.
# Les colonnes étant standardisées, les coefficients sont comparables entre eux.
coeff = logreg.coef_.ravel()

df_logreg = pd.DataFrame({
    "variables": X_train_sc.columns,
    "coef": coeff
})

df_logreg["abs_coef"] = df_logreg["coef"].abs()

print(df_logreg.sort_values(by="abs_coef", ascending=False).head(15))

# ----- Figure : les 15 coefficients les plus influents -----

os.makedirs("figures", exist_ok=True)

top15 = df_logreg.sort_values(by="abs_coef", ascending=False).head(15).iloc[::-1]
couleur_baisse = "#2a78d6"
couleur_hausse = "#e34948"
couleurs = [couleur_baisse if c < 0 else couleur_hausse for c in top15["coef"]]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top15["variables"], top15["coef"], height=0.6, color=couleurs)
ax.axvline(0, color="#52514e", linewidth=1)
ax.set_xlabel("Coefficient de la régression logistique (variables standardisées)")
ax.set_title("Les 15 variables les plus influentes sur la probabilité de défaut")
ax.legend(handles=[Patch(color=couleur_baisse, label="Diminue le risque de défaut"),
                   Patch(color=couleur_hausse, label="Augmente le risque de défaut")],
          loc="lower left", frameon=False)
ax.grid(axis="x", color="#e6e5e1", linewidth=1)
ax.set_axisbelow(True)
for cote in ["top", "right"]:
    ax.spines[cote].set_visible(False)

plt.tight_layout()
plt.savefig("figures/coefficients_top15.png", dpi=150)
plt.close(fig)

# ----- Benchmark : gradient boosting -----

# Modèle non linéaire (arbres) : combien de Gini perd-on en choisissant le modèle interprétable ?
# Les arbres n'ont pas besoin de standardisation : on utilise X_train_enc et X_test_enc.
model_grad = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, early_stopping=True, random_state=42)
model_grad.fit(X_train_enc, y_train)
pd_test_grad = model_grad.predict_proba(X_test_enc)[:, 1]
print(pd_test_grad.mean())

roc_auc_grad = roc_auc_score(y_test, pd_test_grad)
Gini_grad = 2*roc_auc_grad - 1
print(roc_auc_grad, Gini_grad)

# ----- Sauvegarde des PD pour 03_classes_risques.py -----

pd_train = logreg.predict_proba(X_train_sc)[:, 1]

pd_train = pd.Series(pd_train, index=X_train_enc.index)
pd_test = pd.Series(pd_test, index=X_test_enc.index)

pd_train.to_pickle("data/processed/pd_train.pkl")
pd_test.to_pickle("data/processed/pd_test.pkl")
