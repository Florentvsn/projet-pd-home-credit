"""
01_preparation.py : préparation des données (nettoyage, découpage train/test, encodage).

Entrée  : data/application_train.csv (à télécharger sur Kaggle, voir data/README.md)
Sortie  : data/processed/ -> X_train, X_test, y_train, y_test, ids_test (fichiers .pkl)

Utilisation (depuis la racine du projet) : python 01_preparation.py

Règle appliquée partout : toute statistique (médiane, plafond, modalités fréquentes)
est calculée sur le train uniquement, puis appliquée au train et au test.
Le détail et la justification de chaque choix sont dans le README.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ----- Lecture des données -----

df = pd.read_csv('data/application_train.csv')

# ----- Suppression des colonnes à plus de 50 % de valeurs manquantes -----

# Exceptions : OWN_CAR_AGE et EXT_SOURCE_1, jugées potentiellement importantes.
# Limite connue : ce taux est calculé sur tout le fichier, avant le découpage train/test.
taux_manquants = df.isna().mean()
liste_colonnes_a_supprimer = list(taux_manquants[taux_manquants>0.5].index)
liste_colonnes_a_supprimer.remove('OWN_CAR_AGE')
liste_colonnes_a_supprimer.remove('EXT_SOURCE_1')

df_clean = df.drop(columns=liste_colonnes_a_supprimer)
print(df_clean.shape, "TARGET" in df_clean.columns, "OWN_CAR_AGE" in df_clean.columns, "EXT_SOURCE_1" in df_clean.columns)

# ----- Découpage train / test (80 / 20, stratifié sur TARGET) -----

train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean["TARGET"])
print(train_df.shape, test_df.shape)
print(train_df["TARGET"].mean(), test_df["TARGET"].mean())

# ----- Valeurs manquantes : colonnes indicatrices -----

# L'absence d'information est conservée comme variable (0 = renseigné, 1 = manquant).
train_df["BUREAU_MANQUANT"] = train_df["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().astype(int)
train_df["EXT_SOURCE_1_MANQUANT"] = train_df["EXT_SOURCE_1"].isna().astype(int)
test_df["BUREAU_MANQUANT"] = test_df["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().astype(int)
test_df["EXT_SOURCE_1_MANQUANT"] = test_df["EXT_SOURCE_1"].isna().astype(int)

print(train_df.shape, test_df.shape)

# ----- Valeurs manquantes : remplissage -----

# OWN_CAR_AGE : valeurs manquantes remplacées par 0.
train_df["OWN_CAR_AGE"] = train_df["OWN_CAR_AGE"].fillna(0)
test_df["OWN_CAR_AGE"] = test_df["OWN_CAR_AGE"].fillna(0)

# Colonnes catégorielles : modalité "Manquant".
train_df["OCCUPATION_TYPE"] = train_df["OCCUPATION_TYPE"].fillna("Manquant")
train_df["NAME_TYPE_SUITE"] = train_df["NAME_TYPE_SUITE"].fillna("Manquant")
train_df["EMERGENCYSTATE_MODE"] = train_df["EMERGENCYSTATE_MODE"].fillna("Manquant")
test_df["OCCUPATION_TYPE"] = test_df["OCCUPATION_TYPE"].fillna("Manquant")
test_df["NAME_TYPE_SUITE"] = test_df["NAME_TYPE_SUITE"].fillna("Manquant")
test_df["EMERGENCYSTATE_MODE"] = test_df["EMERGENCYSTATE_MODE"].fillna("Manquant")

# Colonnes numériques restantes : médiane du train.
taux_manquants = train_df.isna().mean()
manquants = taux_manquants[taux_manquants>0].sort_values(ascending=False)

series_medianes = train_df[manquants.index].median()
train_df[manquants.index] = train_df[manquants.index].fillna(series_medianes)
test_df[manquants.index] = test_df[manquants.index].fillna(series_medianes)

print(len(manquants))
print(train_df.isna().sum().sum(), test_df.isna().sum().sum())
print(train_df.shape, test_df.shape)

# ----- DAYS_EMPLOYED : valeur aberrante 365243 -----

# 365243 jours (environ 1000 ans) est une valeur sentinelle associée à l'absence d'emploi : remplacée par la médiane du train.
# SANS_EMPLOI conserve l'information (utilisée pour contrôle uniquement, voir plus bas).
train_df["SANS_EMPLOI"] = (train_df["DAYS_EMPLOYED"] == 365243).astype(int)
test_df["SANS_EMPLOI"] = (test_df["DAYS_EMPLOYED"] == 365243).astype(int)

mediane_emploi = train_df.loc[train_df["DAYS_EMPLOYED"] != 365243, "DAYS_EMPLOYED"].median()

train_df["DAYS_EMPLOYED"] = train_df["DAYS_EMPLOYED"].replace(365243, mediane_emploi)
test_df["DAYS_EMPLOYED"] = test_df["DAYS_EMPLOYED"].replace(365243, mediane_emploi)

print(train_df["DAYS_EMPLOYED"].max(), test_df["DAYS_EMPLOYED"].max())
print(train_df["DAYS_EMPLOYED"].mean())
print(train_df.shape, test_df.shape)
print(train_df["SANS_EMPLOI"].mean())

# ----- AMT_INCOME_TOTAL : plafonnement au 99,9e percentile du train -----

# Quelques revenus extrêmes (plusieurs millions) fausseraient les coefficients.
plafond_revenu = train_df["AMT_INCOME_TOTAL"].quantile(0.999)
train_df["AMT_INCOME_TOTAL"] =  train_df["AMT_INCOME_TOTAL"].clip(upper=plafond_revenu)
test_df["AMT_INCOME_TOTAL"] = test_df["AMT_INCOME_TOTAL"].clip(upper=plafond_revenu)
print(plafond_revenu, train_df["AMT_INCOME_TOTAL"].max(), test_df["AMT_INCOME_TOTAL"].max())

# ----- ORGANIZATION_TYPE : regroupement des modalités rares -----

# Les organisations représentant moins de 1 % du train sont regroupées dans "RARE" :
# trop peu de clients pour estimer leur taux de défaut de façon fiable.
seuil = 0.01 * len(train_df)
effectifs =  train_df["ORGANIZATION_TYPE"].value_counts()
frequentes = effectifs[effectifs >= seuil].index
train_df["ORGANIZATION_TYPE"] = train_df["ORGANIZATION_TYPE"].where(train_df["ORGANIZATION_TYPE"].isin(frequentes),"RARE")
test_df["ORGANIZATION_TYPE"] = test_df["ORGANIZATION_TYPE"].where(test_df["ORGANIZATION_TYPE"].isin(frequentes),"RARE")
print(train_df["ORGANIZATION_TYPE"].nunique(), test_df["ORGANIZATION_TYPE"].nunique())
print((train_df["ORGANIZATION_TYPE"] == "RARE").sum())
print(train_df.groupby("ORGANIZATION_TYPE")["TARGET"].agg(["mean", "count"]).loc["RARE"])

# ----- CODE_GENDER : variable sensible supprimée -----

# Raisons réglementaires (non-discrimination), voir README.
train_df = train_df.drop(columns=["CODE_GENDER"])
test_df = test_df.drop(columns=["CODE_GENDER"])
print(train_df.shape, test_df.shape)

# ----- Séparation X / y -----

# y : la réponse à prédire.
y_train = train_df["TARGET"]
y_test = test_df["TARGET"]

# Identifiants du test conservés hors du modèle, pour retrouver à quel client correspond chaque prédiction.
ids_test = test_df["SK_ID_CURR"]

# X : sans TARGET, sans identifiant, et sans SANS_EMPLOI (doublon de ORGANIZATION_TYPE_XNA).
X_train = train_df.drop(columns=["TARGET", "SK_ID_CURR", "SANS_EMPLOI"])
X_test = test_df.drop(columns=["TARGET", "SK_ID_CURR", "SANS_EMPLOI"])

print(X_train.shape, X_test.shape)
print(y_train.mean(), y_test.mean())

# ----- Encodage des variables catégorielles -----

# Regroupement des modalités rares de NAME_INCOME_TYPE et NAME_FAMILY_STATUS (seuil 0,1 % du train).
seuil_cat = 0.001 * len(X_train)
for col in ["NAME_INCOME_TYPE", "NAME_FAMILY_STATUS"]:
    eff = X_train[col].value_counts()
    frequentes_col = eff[eff >= seuil_cat].index
    X_train[col] = X_train[col].where(X_train[col].isin(frequentes_col), "RARE")
    X_test[col] = X_test[col].where(X_test[col].isin(frequentes_col), "RARE")

# Suppression des FLAG_DOCUMENT n'ayant que très peu de clients à 1 (moins de 100).
flags_rares = ["FLAG_DOCUMENT_12", "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_4",
               "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_21"]
X_train = X_train.drop(columns=flags_rares)
X_test = X_test.drop(columns=flags_rares)

# One-hot encoding : colonnes texte -> colonnes 0/1 (la première modalité de chaque variable sert de référence).
X_train_enc = pd.get_dummies(X_train, drop_first=True, dtype=int)
X_test_enc = pd.get_dummies(X_test, drop_first=True, dtype=int)

# Le test reçoit exactement les mêmes colonnes que le train, dans le même ordre.
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

# Vérifications : même colonnes, plus aucune colonne texte.
print(X_train_enc.shape, X_test_enc.shape)
print((X_train_enc.columns == X_test_enc.columns).all())
print(X_train_enc.select_dtypes(exclude="number").columns.tolist())

# NAME_FAMILY_STATUS_RARE ne concerne que 2 clients : colonne supprimée.
X_train_enc = X_train_enc.drop(columns=["NAME_FAMILY_STATUS_RARE"])
X_test_enc = X_test_enc.drop(columns=["NAME_FAMILY_STATUS_RARE"])
print(X_train_enc.shape, X_test_enc.shape)

# ----- Sauvegarde pour 02_modele.py -----

os.makedirs("data/processed", exist_ok=True)

X_train_enc.to_pickle("data/processed/X_train.pkl")
X_test_enc.to_pickle("data/processed/X_test.pkl")
y_train.to_pickle("data/processed/y_train.pkl")
y_test.to_pickle("data/processed/y_test.pkl")
ids_test.to_pickle("data/processed/ids_test.pkl")
