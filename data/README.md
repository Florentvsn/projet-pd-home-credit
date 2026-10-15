# Données

Les données ne sont pas incluses dans ce dépôt : elles proviennent d'une compétition Kaggle dont les conditions n'autorisent pas la redistribution, et `application_train.csv` dépasse la limite de taille de GitHub (100 Mo par fichier).

## Télécharger le fichier

1. Se connecter à Kaggle (compte gratuit).
2. Ouvrir la compétition [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) et accepter ses règles (nécessaire pour télécharger).
3. Dans l'onglet *Data*, télécharger `application_train.csv`.
4. Le placer dans ce dossier :

```
data/
└── application_train.csv
```

## Dossier `processed/`

Le dossier `data/processed/` est créé automatiquement par `01_preparation.py` et contient les fichiers intermédiaires (`.pkl`) échangés entre les scripts. Il peut être supprimé et régénéré à tout moment en relançant les scripts dans l'ordre.

## Autres fichiers de la compétition

Cette version du projet n'utilise que `application_train.csv`. Les autres fichiers (`bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv`) sont prévus pour une version ultérieure.
