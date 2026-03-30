# **Deciding with Numbers / Décider avec des chiffres**

**Régression linéaire : des données brutes aux décisions pondérées**

Ce dépôt fait partie de la série **AI From Scratch**.  
Il montre, étape par étape, comment un système d’IA simple peut **agréger de l’information**, attribuer des **poids** à différents signaux, puis produire une **décision**.

L’idée centrale est simple :

> un système d’IA ne “comprend” pas une situation comme un humain — il transforme des informations en nombres, combine ces nombres, puis décide à partir de cette combinaison.

Dans ce projet, on utilise la **régression linéaire** pour illustrer cette idée avec un exemple concret : **la prédiction des retards de vols**.

---

## Quick Overview

Ce repo contient deux façons complémentaires d’explorer la même idée :

- un **notebook** pour les débutants et les lecteurs non techniques
- des **scripts** pour une utilisation plus technique et plus professionnelle

### Le notebook

Le notebook (`manual.ipynb`) est la version pédagogique.  
Il explique d’abord l’intuition, puis introduit progressivement :

- les données tabulaires
- les types de variables
- les statistiques de base
- la corrélation
- la notion de poids
- la régression linéaire à la main avec NumPy
- la régression linéaire avec scikit-learn
- un exemple plus réaliste avec 50 000 vols
- l’impact de la préparation des données sur la qualité du modèle

Il est conçu pour les personnes qui veulent **comprendre ce qu’il se passe**, pas seulement exécuter du code.

### Les scripts

Les scripts sont la version plus “pro”.  
Ils chargent `dataset_vols.csv`, préparent les données, entraînent un modèle de régression, puis l’évaluent proprement.

Fichiers :

- [Python](script_py.py)
- [R](script.R)
- [Julia](script.jl)

Ce qu’ils font :

- chargent `dataset_vols.csv`
- utilisent les 25 variables d’entrée et la cible `retard_arrivee_min`
- proposent 3 stratégies de préparation :
  - `proper` : nettoyage propre, imputation, one-hot encoding, normalisation
  - `minimal` : nettoyage minimal, suppression des valeurs manquantes, one-hot encoding
  - `raw` : version volontairement dégradée pour montrer l’impact d’une information de mauvaise qualité
- affichent :
  - le nombre de lignes utilisées
  - le nombre de variables utilisées
  - les décisions de préparation
  - la MAE
  - la RMSE
  - le R²

Commandes typiques :

```bash
python script_py.py --data dataset_vols.csv --strategy proper
````

```bash id="cexsae"
Rscript script.R dataset_vols.csv proper
```

```bash id="z0di76"
julia script.jl dataset_vols.csv proper
```

Stratégies disponibles :

* `proper`
* `minimal`
* `raw`

---

## Pourquoi ce repo existe

Ce projet ne parle pas seulement de régression.

Il sert aussi à montrer une idée plus large :

* un système d’IA consomme de l’information
* cette information doit être choisie, nettoyée, structurée et encodée
* ensuite, le système l’agrège
* et ce n’est qu’après cela qu’il peut produire une décision utile

L’objectif n’est donc pas seulement “d’ajuster un modèle”.
L’objectif est de montrer que :

**la qualité d’une décision dépend fortement de la qualité de l’information agrégée.**

C’est vrai pour le machine learning tabulaire, et cela prépare aussi une intuition plus générale pour mieux comprendre les systèmes d’IA modernes.

---

## Structure du repo

```text id="uip831"
bp1-linear-regression/
├── dataset_vols.csv      # Dataset principal : 50 000 vols, 25 variables + cible
├── manual.ipynb          # Notebook pédagogique pas à pas
├── README.md             # Présentation du projet
├── script.jl             # Implémentation Julia
├── script_py.py          # Implémentation Python
└── script.R              # Implémentation R
```

### Description rapide des fichiers

* `dataset_vols.csv`
  Dataset synthétique utilisé dans le projet. Il contient 50 000 vols et plusieurs types de variables :
  numériques, binaires et catégorielles.

* `manual.ipynb`
  Notebook pédagogique principal. C’est le meilleur point d’entrée pour les lecteurs non techniques.

* `script_py.py`
  Implémentation professionnelle du workflow en Python.

* `script.R`
  Implémentation professionnelle du workflow en R.

* `script.jl`
  Implémentation professionnelle du workflow en Julia.

* `README.md`
  Ce fichier.

---

## Ce qu’on apprend dans ce projet

En parcourant ce dépôt, on comprend que :

* une prédiction est une forme de **décision**
* une décision se construit en **agrégeant plusieurs signaux**
* la régression linéaire est une méthode simple et puissante pour combiner de l’information pondérée
* le modèle n’est qu’une partie du système
* la **qualité des données** et la **qualité de leur préparation** jouent un rôle majeur

---

## Parcours conseillé

Si tu débutes :

1. commence par `manual.ipynb`
2. comprends l’intuition derrière les poids et l’agrégation
3. explore l’exemple réaliste sur les retards de vols
4. puis passe aux scripts pour voir une version plus orientée “production”

Si tu es déjà technique :

1. regarde directement les scripts
2. compare les différentes stratégies de préparation
3. observe comment la qualité des données influence la performance du modèle

---

## Message principal

L’idée centrale de ce dépôt est la suivante :

> **L’IA commence par l’information.**
> Avant de construire un bon modèle, il faut construire un bon système de décision.
> Et avant de construire un bon système de décision, il faut de bonnes informations à agréger.

---

## Et ensuite ?

Ce dépôt se concentre sur la **régression linéaire**, où la sortie est une valeur continue.

La suite naturelle est la **régression logistique**, où le système ne prédit plus “combien”, mais **dans quelle catégorie** un exemple a le plus de chances de tomber.

Autrement dit, la prochaine étape n’est plus :

* “combien de minutes de retard ?”

mais plutôt :

* “retard ou pas de retard ?”
* ou plus généralement : “dans quelle classe ranger cet exemple ?”

On retrouvera alors exactement la même idée de fond :

**prendre de l’information, lui donner un poids, l’agréger, puis décider.**