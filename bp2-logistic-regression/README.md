# Logistic Regression / Perceptron complet

Ce dossier constitue la **suite directe de BP1**.

Dans le premier chapitre, on a construit la logique de base :

- agréger plusieurs informations ;
- apprendre des poids ;
- mesurer l’erreur ;
- corriger les paramètres ;
- comprendre le **perceptron simple**.

Ici, on va un cran plus loin :

- le score brut ne suffit plus ;
- on introduit une **fonction d’activation** ;
- on transforme un score en **probabilité** ;
- on construit le **perceptron complet** ;
- on arrive à la **régression logistique**.

Autrement dit, ce chapitre introduit une idée centrale :

> un neurone artificiel = partie linéaire + biais + activation + apprentissage.

---

## Contenu du dossier

- `2_manual.ipynb`  
  Notebook principal d’accompagnement du blog post.  
  Il montre pas à pas :
  - le passage du score à la probabilité ;
  - la sigmoïde ;
  - la log-loss ;
  - le neurone artificiel binaire ;
  - la régression logistique ;
  - les limites d’un neurone seul.

- `2_script.py`  
  Implémentation Python plus scriptée du cas d’application.

- `2_script.R`  
  Version R du même cas.

- `2_script.jl`  
  Version Julia du même cas.

- `dataset_streaming_churn.csv`  
  Dataset utilisé dans le cas d’application concret.

- `README.md`  
  Ce fichier.

---

## Cas d’usage

Le cas illustré ici est un problème moderne et concret :

**prédire le churn d’un utilisateur sur une plateforme numérique**.

La cible est binaire :

- `1` → l’utilisateur quitte la plateforme
- `0` → l’utilisateur reste

Cela permet d’illustrer naturellement :

- la classification binaire ;
- la sortie probabiliste ;
- la décision à partir d’un seuil ;
- la régression logistique comme premier vrai neurone.

---

## À quoi sert ce chapitre ?

Ce dossier sert à :

- comprendre comment on passe d’un **score** à une **probabilité** ;
- voir pourquoi la **loss** doit changer ;
- implémenter un neurone binaire **from scratch** ;
- comparer cette logique à des versions plus pratiques ;
- préparer la suite : les **réseaux multicouches**.

---

## Prérequis

Si certaines bases te semblent floues, notamment :

- manipulation de tableaux ;
- pandas / numpy ;
- matrices ;
- statistiques descriptives ;
- logique générale de la régression ;

alors commence par :

- [python_stats_data_analysis_foundations](0_python_stats_data_analysis_foundations.ipynb)

Puis regarde :

- [1_manual.ipynb](1_manual.ipynb)

BP2 suppose que ces fondations sont déjà globalement comprises.

---

## Fil directeur

Le fil rouge reste le même que dans toute la série :

> transformer des informations en décision.

Dans BP1, cela passait par une somme pondérée.  
Dans BP2, cette somme pondérée devient la base d’un **neurone artificiel** grâce à l’ajout d’une **fonction d’activation**.

---

## Suite logique

La limite naturelle de ce chapitre est la suivante :

- un seul neurone reste limité ;
- certaines frontières de décision sont trop complexes ;
- il faut donc assembler **plusieurs neurones en couches**.

C’est exactement ce qui mènera vers le chapitre suivant.