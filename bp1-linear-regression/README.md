# **Deciding with Numbers / Décider avec des chiffres**

Fichiers :

* [Python](script_py.py)
* [R](script.R)
* [Julia](script.jl)

Ce qu’ils font :

* chargent `dataset_vols.csv`
* utilisent tes 25 variables et la cible `retard_arrivee_min`
* proposent 3 stratégies :

  * `proper` : nettoyage propre, imputation, one-hot, normalisation
  * `minimal` : nettoyage minimal, drop des NaN, one-hot
  * `raw` : version volontairement dégradée pour montrer l’impact de la qualité de l’info
* affichent :

  * nombre de lignes utilisées
  * nombre de variables utilisées
  * décisions de préparation
  * MAE
  * RMSE
  * R²

Commandes typiques :

```bash
python script_py.py --data dataset_vols.csv --strategy proper
```

```bash
Rscript script.R dataset_vols.csv proper
```

```bash
julia script.jl dataset_vols.csv proper
```
