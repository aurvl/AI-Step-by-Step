#!/usr/bin/env Rscript

# Régression linéaire professionnelle sur dataset_vols.csv
#
# Usage:
#   Rscript script.R dataset_vols.csv proper
#   Rscript script.R dataset_vols.csv minimal
#   Rscript script.R dataset_vols.csv raw

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(recipes)
  library(parsnip)
  library(workflows)
  library(rsample)
  library(yardstick)
})

args <- commandArgs(trailingOnly = TRUE)
data_path <- ifelse(length(args) >= 1, args[1], "dataset_vols.csv")
strategy  <- ifelse(length(args) >= 2, args[2], "proper")

target_col <- "retard_arrivee_min"

categorical_cols <- c(
  "compagnie", "aeroport_depart", "aeroport_arrivee",
  "type_avion", "jour_semaine"
)

binary_cols <- c(
  "vol_international", "hub_depart", "greve_sol", "changement_equipage"
)

numeric_cols <- c(
  "retard_depart_min", "meteo_score", "heure_depart", "nb_escales",
  "charge_reseau", "distance_km", "vent_kmh", "precipitation_mm",
  "temperature_c", "congestion_aeroport", "experience_captain_ans",
  "maintenance_score", "ponctualite_route_hist", "age_avion_ans",
  "rotations_24h", "passagers"
)

if (!file.exists(data_path)) {
  stop(paste("Fichier introuvable :", normalizePath(data_path, mustWork = FALSE)))
}

df <- read_csv(data_path, show_col_types = FALSE)

if (!(target_col %in% names(df))) {
  stop(paste("La cible", target_col, "est absente du dataset."))
}

build_experiment <- function(df, strategy = "proper") {
  data <- df
  all_features <- setdiff(names(data), target_col)
  decisions <- c()

  if (strategy == "proper") {
    data <- distinct(data)
    decisions <- c(
      "Suppression des doublons",
      "Imputation des valeurs manquantes",
      "One-hot encoding des variables catégorielles",
      "Normalisation des variables numériques"
    )
    selected_features <- all_features

    rec <- recipe(as.formula(paste(target_col, "~ .")), data = data[, c(selected_features, target_col)]) %>%
      step_impute_median(all_numeric_predictors()) %>%
      step_impute_mode(all_nominal_predictors()) %>%
      step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
      step_normalize(all_numeric_predictors())

  } else if (strategy == "minimal") {
    data <- distinct(data) %>% tidyr::drop_na()
    decisions <- c(
      "Suppression des doublons",
      "Suppression des lignes contenant des NaN",
      "One-hot encoding des variables catégorielles",
      "Pas de normalisation"
    )
    selected_features <- all_features

    rec <- recipe(as.formula(paste(target_col, "~ .")), data = data[, c(selected_features, target_col)]) %>%
      step_dummy(all_nominal_predictors(), one_hot = TRUE)

  } else if (strategy == "raw") {
    decisions <- c(
      "Conservation des doublons",
      "Imputation grossière des NaN",
      "Encodage ordinal simple des catégories",
      "Suppression volontaire de plusieurs variables très informatives"
    )

    dropped_features <- c(
      "retard_depart_min",
      "charge_reseau",
      "congestion_aeroport",
      "ponctualite_route_hist",
      "maintenance_score",
      "vent_kmh",
      "precipitation_mm",
      "temperature_c"
    )

    selected_features <- setdiff(all_features, dropped_features)

    data2 <- data[, c(selected_features, target_col)]

    # Imputation grossière
    for (col in names(data2)) {
      if (col == target_col) next
      if (is.character(data2[[col]])) {
        data2[[col]][is.na(data2[[col]])] <- "Unknown"
        data2[[col]] <- as.integer(as.factor(data2[[col]]))
      } else {
        data2[[col]][is.na(data2[[col]])] <- 0
      }
    }

    split <- initial_split(data2, prop = 0.8)
    train_data <- training(split)
    test_data <- testing(split)

    model_spec <- linear_reg(penalty = 1, mixture = 0) %>% set_engine("glmnet")

    wf <- workflow() %>%
      add_model(model_spec) %>%
      add_formula(as.formula(paste(target_col, "~ .")))

    fitted <- fit(wf, data = train_data)
    preds <- predict(fitted, new_data = test_data) %>%
      bind_cols(test_data %>% select(all_of(target_col)))

    mae_val <- mae(preds, truth = all_of(target_col), estimate = .pred)$.estimate
    rmse_val <- rmse(preds, truth = all_of(target_col), estimate = .pred)$.estimate
    r2_val <- rsq(preds, truth = all_of(target_col), estimate = .pred)$.estimate

    return(list(
      strategy = strategy,
      n_rows_used = nrow(data2),
      n_features_used = length(selected_features),
      decisions = decisions,
      mae = mae_val,
      rmse = rmse_val,
      r2 = r2_val
    ))
  } else {
    stop("strategy doit être 'proper', 'minimal' ou 'raw'.")
  }

  split <- initial_split(data[, c(selected_features, target_col)], prop = 0.8)
  train_data <- training(split)
  test_data <- testing(split)

  model_spec <- linear_reg(penalty = 1, mixture = 0) %>% set_engine("glmnet")

  wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(model_spec)

  fitted <- fit(wf, data = train_data)
  preds <- predict(fitted, new_data = test_data) %>%
    bind_cols(test_data %>% select(all_of(target_col)))

  mae_val <- mae(preds, truth = all_of(target_col), estimate = .pred)$.estimate
  rmse_val <- rmse(preds, truth = all_of(target_col), estimate = .pred)$.estimate
  r2_val <- rsq(preds, truth = all_of(target_col), estimate = .pred)$.estimate

  list(
    strategy = strategy,
    n_rows_used = nrow(data),
    n_features_used = length(selected_features),
    decisions = decisions,
    mae = mae_val,
    rmse = rmse_val,
    r2 = r2_val
  )
}

print_results <- function(res) {
  labels <- c(
    proper = "contexte riche et propre",
    minimal = "nettoyage minimal",
    raw = "contexte pauvre"
  )

  cat("\n=== Approche :", labels[[res$strategy]], "===\n")
  cat("Décisions de préparation :\n")
  for (d in res$decisions) {
    cat("  -", d, "\n")
  }

  cat("Lignes utilisées     :", format(res$n_rows_used, big.mark = ","), "\n")
  cat("Variables utilisées  :", res$n_features_used, "\n")
  cat(sprintf("MAE                  : %.3f\n", res$mae))
  cat(sprintf("RMSE                 : %.3f\n", res$rmse))
  cat(sprintf("R²                   : %.4f\n", res$r2))
}

res <- build_experiment(df, strategy = strategy)
print_results(res)