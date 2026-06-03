library(dplyr)
library(readr)
library(caret)
library(pROC)

# ============================================================
# 0) OUTILS
# ============================================================

sigmoid <- function(z) {
  z <- pmax(pmin(z, 500), -500)
  1 / (1 + exp(-z))
}

log_loss <- function(y_true, y_proba, eps = 1e-12) {
  y_proba <- pmax(pmin(y_proba, 1 - eps), eps)
  -mean(y_true * log(y_proba) + (1 - y_true) * log(1 - y_proba))
}

fit_scaler <- function(X) {
  means <- apply(X, 2, mean)
  sds <- apply(X, 2, sd)
  sds[sds == 0] <- 1
  list(means = means, sds = sds)
}

transform_scaler <- function(X, scaler) {
  scale(X, center = scaler$means, scale = scaler$sds)
}

# ============================================================
# 1) NEURONE ARTIFICIEL
# ============================================================

fit_neuron <- function(X, y, learning_rate = 0.05, epochs = 3000, seed = 42, verbose = FALSE) {
  set.seed(seed)
  n <- nrow(X)
  p <- ncol(X)

  w <- rnorm(p, mean = 0, sd = 0.1)
  b <- 0
  losses <- numeric(epochs)

  for (epoch in 1:epochs) {
    z <- X %*% w + b
    y_proba <- sigmoid(z)

    losses[epoch] <- log_loss(y, y_proba)

    dz <- as.vector(y_proba) - y
    dw <- colMeans(X * dz)
    db <- mean(dz)

    w <- w - learning_rate * dw
    b <- b - learning_rate * db

    if (verbose && epoch %% 500 == 0) {
      cat(sprintf("Epoch %04d | Log-loss = %.4f\n", epoch, losses[epoch]))
    }
  }

  list(w = w, b = b, losses = losses)
}

predict_proba_neuron <- function(model, X) {
  z <- X %*% model$w + model$b
  as.vector(sigmoid(z))
}

predict_label_neuron <- function(model, X, threshold = 0.5) {
  ifelse(predict_proba_neuron(model, X) >= threshold, 1, 0)
}

# ============================================================
# 2) IMPORTER LE DATASET
# ============================================================

df <- read_csv("dataset_streaming_churn.csv", show_col_types = FALSE)

cat("\nDataset chargé :\n")
print(head(df))
cat("\nTaux de churn :", round(mean(df$target), 4), "\n")

# ============================================================
# 3) FEATURE ENGINEERING
# ============================================================

df <- df %>%
  mutate(
    engagement_score = 0.4 * watch_hours_week + 0.4 * sessions_week + 20 * content_completion_rate,
    friction_score = 1.5 * support_tickets_90d + 2.5 * payment_failures_6m + 0.08 * days_since_last_login,
    price_per_hour = monthly_price / (watch_hours_week + 1)
  )

# ============================================================
# 4) TRAIN / TEST SPLIT
# ============================================================

set.seed(42)
train_idx <- createDataPartition(df$target, p = 0.7, list = FALSE)

train_df <- df[train_idx, ]
test_df  <- df[-train_idx, ]

cat("\nTrain shape :", dim(train_df), "\n")
cat("Test shape  :", dim(test_df), "\n")

cat("\nRépartition de la cible (train) :\n")
print(prop.table(table(train_df$target)))

# ============================================================
# 5) PREPROCESSING
# ============================================================

# formule complète pour model.matrix
formula_all <- as.formula(target ~ .)

X_train <- model.matrix(formula_all, data = train_df)[, -1]
X_test  <- model.matrix(formula_all, data = test_df)[, -1]

y_train <- train_df$target
y_test  <- test_df$target

scaler <- fit_scaler(X_train)
X_train_scaled <- transform_scaler(X_train, scaler)
X_test_scaled  <- transform_scaler(X_test, scaler)

# ============================================================
# 6) CROSS-VALIDATION
# ============================================================

set.seed(42)
folds <- createFolds(y_train, k = 5, returnTrain = FALSE)

cv_scores <- c()

for (i in seq_along(folds)) {
  val_idx <- folds[[i]]
  tr_idx <- setdiff(seq_len(nrow(X_train_scaled)), val_idx)

  X_tr <- X_train_scaled[tr_idx, , drop = FALSE]
  y_tr <- y_train[tr_idx]

  X_val <- X_train_scaled[val_idx, , drop = FALSE]
  y_val <- y_train[val_idx]

  model_cv <- fit_neuron(X_tr, y_tr, learning_rate = 0.05, epochs = 3000, seed = 42)
  y_val_pred <- predict_label_neuron(model_cv, X_val)

  acc <- mean(y_val_pred == y_val)
  cv_scores <- c(cv_scores, acc)
}

cat("\n==============================\n")
cat("CROSS-VALIDATION (train set)\n")
cat("==============================\n")
cat("Scores par fold :", round(cv_scores, 4), "\n")
cat("Accuracy moyenne :", round(mean(cv_scores), 4), "\n")
cat("Ecart-type       :", round(sd(cv_scores), 4), "\n")

# ============================================================
# 7) ENTRAINEMENT FINAL
# ============================================================

model <- fit_neuron(X_train_scaled, y_train, learning_rate = 0.05, epochs = 3000, seed = 42, verbose = TRUE)

# ============================================================
# 8) PREDICTIONS + METRICS
# ============================================================

y_proba <- predict_proba_neuron(model, X_test_scaled)
y_pred <- predict_label_neuron(model, X_test_scaled)

cat("\n==============================\n")
cat("EVALUATION NEURONE ARTIFICIEL\n")
cat("==============================\n")
cat("Accuracy :", round(mean(y_pred == y_test), 4), "\n")

roc_obj <- roc(y_test, y_proba, quiet = TRUE)
cat("ROC AUC  :", round(as.numeric(auc(roc_obj)), 4), "\n")

cat("\nConfusion matrix :\n")
print(table(Predicted = y_pred, Actual = y_test))

# métriques simples classe 1
tp <- sum(y_pred == 1 & y_test == 1)
tn <- sum(y_pred == 0 & y_test == 0)
fp <- sum(y_pred == 1 & y_test == 0)
fn <- sum(y_pred == 0 & y_test == 1)

precision <- tp / (tp + fp + 1e-12)
recall <- tp / (tp + fn + 1e-12)
f1 <- 2 * precision * recall / (precision + recall + 1e-12)

cat("\nClasse 1 :\n")
cat("Precision :", round(precision, 4), "\n")
cat("Recall    :", round(recall, 4), "\n")
cat("F1-score  :", round(f1, 4), "\n")