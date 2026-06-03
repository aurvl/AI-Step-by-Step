using CSV
using DataFrames
using Random
using Statistics
using StatsBase
using LinearAlgebra

# ============================================================
# 0) OUTILS
# ============================================================

sigmoid(z) = 1.0 ./ (1.0 .+ exp.(-clamp.(z, -500, 500)))

function log_loss(y_true, y_proba; eps=1e-12)
    y_proba = clamp.(y_proba, eps, 1 - eps)
    return -mean(y_true .* log.(y_proba) .+ (1 .- y_true) .* log.(1 .- y_proba))
end

function fit_scaler(X::Matrix{Float64})
    means = vec(mean(X, dims=1))
    stds = vec(std(X, dims=1))
    stds[stds .== 0.0] .= 1.0
    return means, stds
end

function transform_scaler(X::Matrix{Float64}, means::Vector{Float64}, stds::Vector{Float64})
    return (X .- means') ./ stds'
end

function train_test_split_stratified(df::DataFrame, target_col::Symbol; test_size=0.3, seed=42)
    Random.seed!(seed)
    idx0 = findall(df[!, target_col] .== 0)
    idx1 = findall(df[!, target_col] .== 1)

    shuffle!(idx0)
    shuffle!(idx1)

    n0_test = round(Int, length(idx0) * test_size)
    n1_test = round(Int, length(idx1) * test_size)

    test_idx = vcat(idx0[1:n0_test], idx1[1:n1_test])
    train_idx = setdiff(1:nrow(df), test_idx)

    return df[train_idx, :], df[test_idx, :]
end

function one_hot_encode(train_df::DataFrame, test_df::DataFrame, categorical_cols::Vector{Symbol})
    train_enc = copy(train_df)
    test_enc = copy(test_df)

    for col in categorical_cols
        levels = unique(train_df[!, col])

        for lev in levels
            newname = Symbol(string(col) * "_" * string(lev))
            train_enc[!, newname] = Int.(train_df[!, col] .== lev)
            test_enc[!, newname]  = Int.(test_df[!, col] .== lev)
        end

        select!(train_enc, Not(col))
        select!(test_enc, Not(col))
    end

    return train_enc, test_enc
end

# ============================================================
# 1) NEURONE ARTIFICIEL
# ============================================================

mutable struct ArtificialNeuron
    w::Vector{Float64}
    b::Float64
    loss_history::Vector{Float64}
end

function fit_neuron(X::Matrix{Float64}, y::Vector{Float64}; learning_rate=0.05, epochs=3000, seed=42, verbose=false)
    Random.seed!(seed)
    n_samples, n_features = size(X)

    w = 0.1 .* randn(n_features)
    b = 0.0
    losses = Float64[]

    for epoch in 1:epochs
        z = X * w .+ b
        y_proba = sigmoid(z)

        loss = log_loss(y, y_proba)
        push!(losses, loss)

        dz = y_proba .- y
        dw = vec((transpose(X) * dz) ./ n_samples)
        db = mean(dz)

        w .-= learning_rate .* dw
        b -= learning_rate * db

        if verbose && epoch % 500 == 0
            println("Epoch $(lpad(epoch, 4, '0')) | Log-loss = $(round(loss, digits=4))")
        end
    end

    return ArtificialNeuron(w, b, losses)
end

function predict_proba(model::ArtificialNeuron, X::Matrix{Float64})
    z = X * model.w .+ model.b
    return sigmoid(z)
end

function predict_label(model::ArtificialNeuron, X::Matrix{Float64}; threshold=0.5)
    y_proba = predict_proba(model, X)
    return Int.(y_proba .>= threshold)
end

function accuracy_score(y_true::Vector{Int}, y_pred::Vector{Int})
    return mean(y_true .== y_pred)
end

function roc_auc_score_manual(y_true::Vector{Int}, y_score::Vector{Float64})
    pos = y_score[y_true .== 1]
    neg = y_score[y_true .== 0]
    total = length(pos) * length(neg)

    s = 0.0
    for p in pos, n in neg
        if p > n
            s += 1
        elseif p == n
            s += 0.5
        end
    end
    return s / total
end

function classification_metrics(y_true::Vector{Int}, y_pred::Vector{Int})
    tp = sum((y_pred .== 1) .& (y_true .== 1))
    tn = sum((y_pred .== 0) .& (y_true .== 0))
    fp = sum((y_pred .== 1) .& (y_true .== 0))
    fn = sum((y_pred .== 0) .& (y_true .== 1))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return Dict(
        "tp" => tp, "tn" => tn, "fp" => fp, "fn" => fn,
        "precision" => precision, "recall" => recall, "f1" => f1
    )
end

# ============================================================
# 2) IMPORTER LE DATASET
# ============================================================

df = CSV.read("dataset_streaming_churn.csv", DataFrame)

println("\nDataset chargé :")
println(first(df, 5))
println("\nTaux de churn : ", round(mean(df.target), digits=4))

# ============================================================
# 3) FEATURE ENGINEERING
# ============================================================

df.engagement_score = 0.4 .* df.watch_hours_week .+ 0.4 .* df.sessions_week .+ 20 .* df.content_completion_rate
df.friction_score = 1.5 .* df.support_tickets_90d .+ 2.5 .* df.payment_failures_6m .+ 0.08 .* df.days_since_last_login
df.price_per_hour = df.monthly_price ./ (df.watch_hours_week .+ 1)

# ============================================================
# 4) TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = train_test_split_stratified(df, :target; test_size=0.3, seed=42)

println("\nTrain shape : ", size(train_df))
println("Test shape  : ", size(test_df))

println("\nRépartition de la cible (train) :")
counts = countmap(train_df.target)
total = sum(values(counts))
println(Dict(k => v / total for (k, v) in counts))

# ============================================================
# 5) PREPROCESSING
# ============================================================

categorical_cols = Symbol[]
numeric_cols = Symbol[]

for c in names(train_df)
    if c == :target
        continue
    elseif eltype(train_df[!, c]) <: AbstractString
        push!(categorical_cols, c)
    else
        push!(numeric_cols, c)
    end
end

train_enc, test_enc = one_hot_encode(train_df, test_df, categorical_cols)

X_train_df = select(train_enc, Not(:target))
X_test_df = select(test_enc, Not(:target))

X_train = Matrix{Float64}(X_train_df)
X_test = Matrix{Float64}(X_test_df)

y_train = Float64.(train_enc.target)
y_test = Float64.(test_enc.target)

means, stds = fit_scaler(X_train)
X_train = transform_scaler(X_train, means, stds)
X_test = transform_scaler(X_test, means, stds)

# ============================================================
# 6) CROSS-VALIDATION
# ============================================================

Random.seed!(42)
n = size(X_train, 1)
idx = shuffle(1:n)

fold_size = div(n, 5)
cv_scores = Float64[]

for fold in 1:5
    start_idx = (fold - 1) * fold_size + 1
    end_idx = fold == 5 ? n : fold * fold_size

    val_idx = idx[start_idx:end_idx]
    tr_idx = setdiff(1:n, val_idx)

    X_tr = X_train[tr_idx, :]
    y_tr = y_train[tr_idx]

    X_val = X_train[val_idx, :]
    y_val = Int.(y_train[val_idx])

    model_cv = fit_neuron(X_tr, y_tr; learning_rate=0.05, epochs=3000, seed=42, verbose=false)
    y_val_pred = predict_label(model_cv, X_val)

    push!(cv_scores, accuracy_score(y_val, y_val_pred))
end

println("\n==============================")
println("CROSS-VALIDATION (train set)")
println("==============================")
println("Scores par fold : ", round.(cv_scores, digits=4))
println("Accuracy moyenne : ", round(mean(cv_scores), digits=4))
println("Ecart-type       : ", round(std(cv_scores), digits=4))

# ============================================================
# 7) ENTRAINEMENT FINAL
# ============================================================

model = fit_neuron(X_train, y_train; learning_rate=0.05, epochs=3000, seed=42, verbose=true)

# ============================================================
# 8) PREDICTIONS + METRICS
# ============================================================

y_proba = predict_proba(model, X_test)
y_pred = predict_label(model, X_test)

println("\n==============================")
println("EVALUATION NEURONE ARTIFICIEL")
println("==============================")
println("Accuracy : ", round(accuracy_score(Int.(y_test), y_pred), digits=4))
println("ROC AUC  : ", round(roc_auc_score_manual(Int.(y_test), y_proba), digits=4))

metrics = classification_metrics(Int.(y_test), y_pred)
println("\nConfusion matrix :")
println([metrics["tn"] metrics["fp"]; metrics["fn"] metrics["tp"]])

println("\nClasse 1 :")
println("Precision : ", round(metrics["precision"], digits=4))
println("Recall    : ", round(metrics["recall"], digits=4))
println("F1-score  : ", round(metrics["f1"], digits=4))