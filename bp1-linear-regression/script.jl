# Régression linéaire professionnelle sur dataset_vols.csv
#
# Usage:
#   julia script.jl dataset_vols.csv proper
#   julia script.jl dataset_vols.csv minimal
#   julia script.jl dataset_vols.csv raw

using CSV
using DataFrames
using Statistics
using CategoricalArrays
using GLM
using StatsModels
using Random

const TARGET_COL = :retard_arrivee_min

const CATEGORICAL_COLS = [
    :compagnie, :aeroport_depart, :aeroport_arrivee, :type_avion, :jour_semaine
]

const BINARY_COLS = [
    :vol_international, :hub_depart, :greve_sol, :changement_equipage
]

const NUMERIC_COLS = [
    :retard_depart_min, :meteo_score, :heure_depart, :nb_escales, :charge_reseau,
    :distance_km, :vent_kmh, :precipitation_mm, :temperature_c, :congestion_aeroport,
    :experience_captain_ans, :maintenance_score, :ponctualite_route_hist,
    :age_avion_ans, :rotations_24h, :passagers
]

function mae(y_true, y_pred)
    mean(abs.(y_true .- y_pred))
end

function rmse(y_true, y_pred)
    sqrt(mean((y_true .- y_pred).^2))
end

function r2(y_true, y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    1 - ss_res / ss_tot
end

function drop_duplicates(df::DataFrame)
    unique(df)
end

function impute_median!(df::DataFrame, cols)
    for c in cols
        vals = skipmissing(df[!, c])
        med = median(collect(vals))
        replace!(df[!, c], missing => med)
    end
end

function impute_mode_cat!(df::DataFrame, cols)
    for c in cols
        vals = collect(skipmissing(df[!, c]))
        mode_val = isempty(vals) ? "Unknown" : sort(collect(countmap(vals)), by = x -> x[2], rev = true)[1][1]
        replace!(df[!, c], missing => mode_val)
    end
end

function impute_zero!(df::DataFrame, cols)
    for c in cols
        replace!(df[!, c], missing => 0)
    end
end

function impute_unknown!(df::DataFrame, cols)
    for c in cols
        replace!(df[!, c], missing => "Unknown")
    end
end

function standardize!(df::DataFrame, cols)
    for c in cols
        μ = mean(df[!, c])
        σ = std(df[!, c])
        if σ > 0
            df[!, c] = (df[!, c] .- μ) ./ σ
        end
    end
end

function prepare_formula(features)
    term(TARGET_COL) ~ sum(term.(features))
end

function build_experiment(df::DataFrame, strategy::String = "proper")
    data = copy(df)
    all_features = [c for c in names(data) if c != TARGET_COL]
    decisions = String[]

    if strategy == "proper"
        data = drop_duplicates(data)
        push!(decisions, "Suppression des doublons")
        push!(decisions, "Imputation des valeurs manquantes")
        push!(decisions, "Encodage catégoriel via formula/contrasts")
        push!(decisions, "Normalisation des variables numériques")

        selected_features = all_features
        impute_median!(data, NUMERIC_COLS)
        impute_mode_cat!(data, CATEGORICAL_COLS)
        standardize!(data, NUMERIC_COLS)

        for c in CATEGORICAL_COLS
            data[!, c] = categorical(data[!, c])
        end

    elseif strategy == "minimal"
        data = drop_duplicates(data)
        data = dropmissing(data)
        push!(decisions, "Suppression des doublons")
        push!(decisions, "Suppression des lignes contenant des NaN")
        push!(decisions, "Encodage catégoriel via formula/contrasts")
        push!(decisions, "Pas de normalisation")

        selected_features = all_features
        for c in CATEGORICAL_COLS
            data[!, c] = categorical(data[!, c])
        end

    elseif strategy == "raw"
        push!(decisions, "Conservation des doublons")
        push!(decisions, "Imputation grossière des NaN (0 / 'Unknown')")
        push!(decisions, "Encodage pauvre des catégories")
        push!(decisions, "Suppression volontaire de plusieurs variables très informatives")

        dropped_features = [
            :retard_depart_min,
            :charge_reseau,
            :congestion_aeroport,
            :ponctualite_route_hist,
            :maintenance_score,
            :vent_kmh,
            :precipitation_mm,
            :temperature_c
        ]
        selected_features = [c for c in all_features if !(c in dropped_features)]

        num_raw = [c for c in NUMERIC_COLS if c in selected_features]
        cat_raw = [c for c in CATEGORICAL_COLS if c in selected_features]

        impute_zero!(data, num_raw)
        impute_unknown!(data, cat_raw)

        # Encodage ordinal simple
        for c in cat_raw
            data[!, c] = levelcode.(categorical(data[!, c]))
        end

    else
        error("strategy doit être 'proper', 'minimal' ou 'raw'.")
    end

    Random.seed!(42)
    n = nrow(data)
    idx = shuffle(1:n)
    ntrain = floor(Int, 0.8 * n)
    train_idx = idx[1:ntrain]
    test_idx = idx[(ntrain + 1):end]

    train_df = data[train_idx, [selected_features; TARGET_COL]]
    test_df = data[test_idx, [selected_features; TARGET_COL]]

    f = prepare_formula(selected_features)
    model = lm(f, train_df)

    y_pred = predict(model, test_df)
    y_true = test_df[!, TARGET_COL]

    return Dict(
        "strategy" => strategy,
        "n_rows_used" => nrow(data),
        "n_features_used" => length(selected_features),
        "decisions" => decisions,
        "mae" => mae(y_true, y_pred),
        "rmse" => rmse(y_true, y_pred),
        "r2" => r2(y_true, y_pred),
    )
end

function print_results(res)
    labels = Dict(
        "proper" => "contexte riche et propre",
        "minimal" => "nettoyage minimal",
        "raw" => "contexte pauvre"
    )

    println("\n=== Approche : $(labels[res["strategy"]]) ===")
    println("Décisions de préparation :")
    for d in res["decisions"]
        println("  - $d")
    end

    println("Lignes utilisées     : $(res["n_rows_used"])")
    println("Variables utilisées  : $(res["n_features_used"])")
    println("MAE                  : $(round(res["mae"], digits=3))")
    println("RMSE                 : $(round(res["rmse"], digits=3))")
    println("R²                   : $(round(res["r2"], digits=4))")
end

data_path = length(ARGS) >= 1 ? ARGS[1] : "dataset_vols.csv"
strategy = length(ARGS) >= 2 ? ARGS[2] : "proper"

if !isfile(data_path)
    error("Fichier introuvable : $data_path")
end

df = CSV.read(data_path, DataFrame)
if !(TARGET_COL in names(df))
    error("La cible $(TARGET_COL) est absente du dataset.")
end

res = build_experiment(df, strategy)
print_results(res)