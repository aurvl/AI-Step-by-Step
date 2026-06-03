import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

# Générer et remplir df (pandas)
def create_data(n=100, seed=123):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "retard_depart": rng.uniform(0, 60, n),
        "meteo": rng.integers(0, 4, n),
        "heure_depart": rng.uniform(0, 23, n),
        "escales": rng.integers(0, 3, n),
    })
    df["retard_final"] = (
        5
        + 0.7 * df["retard_depart"]
        + 4.0 * df["meteo"]
        + 0.3 * df["heure_depart"]
        + 8.0 * df["escales"]
        + rng.normal(0, 5, n)
    )
    return df.drop(columns=['retard_final']), df['retard_final']


def create_class_dataset(n=800, n_features=5, positive_rate=0.51, test_size=0.2, seed=123, class_sep=1.0, noise_scale=1.0, feature_noise=0.0):
    """
    Génère un jeu de données de classification binaire.

    - X: matrice (n, n_features)
    - y: vecteur binaire avec proportion de 1 ~ positive_rate

    Retourne: X_train, y_train, X_test, y_test (NumPy arrays)
    """
    rng = np.random.default_rng(seed)

    # Générer X standard normal
    X = rng.normal(0, 1, size=(n, n_features))

    # Ajouter bruit sur les features si demandé
    if feature_noise > 0:
        X = X + rng.normal(0, feature_noise, size=X.shape)

    # Générer coefficients aléatoires ; appliquer `class_sep` pour contrôler la séparabilité
    w = rng.normal(0, 1, size=(n_features,)) * class_sep
    # bruit appliqué aux logits (contrôlé par noise_scale)
    noise = rng.normal(0, 1, size=n) * noise_scale
    logits_base = X @ w + noise

    # Trouver offset b tel que mean(sigmoid(logits_base + b)) ~= positive_rate
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    # Bisection search on b
    lo, hi = -50.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p = sigmoid(logits_base + mid).mean()
        if p > positive_rate:
            hi = mid
        else:
            lo = mid
    b = 0.5 * (lo + hi)

    probs = sigmoid(logits_base + b)
    y = rng.random(n) < probs
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    return X_train, y_train, X_test, y_test

def create_moons_dataset(
    n_samples=500,
    noise=0.12,
    rotate_deg=-12,
    scale_x=1.35,
    scale_y=0.95,
    shift_x=0.15,
    shift_y=-0.05,
    random_state=42,
    as_dataframe=True
):
    """
    Génère un dataset binaire de type 'two moons', proche visuellement
    de l'image fournie.

    Paramètres
    ----------
    n_samples : int
        Nombre total de points.
    noise : float
        Bruit ajouté aux lunes.
    rotate_deg : float
        Angle de rotation du dataset en degrés.
    scale_x, scale_y : float
        Facteurs d'étirement sur x et y.
    shift_x, shift_y : float
        Décalage global du dataset.
    random_state : int
        Graine aléatoire.
    as_dataframe : bool
        Si True, retourne un DataFrame, sinon retourne (X, y).

    Retour
    ------
    DataFrame avec colonnes ['x1', 'x2', 'target']
    ou bien (X, y)
    """
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )

    # Étirement
    X[:, 0] *= scale_x
    X[:, 1] *= scale_y

    # Rotation
    theta = np.deg2rad(rotate_deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    X = X @ R.T

    # Translation
    X[:, 0] += shift_x
    X[:, 1] += shift_y

    if as_dataframe:
        return pd.DataFrame({
            "x1": X[:, 0],
            "x2": X[:, 1],
            "target": y
        })
    return X, y

def plot_decision_boundary(X, y, w, b, model_predict_proba, title="Frontière de décision"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    proba = model_predict_proba(grid, w, b)
    proba = np.asarray(proba).ravel()
    Z = (proba >= 0.5).astype(int).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.35, cmap="coolwarm")
    plt.contour(xx, yy, proba.reshape(xx.shape), 
                levels=[0.5], colors="white", linewidths=2)
    plt.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolors="white",
        linewidths=0.5,
        alpha=0.85
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.show()


def plot_loss_acc_decision(X, y, loss_history, accuracy_history, model_or_predict,
                           device=None, grid_res=300, figsize=(20,6), cmap='plasma', prob_threshold=0.5):
    """
    Plot (1x3): Loss, Accuracy, Decision boundary (with data points).
    model_or_predict: torch.nn.Module OR callable that accepts (n,2) numpy array and returns
                      probabilities/logits or predict_proba outputs.
    """
    # --- to numpy
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy().ravel()
    else:
        y_np = np.asarray(y).ravel()

    loss_arr = np.asarray(loss_history)
    acc_arr = np.asarray(accuracy_history)

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Loss
    if loss_arr.size:
        axs[0].plot(np.arange(1, len(loss_arr) + 1), loss_arr, marker='o' if len(loss_arr) <= 50 else None)
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].grid(False)

    # Accuracy
    if acc_arr.size:
        axs[1].plot(np.arange(1, len(acc_arr) + 1), acc_arr, marker='o' if len(acc_arr) <= 50 else None)
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid(False)

    # Decision boundary grid
    x_min, x_max = X_np[:,0].min() - 0.5, X_np[:,0].max() + 0.5
    y_min, y_max = X_np[:,1].min() - 0.5, X_np[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                         np.linspace(y_min, y_max, grid_res))
    grid = np.c_[xx.ravel(), yy.ravel()]

    def _probs_from_torch_model(model, grid_np):
        try:
            param = next(model.parameters())
            model_device = param.device
        except StopIteration:
            model_device = torch.device('cpu' if device is None else device)
        was_training = model.training
        model.eval()
        batch = 16384
        parts = []
        with torch.no_grad():
            grid_t = torch.from_numpy(grid_np.astype(np.float32)).to(model_device)
            for i in range(0, grid_t.shape[0], batch):
                out = model(grid_t[i:i+batch])
                parts.append(out.detach().cpu().numpy())
        if was_training:
            model.train()
        probs = np.vstack(parts)
        # reduce to one-d array of class-1 probabilities
        if probs.ndim > 1:
            if probs.shape[1] == 1:
                probs = probs[:,0]
            elif probs.shape[1] == 2:
                # if logits, apply softmax; else assume second column is prob
                if probs.min() < 0 or probs.max() > 1:
                    e = np.exp(probs - probs.max(axis=1, keepdims=True))
                    probs = (e / e.sum(axis=1, keepdims=True))[:,1]
                else:
                    probs = probs[:,1]
            else:
                e = np.exp(probs - probs.max(axis=1, keepdims=True))
                probs = (e / e.sum(axis=1, keepdims=True))[:,1] if probs.shape[1] > 1 else probs[:,0]
        else:
            probs = probs.ravel()
        if probs.min() < 0 or probs.max() > 1:
            probs = 1.0 / (1.0 + np.exp(-probs))
        return probs

    def _probs_from_callable(fn, grid_np):
        batch = 16384
        out_parts = []
        for i in range(0, grid_np.shape[0], batch):
            out = np.asarray(fn(grid_np[i:i+batch]))
            if out.ndim > 1:
                if out.shape[1] == 1:
                    out = out[:,0]
                elif out.shape[1] == 2:
                    out = out[:,1]
                else:
                    if out.min() < 0 or out.max() > 1:
                        e = np.exp(out - out.max(axis=1, keepdims=True))
                        out = (e / e.sum(axis=1, keepdims=True))[:,1]
                    else:
                        out = out[:,1] if out.shape[1] > 1 else out[:,0]
            out_parts.append(out.ravel())
        probs = np.concatenate(out_parts)
        if probs.min() < 0 or probs.max() > 1:
            probs = 1.0 / (1.0 + np.exp(-probs))
        return probs

    if isinstance(model_or_predict, torch.nn.Module):
        probs = _probs_from_torch_model(model_or_predict, grid)
    elif callable(model_or_predict):
        probs = _probs_from_callable(model_or_predict, grid)
    else:
        raise ValueError("model_or_predict must be a torch.nn.Module or a callable")

    Z = probs.reshape(xx.shape)

    cf = axs[2].contourf(xx, yy, Z, levels=50, cmap=cmap, alpha=0.8)
    axs[2].contour(xx, yy, Z, levels=[prob_threshold], colors='w', linewidths=2)
    axs[2].scatter(X_np[:,0], X_np[:,1], c=y_np, cmap=cmap, linewidths=0.4, s=30)
    axs[2].set_xlim(x_min, x_max)
    axs[2].set_ylim(y_min, y_max)
    axs[2].set_title('Decision boundary')
    axs[2].set_xlabel('x1')
    axs[2].set_ylabel('x2')
    plt.colorbar(cf, ax=axs[2], fraction=0.046, pad=0.04, label='P(class 1)')
    plt.tight_layout()
    plt.show()
    return fig, axs