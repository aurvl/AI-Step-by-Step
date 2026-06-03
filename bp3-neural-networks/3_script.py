import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from colorama import Fore


# =====================================
# UTILS
# =====================================
def printer(text, type="i"):
    print()
    if type=="i":
        print(Fore.WHITE, "[INFO] " + text, Fore.RESET)
    elif type=="r":
        print(Fore.GREEN, "[RESULT] " + text, Fore.RESET)
    elif type=="y":
        print(Fore.BLUE, "[NOTE] " + text, Fore.RESET)
    elif type=="x":
        print(Fore.RED, "[ATTENTION] " + text, Fore.RESET)
    print()


def plot_loss_acc_decision(X, y, loss_history, accuracy_history, model_or_predict,
                           device=None, grid_res=300, figsize=(20,6), cmap='plasma', prob_threshold=0.5,
                           feature_idx=(0, 1), baseline='zeros', class_names=None,
                           projection='raw', pca=None):
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
    if X_np.ndim != 2 or X_np.shape[1] < 2:
        raise ValueError("X must be 2D with at least 2 features")

    if projection == 'pca':
        if pca is None:
            pca = PCA(n_components=2, random_state=42)
            pca.fit(X_np)
        X_2d = pca.transform(X_np)

        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                             np.linspace(y_min, y_max, grid_res))
        grid_2d = np.c_[xx.ravel(), yy.ravel()]
        grid_full = pca.inverse_transform(grid_2d).astype(np.float32)

        scatter_x = X_2d[:, 0]
        scatter_y = X_2d[:, 1]
        xlab, ylab = 'PC1', 'PC2'
        title_suffix = ' (PCA)'
    elif projection == 'raw':
        f1, f2 = feature_idx
        if not (0 <= f1 < X_np.shape[1] and 0 <= f2 < X_np.shape[1]):
            raise ValueError(f"feature_idx out of range for X with {X_np.shape[1]} features")

        x_min, x_max = X_np[:, f1].min() - 0.5, X_np[:, f1].max() + 0.5
        y_min, y_max = X_np[:, f2].min() - 0.5, X_np[:, f2].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                             np.linspace(y_min, y_max, grid_res))
        grid_2d = np.c_[xx.ravel(), yy.ravel()]

        if baseline == 'zeros':
            base = np.zeros((X_np.shape[1],), dtype=np.float32)
        elif baseline == 'mean':
            base = X_np.mean(axis=0).astype(np.float32)
        else:
            raise ValueError("baseline must be 'zeros' or 'mean'")

        grid_full = np.tile(base, (grid_2d.shape[0], 1))
        grid_full[:, f1] = grid_2d[:, 0]
        grid_full[:, f2] = grid_2d[:, 1]

        scatter_x = X_np[:, f1]
        scatter_y = X_np[:, f2]
        xlab, ylab = f'x{f1 + 1}', f'x{f2 + 1}'
        title_suffix = ''
    else:
        raise ValueError("projection must be 'raw' or 'pca'")

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

    # For decision boundary: if model outputs multi-class logits, plot argmax regions.
    if isinstance(model_or_predict, torch.nn.Module):
        model = model_or_predict
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
            grid_t = torch.from_numpy(grid_full.astype(np.float32)).to(model_device)
            for i in range(0, grid_t.shape[0], batch):
                out = model(grid_t[i:i+batch])
                parts.append(out.detach().cpu())
        if was_training:
            model.train()

        out_t = torch.cat(parts, dim=0)
        if out_t.ndim == 1:
            out_t = out_t.unsqueeze(1)
        if out_t.shape[1] == 1:
            probs = torch.sigmoid(out_t).squeeze(1).numpy()
            Z = probs.reshape(xx.shape)
            cf = axs[2].contourf(xx, yy, Z, levels=50, cmap=cmap, alpha=0.8)
            axs[2].contour(xx, yy, Z, levels=[prob_threshold], colors='w', linewidths=2)
            cbar = plt.colorbar(cf, ax=axs[2], fraction=0.046, pad=0.04)
            cbar.set_label('P(class 1)')
        else:
            preds = torch.argmax(out_t, dim=1).numpy().astype(np.int32)
            Z = preds.reshape(xx.shape)
            levels = np.arange(out_t.shape[1] + 1) - 0.5
            cf = axs[2].contourf(xx, yy, Z, levels=levels, cmap=cmap, alpha=0.8)
            cbar = plt.colorbar(cf, ax=axs[2], fraction=0.046, pad=0.04, ticks=np.arange(out_t.shape[1]))
            if class_names is not None and len(class_names) == out_t.shape[1]:
                cbar.ax.set_yticklabels(class_names)
            cbar.set_label('Classe prédite')
    elif callable(model_or_predict):
        probs = _probs_from_callable(model_or_predict, grid_full)
        Z = probs.reshape(xx.shape)
        cf = axs[2].contourf(xx, yy, Z, levels=50, cmap=cmap, alpha=0.8)
        axs[2].contour(xx, yy, Z, levels=[prob_threshold], colors='w', linewidths=2)
        cbar = plt.colorbar(cf, ax=axs[2], fraction=0.046, pad=0.04)
        cbar.set_label('P(class 1)')
    else:
        raise ValueError("model_or_predict must be a torch.nn.Module or a callable")

    axs[2].scatter(scatter_x, scatter_y, c=y_np, cmap=cmap, linewidths=0.4, s=30)
    axs[2].set_xlim(x_min, x_max)
    axs[2].set_ylim(y_min, y_max)
    axs[2].set_title('Decision boundary' + title_suffix)
    axs[2].set_xlabel(xlab)
    axs[2].set_ylabel(ylab)
    plt.tight_layout()
    plt.show()
    return fig, axs


def plot_confusion_matrix(cm, class_names=None, normalize=False, cmap='Blues', figsize=(6,6), annot=True, fmt='.2f'):
    """
    Affiche une matrice de confusion.
    cm : array-like (K,K) — matrice de confusion (par ex sortie de sklearn.metrics.confusion_matrix)
    class_names : liste de K labels (optionnel)
    normalize : bool — si True, affiche les pourcentages par ligne
    """
    cm = np.array(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # éviter division par zéro
        row_sums[row_sums == 0] = 1
        cm_display = cm / row_sums
    else:
        cm_display = cm

    plt.figure(figsize=figsize)
    heatmap_fmt = fmt if normalize else 'd'
    sns.heatmap(cm_display, annot=annot, fmt=heatmap_fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names, cbar=True, square=True,
                linewidths=0.5, linecolor='gray')
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité terrain')
    title = 'Matrice de confusion (normalisée)' if normalize else 'Matrice de confusion'
    plt.title(title)
    plt.tight_layout()
    plt.show()

# =====================================
# DATASET
# =====================================
# !!! IMPORTANT : Modifier le chemin!!! 
path = r"C:\Users\aurel\Desktop\Projects\PUSH ON GIT\AI-Step-by-Step\bp3-neural-networks\dataset_ecommerce.csv"
df = pd.read_csv(path)
printer("Dataset succesfully loaded", type="r")

print("Dataset columns : ", df.columns.to_list())
# On distingue bien les la target : variable d'intéret
# CONTEXT : Le dataset porte sur un problème réel de détection du risque 
# sur des transactions e-commerce. Chaque ligne représente une transaction 
# effectuée par un utilisateur, avec des informations comportementales, 
# techniques et historiques.
print()
print("Shape du dataset :", df.shape)
# Nombre d’observations : 14 560
# Nombre de features : 13 variables numériques
# Target : classification en 3 classes
#       legit → transaction normale
#       suspicious → comportement atypique
#       fraud → forte probabilité de fraude

print()
print(df.info()) # inputs type: float / target: categorial
print(df.isna().sum()) # df contient des missing values
print(df.duplicated().sum()) # pas de duplicates

# =====================================
# VISUALIZATIONS
# =====================================
printer("Visualizations...")
# Style dark
plt.style.use("dark_background")

# 1) Distribution des classes
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="target", order=["legit", "suspicious", "fraud"])
plt.title("Distribution des classes")
plt.xlabel("Classe")
plt.ylabel("Nombre de transactions")
# plt.show()


# 2) Montant de transaction selon la classe
plt.figure(figsize=(8, 4))
sns.boxplot(
    data=df,
    x="target",
    y="transaction_amount",
    order=["legit", "suspicious", "fraud"],
    showfliers=False
)
plt.title("Montant des transactions selon la classe")
plt.xlabel("Classe")
plt.ylabel("Montant de transaction")
# plt.show()


# 3) Interaction entre risque IP et confiance device
sample_df = df.sample(3000, random_state=42)

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=sample_df,
    x="device_trust_score",
    y="ip_risk_score",
    hue="target",
    hue_order=["legit", "suspicious", "fraud"],
    alpha=0.65
)
plt.title("Interaction entre confiance device et risque IP")
plt.xlabel("Device trust score")
plt.ylabel("IP risk score")
plt.legend(title="Classe")
# plt.show()
plt.close()

# =====================================
# PREPROCESSING
# =====================================
printer("Preprocessing...")
X_brut = df.drop(columns=['target'])
y_brut = df['target']

order = ['legit', 'suspicious', 'fraud']
mapping = {lab: i for i, lab in enumerate(order)}
y_lab = y_brut.map(mapping).astype(int)  # 0,1,2 selon l'ordre voulu

X_trainnp, X_remain, y_trainnp, y_remain = train_test_split(
    X_brut, y_lab,
    test_size=0.30,
    random_state=42,
    stratify=y_lab
)

print("Train shape :", X_trainnp.shape)
print("Test shape  :", X_remain.shape)

print("\nRépartition de la cible :")
vals, counts = np.unique(y_trainnp, return_counts=True)
print(dict(zip(vals, counts / counts.sum())))

numeric_features = X_brut.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_brut.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
    ]
)

# Fit le preprocessor sur les training data et transform datasets
preprocessor.fit(X_trainnp)

# On transforme les training et remaining sets (output: numpy arrays)
X_trainnp_proc = preprocessor.transform(X_trainnp)
X_remain_proc = preprocessor.transform(X_remain)

# Optionnel : diviser X_remain_proc en validation et test (50/50 -> 15%/15% du total)
X_valnp, X_testnp, y_valnp, y_testnp = train_test_split(
    X_remain_proc, y_remain,
    test_size=0.5,
    random_state=42,
    stratify=y_remain
)

# Convertir en tensor pour notre reseau
X_train = torch.from_numpy(X_trainnp_proc).float()
X_val = torch.from_numpy(X_valnp).float()
X_test = torch.from_numpy(X_testnp).float()

# CrossEntropyLoss attend des labels entiers (0..C-1) en torch.int64 et shape (N,)
y_train = torch.from_numpy(y_trainnp.to_numpy().astype(np.int64)).long()
y_val = torch.from_numpy(y_valnp.to_numpy().astype(np.int64)).long()
y_test = torch.from_numpy(y_testnp.to_numpy().astype(np.int64)).long()

print("\nX_train shape:", X_train.shape)
print("X_val shape  :", X_val.shape)
print("X_test shape :", X_test.shape)
print()
print(f"y_train.shape = {y_train.shape}")
print(f"y_val.shape = {y_val.shape}")
print(f"y_test.shape = {y_test.shape}")

printer("Preprocessing succesfully did !", type="r")

# PCA (espace PC1/PC2) pour visualiser la frontière de décision
pca_2d = PCA(n_components=2, random_state=42)
pca_2d.fit(X_trainnp_proc)

# =====================================
# Model
# =====================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        """
        input_dim   : nombre de features d'entrée
        hidden_dims : liste contenant le nombre de neurones
                      dans chaque couche cachée
        output_dim  : dimension de sortie (1 pour binaire)
        """
        super().__init__()

        layers = []

        current_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ReLU())
            current_dim = h

        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = MLP(input_dim=X_train.shape[1], hidden_dims=[64, 64, 64], output_dim=3)

print(model)
printer("Model succesfully created !", type="r")


# =====================================
# Training
# =====================================
# Loss et optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.000015)

epochs = 20000
train_loss_history = []
train_accuracy_history = []
val_loss_history = []
val_accuracy_history = []

for epoch in tqdm(range(epochs)):
    # Forward
    model.train()
    y_hat = model(X_train)
    # Loss
    loss = criterion(y_hat, y_train)
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        # Train metrics (sklearn)
        with torch.no_grad():
            preds_train = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
            train_target = y_train.detach().cpu().numpy()
            acc = accuracy_score(train_target, preds_train)
        
        # Valid metrics
        model.eval()
        with torch.no_grad():
            y_val_hat = model(X_val)
            val_loss = criterion(y_val_hat, y_val)
            preds_val = torch.argmax(y_val_hat, dim=1).detach().cpu().numpy()
            val_target = y_val.detach().cpu().numpy()
            val_acc = accuracy_score(val_target, preds_val)
        
        # Suivi
        train_loss_history.append(loss.item())
        train_accuracy_history.append(acc)
        val_loss_history.append(val_loss.item())
        val_accuracy_history.append(val_acc)


# Résultat final
printer(f"Final loss : {train_loss_history[-1]:.3f}", "r")
plot_loss_acc_decision(
    X_train, y_train,
    train_loss_history, train_accuracy_history,
    model,
    class_names=['legit', 'suspicious', 'fraud'],
    projection='pca',
    pca=pca_2d,
)
plot_loss_acc_decision(
    X_val, y_val,
    val_loss_history, val_accuracy_history,
    model,
    class_names=['legit', 'suspicious', 'fraud'],
    projection='pca',
    pca=pca_2d,
)


# =====================================
# EVALUATION
# =====================================
printer("Evaluating the model (OOS)...")
model.eval()
with torch.no_grad():
    logits_test = model(X_test)
    test_loss = criterion(logits_test, y_test).item()
    pred_labels = torch.argmax(logits_test, dim=1).detach().cpu().numpy()
    target = y_test.detach().cpu().numpy()
    probas = torch.softmax(logits_test, dim=1).detach().cpu().numpy()

test_acc = accuracy_score(target, pred_labels)
test_roc = roc_auc_score(target, probas, multi_class='ovr', average='macro')
report = classification_report(
    target,
    pred_labels,
    labels=[0, 1, 2],
    target_names=['legit', 'suspicious', 'fraud'],
    zero_division=0,
)
cm = confusion_matrix(target, pred_labels, labels=[0, 1, 2])

printer(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f} | Test roc(ovr): {test_roc:.4f}", "r")
print(report)
plot_confusion_matrix(cm, class_names=['legit', 'suspicious', 'fraud'], normalize=True, fmt='.2%')


# fin.