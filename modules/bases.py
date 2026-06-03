import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score

#=============================================================
# SIMPLE PERCEPTRON
#=============================================================
def initialization(features):
    """
    Ici on crée une fonction qui va prendre en entrée le nombre de variables (features)
    et qui va retourner des poids et un biais initialisés aléatoirement
    """
    random.seed(42)
    np.random.seed(42)
    
    # Initialize weights and bias
    w = np.random.randn(features) # génère un vecteur de poids aléatoires de la taille du nombre de features
    b = np.random.randn()         # génère un nombre aléatoire pour le biais
    return w, b

def linear_equation(X, w, b):
    """
    Cette fonction prend en entrée un vecteur d'entrée X, un vecteur de poids w et un biais b
    et qui retourne la prédiction du perceptron : y_hat = w·X + b
    """
    X = np.asarray(X)
    w = np.asarray(w)
    return X @ w + b

def loss_function(y_reel, y_pred):
    """
    Cette fonction prend en entrée les valeurs réelles et prédites et qui retourne la MSE
    """
    return np.mean((y_reel - y_pred) ** 2)

def lin_gradients(X, y_reel, y_pred):
    """
    Cette fonction calcule les gradients de la MSE par rapport aux poids et au biais
    1) dL_dy_pred : gradient de la MSE par rapport à la prédiction
        => 2 * (y_pred - y_reel) / n
    2) dw : gradient de la MSE par rapport aux poids
        => dL_dy_pred * X
    3) db : gradient de la MSE par rapport au biais
        => somme de dL_dy_pred
     La fonction retourne dw et db
     (les gradients sont utilisés pour ajuster les poids et le biais dans la direction qui réduit la MSE)
    """
    n = len(y_reel)
    dL_dy_pred = 2 * (y_pred - y_reel) / n  # gradient de la MSE par rapport à y_pred
    dw = np.dot(dL_dy_pred, X)              # gradient de la MSE par rapport aux poids
    db = np.sum(dL_dy_pred)                 # gradient de la MSE par rapport au biais
    return dw, db

def update_parameters(w, b, dw, db, learning_rate):
    """
    Cette fonction met à jour les poids et le biais en utilisant les gradients et un taux d'apprentissage
    w = w - learning_rate * dw
    b = b - learning_rate * db
    La fonction retourne les nouveaux poids et biais mis à jour
    """
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b


def perceptron(X, y_reel, learning_rate=0.0001, epochs=100, plot=True):
    # 1) Initialisation des poids et du biais
    features = X.shape[1]
    w, b = initialization(features)

    Losses = [] # pour pouvoir enregistrer notre loss
    
    for epoch in range(epochs): # les epochs sont les itération du modèle sur ttes nos data
        # 2) Prédiction
        y_pred = linear_equation(X, w, b)

        # 3) Calcul de la loss
        error = loss_function(y_reel, y_pred)
        Losses.append(error)

        # 4) Calcul des gradients
        dw, db = lin_gradients(X, y_reel, y_pred)

        # 5) Mise à jour des paramètres
        w, b = update_parameters(w, b, dw, db, learning_rate)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} => Loss: {error:.4f}")

    if plot:
        plt.style.use("dark_background")
        plt.figure(figsize=(10, 4))
        plt.plot(Losses, color='lime')
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.show()
    
    return w, b, Losses

def forward(X, W, b):
    z = W @ X.T + b
    a = 1 / (1 + np.exp(-z))
    return a
def log_loss(y, a, eps=1e-8):
    y = np.ravel(y)
    # on ajoute eps pour eviter un a = 0 et provoquer un log(0)=>!error!
    return -np.mean(y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps))
def gradients(X, y, a):
    """
    Cette fonction calcule les gradients de la Log-Loss
    """
    n = len(y)
    dL_da = 2 * (a - y) / n
    dw = np.dot(dL_da, X)
    db = np.sum(dL_da)
    return dw, db

def artificial_neuron(X, y, learning_rate=0.001, epochs=100, plot=True):
    # 1) Initialisation des poids et du biais
    features = X.shape[1]
    W, b = initialization(features)

    Losses = [] # pour pouvoir enregistrer notre loss
    Acc    = []
    
    for epoch in tqdm(range(epochs)): # les epochs sont les itération du modèle sur ttes nos data
        # 2) Prédiction
        a = forward(X, W, b)

        # 3) Calcul de la loss et de la prediction
        error = log_loss(y, a)
        Losses.append(error)
        
        y_pred = a >= 0.5
        accuracy = accuracy_score(y, y_pred)
        Acc.append(accuracy)

        # 4) Calcul des gradients
        dW, db = gradients(X, y, a)

        # 5) Mise à jour des paramètres
        W, b = update_parameters(W, b, dW, db, learning_rate)

        # if epoch % 5 == 0:
        #     print(f"Epoch {epoch:03d} => Loss: {error:.4f}")

    if plot:
        plt.style.use("dark_background")
        plt.figure(figsize=(10, 4))
        _, ax = plt.subplots(1, 2, figsize=(18, 6))
        ax[0].plot(Losses, color='lime')
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Log-Loss")
        
        ax[1].plot(Acc, color='skyblue')
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        plt.show()
    
    return W, b, Losses, Acc

def predict(X, W, b, diagnose=True, y=None):
    pred_proba = forward(X, W, b)
    pred_labels = pred_proba >= 0.5 # pour retourner 1 si la proba >= 0.5 et 0 sinon
    
    if diagnose:
        if y is None:
            raise(ValueError("y doit être fournis"))
        accuracy = accuracy_score(y, pred_labels)
        print(f"Précision du model : {accuracy}")
    return pred_proba, pred_labels

