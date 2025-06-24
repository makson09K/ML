import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Funkcja do bezpiecznego tekstu (usuwanie znaków nieobsługiwanych przez Latin-1)
def safe(text: str) -> str:
    return text.replace('–', '-').replace('—', '-')

# 1. Wczytanie danych i analiza eksploracyjna (EDA)
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
sns.pairplot(df, hue='target', diag_kind='hist')
plt.savefig("eda_pairplot.png")
plt.clf()

# 2. Przygotowanie danych (normalizacja, podział)
X = df[data.feature_names]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 3. Klasyfikator drzew decyzyjnych
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
plt.figure(figsize=(8,6))
plot_tree(tree, feature_names=data.feature_names,
          class_names=data.target_names, filled=True)
plt.savefig("tree.png")
plt.clf()
y_pred_tree = tree.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)

# 4. Sztuczna sieć neuronowa (MLP) z solverem adam
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 5),
    activation='relu',
    solver='adam',             # stochastyczny solver z loss_curve_
    learning_rate_init=0.01,
    tol=1e-4,
    max_iter=1000,             # zwiększona liczba epok
    random_state=42,
    verbose=False
)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)

# Wykres funkcji straty (loss curve)
loss_values = mlp.loss_curve_

# 5. Rysowanie wykresu strat
plt.plot(range(1, len(loss_values)+1), loss_values, marker='o')
plt.title("MLP Loss Curve (solver=adam)")
plt.xlabel("Epoka")
plt.ylabel("Strata")
plt.savefig("mlp_loss.png")
plt.clf()

# 6. Macierz pomyłek dla MLP
cm = confusion_matrix(y_test, y_pred_mlp)
disp = ConfusionMatrixDisplay(cm, display_labels=data.target_names)
disp.plot()
plt.savefig("conf_matrix.png")
plt.clf()