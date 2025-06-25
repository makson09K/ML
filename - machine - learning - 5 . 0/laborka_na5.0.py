import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import os

print("Working dir:", os.getcwd())

# ----------------------------
# 1) Przygotowanie danych Iris
# ----------------------------
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 1.1 EDA: pairplot
eda_plot = sns.pairplot(df, hue='target', diag_kind='hist')
eda_plot.fig.savefig("eda_pairplot.png")
plt.close('all')

X = df[data.feature_names].values
y = df['target'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ----------------------------
# 2) Drzewo decyzyjne
# ----------------------------
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)

plt.figure(figsize=(8,6))
plot_tree(tree, feature_names=data.feature_names,
          class_names=data.target_names, filled=True)
plt.savefig("tree.png")
plt.close()

# ----------------------------
# 3) Dwuwarstwowa sieć w PyTorch
# ----------------------------
class TwoLayerNet(nn.Module):
    def __init__(self, inp, hid, out):
        super().__init__()
        self.fc1 = nn.Linear(inp, hid)
        self.fc2 = nn.Linear(hid, out)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cpu")
Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
yt = torch.tensor(y_train, dtype=torch.long).to(device)
ds = TensorDataset(Xt, yt)
loader = DataLoader(ds, batch_size=16, shuffle=True)

# ----------------------------
# 4) Ręczne optymalizatory
# ----------------------------
def train_manual(opt, lr=1e-2, momentum=0.9, epochs=100):
    model = TwoLayerNet(4,10,3).to(device)
    criterion = nn.CrossEntropyLoss()
    velocity = {p: torch.zeros_like(p) for p in model.parameters()}
    grad_sq = {p: torch.zeros_like(p) for p in model.parameters()}
    loss_hist = []
    for ep in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                g = p.grad
                if opt in ('adagrad','both'):
                    grad_sq[p] += g * g
                    adapted_lr = lr / (torch.sqrt(grad_sq[p]) + 1e-8)
                else:
                    adapted_lr = lr
                if opt in ('momentum','both'):
                    velocity[p] = momentum * velocity[p] - adapted_lr * g
                    p.data += velocity[p]
                else:
                    p.data -= adapted_lr * g
            total_loss += loss.item() * xb.size(0)
        loss_hist.append(total_loss / len(loader.dataset))
    # Zapis loss curve i confusion matrix
    plt.plot(loss_hist, marker='o')
    plt.title(f"Loss curve ({opt})")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy")
    plt.grid(True)
    loss_file = f"loss_{opt}.png"
    plt.savefig(loss_file)
    plt.close()

    model.eval()
    Xtst = torch.tensor(X_test, dtype=torch.float32).to(device)
    preds = model(Xtst).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=data.target_names)
    disp.plot()
    plt.title(f"Confusion ({opt})")
    cm_file = f"cm_{opt}.png"
    plt.savefig(cm_file)
    plt.close()

    return acc

acc_m = train_manual('momentum')
acc_a = train_manual('adagrad')
acc_b = train_manual('both')


