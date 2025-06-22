import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ----------------------------
# Definicja sieci dwuwarstwowej
# ----------------------------
class TwoLayerNet(nn.Module):
    def __init__(self, inp, hid, out):
        super().__init__()
        self.fc1 = nn.Linear(inp, hid)
        self.fc2 = nn.Linear(hid, out)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# ----------------------------
# Funkcja uczenia i zapisu błędów gradientu
# ----------------------------
def train_net(model, X, y, epochs=1000, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    l1, l2 = [], []
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        # Średnia wartość gradientu jako przybliżenie błędu warstwy
        l1.append(model.fc1.weight.grad.abs().mean().item())
        l2.append(model.fc2.weight.grad.abs().mean().item())
        optimizer.step()
    return l1, l2, model

# ----------------------------
# 1) Uczenie na zbiorze XOR
# ----------------------------
X_xor = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y_xor = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)
net = TwoLayerNet(2,4,1)
l1_xor, l2_xor, net = train_net(net, X_xor, y_xor, epochs=5000, lr=0.1)
# Wykres błędów gradientów
plt.plot(l1_xor, label='warstwa1'); plt.plot(l2_xor, label='warstwa2')
plt.title('XOR: gradienty wg warstwy'); plt.legend(); plt.show()

# ----------------------------
# 2) Uczenie na zbiorze Titanic
# ----------------------------
df = sns.load_dataset('titanic').dropna(subset=['age','fare','sex','embarked','survived'])
df['sex'] = df['sex'].map({'male':0,'female':1})
df['embarked'] = df['embarked'].map({'S':0,'C':1,'Q':2})

X = df[['age','fare','sex','embarked']].values
y = df['survived'].values.reshape(-1,1)
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
Xt = torch.tensor(X_train, dtype=torch.float32)
yt = torch.tensor(y_train, dtype=torch.float32)

net2 = TwoLayerNet(4,8,1)
l1_tit, l2_tit, net2 = train_net(net2, Xt, yt, epochs=1000, lr=0.01)
# Wykres błędów gradientów
plt.plot(l1_tit, label='warstwa1'); plt.plot(l2_tit, label='warstwa2')
plt.title('Titanic: gradienty wg warstwy'); plt.legend(); plt.show()
