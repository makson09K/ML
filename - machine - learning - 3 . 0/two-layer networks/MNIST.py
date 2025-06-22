import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Hiperparametry
batch_size = 64
epochs = 10
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Ładowanie danych
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# 3. Definicja prostej sieci konwolucyjnej
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. warstwa konwolucyjna
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 28×28 → 28×28, 16 kanałów
        self.pool  = nn.MaxPool2d(2, 2)                          # → 14×14
        # 2. warstwa konwolucyjna
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 14×14 → 14×14, 32 kanały
        # Warstwa w pełni połączona
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN().to(device)

# 4. Optymalizator i funkcja straty
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. Pętla treningowa z zapisem strat (loss) na każdą epokę
train_losses = []
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoka {epoch}/{epochs}, Strata: {epoch_loss:.4f}")

# 6. Wykres błędu treningowego
plt.figure(figsize=(8,5))
plt.plot(train_losses, marker='o')
plt.title("MNIST: średnia strata na epokę")
plt.xlabel("Epoka")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.show()

# 7. Ocena na zbiorze testowym
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

print(f"Dokładność testu: {correct/total:.4f}")
