# 🎯 Cel projektu

**Analiza porównawcza klasyfikatorów:**
- Drze­wa decyzyjne – szybki, interpretable baseline.
- Sieci neuronowej – porównanie gotowej implementacji (`sklearn.MLPClassifier`) z własną implementacją w PyTorch.
- Zbadanie wpływu optymalizatorów (*Momentum*, *Adagrad*, *kombinacja*) na uczenie sieci.

---

## 🧩 Elementy projektu

### 1. Wczytanie i analiza danych (EDA)
- Ładowanie zbioru Iris (150 próbek, 4 cechy, 3 klasy).
- Pairplot cech względem klasy – wizualna ocena korelacji i separowalności.  
  Zapisano plik: `eda_pairplot.png`

### 2. Drzewo decyzyjne (Decision Tree)
- Model: `DecisionTreeClassifier(max_depth=3)`
- Wykres drzewa (`tree.png`) i dokładność (`acc_tree`) jako referencyjny baseline.

### 3. Sieć neuronowa – wersja sklearn
- `MLPClassifier` – warstwy 10 i 5, aktywacja ReLU, solver `adam`, max 1000 epok.
- Wyniki:
  - Krzywa strat: `mlp_loss.png`
  - Macierz pomyłek: `conf_matrix.png`
  - Dokładność: `acc_mlp`

### 4. Sieć neuronowa – PyTorch z ręcznym optymalizatorem
- Model `TwoLayerNet` (4 → 10 → 3) z ReLU.
- Optymalizatory:
  1. **Momentum**
  2. **Adagrad**
  3. **Both**
- Dla każdej opcji przeprowadzony trening:
  - Krzywa strat (`loss_{opt}.png`)
  - Macierz pomyłek (`cm_{opt}.png`)
  - Dokładność (`acc_m`, `acc_a`, `acc_b`)

---

## 📊 Co przedstawia raport?
- Analiza liniowej separowalności klas w zbiorze Iris.
- Drzewo decyzyjne jako prosty baseline.
- `sklearn MLP` – gotowy model z prostym pipeline.
- PyTorch z ręcznym optymalizatorem – pełna kontrola i eksperymenty:
  - Jak działa Momentum?
  - Czy Adagrad adaptuje się do gradientów?
  - Czy kombinacja daje lepsze rezultaty?

Wszystkie modele są oceniane pod kątem:
- Krzywej strat (szybkość i stabilność uczenia).
- Macierzy pomyłek (dokładność klas).
- Dokładności – która konfiguracja daje najlepszą skuteczność.

---

## 🔧 Dlaczego to ważne?
- Pokazuje różnicę między gotowym rozwiązaniem a eksperymentem kontrolowanym.
- Umożliwia praktyczne zrozumienie działania optymalizatorów.
- Porównanie bibliotek `sklearn` i `PyTorch` pod względem użycia i kontroli.

---

## 📎 Załączniki
- `eda_pairplot.png`
- `tree.png`
- `mlp_loss.png`, `conf_matrix.png`
- `loss_momentum.png`, `cm_momentum.png`
- `loss_adagrad.png`, `cm_adagrad.png`
- `loss_both.png`, `cm_both.png`
