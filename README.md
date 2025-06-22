# 🎓 Projekt Zaliczeniowy z Uczenia Maszynowego

**Autorzy:**  
- Artiom Herashchenko  
- Maksym Kudlai  
- Oleksandr Burmahin  

---

## 📌 Opis Projektu

Projekt stanowi kompleksowe opracowanie zagadnień z zakresu uczenia maszynowego. Obejmuje różne techniki – od klasycznych modeli, przez sieci neuronowe, aż po zaawansowane metody optymalizacji.  
Kod został podzielony na moduły odpowiadające poszczególnym zadaniom i progom zaliczeniowym (3.0 – 5.0).

---

## 🗂️ Struktura Projektu

| Plik | Opis |
|------|------|
| `generate_subs.py` | Skrypt do generowania syntetycznego zbioru danych `subscribers.xlsx`. |
| `analiza_subskrybentow.py` | (Na 3.0) Klasyfikacja binarna przy użyciu Naiwnego Bayesa i Drzewa Decyzyjnego (scikit-learn). |
| `sieci_dwuwarstwowe.py` | (Na 3.0) Prosta dwuwarstwowa sieć neuronowa (PyTorch) — problem XOR i zbiór Titanic. |
| `MNIST.py` | (Na 3.0) Konwolucyjna sieć neuronowa do klasyfikacji cyfr (zbiór MNIST). |
| `laborka_na_4.0.py` | (Na 4.0) Analiza zbioru Iris — EDA, Drzewo Decyzyjne i MLP z wykorzystaniem scikit-learn. |
| `laborka_na5.0.py` | (Na 5.0) Ręczna implementacja optymalizatorów (Momentum, Adagrad) bez torch.optim, generowanie raportu PDF. |

---

## 🧠 Zastosowane Techniki

### ✅ Poziom 3.0 – Fundamenty

- **Klasyczne modele (Naive Bayes, Decision Tree)**  
  Przetwarzanie danych za pomocą `Pipeline`, kodowanie cech, skalowanie, wizualizacja wyników (ROC, confusion matrix).

- **Sieci neuronowe (XOR, Titanic)**  
  Własna implementacja dwuwarstwowej sieci neuronowej z wizualizacją gradientów.

- **CNN – MNIST**  
  Konwolucyjna sieć neuronowa z ReLU, MaxPooling, FC. Skuteczna klasyfikacja obrazów.

---

### ✅ Poziom 4.0 – Eksploracja i modelowanie

- **Exploratory Data Analysis (EDA)**  
  Analiza zbioru Iris — pairploty, rozkład cech.

- **Porównanie modeli**  
  Drzewo Decyzyjne i MLPClassifier (scikit-learn), wizualizacja drzewa i krzywa uczenia.

---

### ✅ Poziom 5.0 – Optymalizacja

- **Momentum**  
  Przyspiesza uczenie poprzez akumulację gradientów.

- **Adagrad**  
  Adaptacyjny współczynnik uczenia – osobny dla każdego parametru.

- **Porównanie skuteczności optymalizatorów**  
  Własnoręcznie napisana pętla ucząca, wizualizacja wyników, automatyczny raport PDF.

---

## ▶️ Jak Uruchomić

1. **Instalacja zależności**  
   Zalecane środowisko: Python 3.9+  
   Zainstaluj wymagane pakiety:

   ```bash
   pip install -r requirements.txt