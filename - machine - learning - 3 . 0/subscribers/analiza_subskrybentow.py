# -*- coding: utf-8 -*-
"""
Skrypt do klasyfikacji subskrybentów z wykorzystaniem modeli
Naiwnego Bayes'a i Drzewa Decyzyjnego.

WERSJA FINALNA - Przetestowana i dostosowana do wygenerowanego pliku subscribers.xlsx
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, RocCurveDisplay)

# --- KROK 1: Wczytanie i przygotowanie danych ---

# Nazwa kolumny, którą będziemy przewidywać.
TARGET_COLUMN = 'Subscribes'

try:
    # Wczytanie pliku Excel. Wymaga to zainstalowanej biblioteki 'openpyxl'.
    df = pd.read_excel('subscribers.xlsx', engine='openpyxl')

    # Czyszczenie nazw kolumn na wszelki wypadek (usuwa zbędne spacje)
    df.columns = df.columns.str.strip()

    # Sprawdzenie, czy kolumna docelowa istnieje w pliku
    if TARGET_COLUMN not in df.columns:
        print(f"BŁĄD KRYTYCZNY: Kolumna '{TARGET_COLUMN}' nie została znaleziona w pliku.")
        print(f"Sprawdź, czy na pewno tak się nazywa. Dostępne kolumny to: {list(df.columns)}")
        exit()

except FileNotFoundError:
    print(
        "Błąd: Plik 'subscribers.xlsx' nie został znaleziony. Upewnij się, że znajduje się w tym samym folderze co skrypt.")
    exit()
except ImportError:
    print("Błąd: Brak biblioteki 'openpyxl'. Zainstaluj ją używając polecenia: pip install openpyxl")
    exit()
except Exception as e:
    print(f"Wystąpił nieoczekiwany błąd podczas wczytywania pliku: {e}")
    exit()

# Konwersja zmiennej docelowej na format liczbowy (0/1)
# Używamy .str.lower() aby kod działał zarówno dla 'yes', 'Yes', 'YES' itd.
df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

# Podział danych na cechy (X) i zmienną docelową (y)
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Zdefiniowanie cech numerycznych i kategorycznych zgodnie ze strukturą pliku
numeric_features = ['Income']
categorical_features = ['Sex', 'Residence', 'Age']

# Sprawdzenie, czy wszystkie zdefiniowane kolumny istnieją w danych
for col in numeric_features + categorical_features:
    if col not in X.columns:
        print(f"Błąd konfiguracji: Kolumna '{col}' zdefiniowana jako cecha nie istnieje w pliku.")
        print(f"Dostępne cechy to: {list(X.columns)}")
        exit()

# Podział na zbiór treningowy i testowy z zachowaniem proporcji klas
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- KROK 2: Stworzenie potoków (Pipelines) do przetwarzania i modelowania ---

# Transformer, który zastosuje inne operacje do różnych typów kolumn
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),  # Skalowanie dla cech numerycznych
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Kodowanie dla cech kategorycznych
    ],
    remainder='passthrough'  # Pozostaw inne kolumny bez zmian (jeśli istnieją)
)

# Pipeline dla Naiwnego Klasyfikatora Bayes'a
nb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])

# Pipeline dla Drzewa Decyzyjnego
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# --- KROK 3: Trening modeli ---

print("Rozpoczynam trening modelu Naiwnego Bayes'a...")
nb_pipeline.fit(X_train, y_train)
print("Trening zakończony.")

print("\nRozpoczynam trening modelu Drzewa Decyzyjnego...")
dt_pipeline.fit(X_train, y_train)
print("Trening zakończony.")

# --- KROK 4: Ocena modeli ---

# Predykcje na zbiorze testowym, którego modele nigdy wcześniej nie widziały
y_pred_nb = nb_pipeline.predict(X_test)
y_pred_dt = dt_pipeline.predict(X_test)


# Funkcja do wygodnego wyświetlania wyników
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n--- Wyniki dla modelu: {model_name} ---")

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Dokładność (Accuracy): {accuracy:.4f}")

    print("\nRaport klasyfikacji (Precision, Recall, F1-Score):")
    print(classification_report(y_true, y_pred, target_names=['No', 'Yes']))

    # Rysowanie macierzy pomyłek
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Przewidziana etykieta')
    plt.ylabel('Prawdziwa etykieta')
    plt.title(f'Macierz pomyłek - {model_name}')
    plt.show()


# Ocena obu modeli
evaluate_model(y_test, y_pred_nb, "Naiwny Bayes")
evaluate_model(y_test, y_pred_dt, "Drzewo Decyzyjne")

# Porównanie krzywych ROC na jednym wykresie
print("\nGenerowanie wykresu krzywej ROC do porównania modeli...")
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(nb_pipeline, X_test, y_test, name="Naiwny Bayes", ax=ax)
RocCurveDisplay.from_estimator(dt_pipeline, X_test, y_test, name="Drzewo Decyzyjne", ax=ax)
ax.set_title("Krzywa ROC - Porównanie modeli")
plt.grid()
plt.show()
print("\nAnaliza zakończona.")