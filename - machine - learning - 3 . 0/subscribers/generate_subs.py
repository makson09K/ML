# -*- coding: utf-8 -*-
"""
Ten skrypt generuje plik 'subscribers.xlsx' z przykładowymi danymi,
potrzebny do uruchomienia głównego skryptu analitycznego.
Uruchom go tylko jeden raz.
"""
import pandas as pd
import numpy as np

# Liczba wierszy do wygenerowania
num_rows = 1000

# Możliwe wartości dla kolumn kategorycznych
age_categories = ['young', 'adult', 'middle-aged', 'senior']
sex_categories = ['Male', 'Female']
residence_categories = ['Urban', 'Suburban', 'Rural']
subscribes_categories = ['Yes', 'No']

# Stworzenie słownika z danymi
data = {
    'Age': np.random.choice(age_categories, num_rows, p=[0.2, 0.4, 0.3, 0.1]),
    'Sex': np.random.choice(sex_categories, num_rows),
    'Income': np.random.randint(25000, 150000, num_rows),
    'Residence': np.random.choice(residence_categories, num_rows, p=[0.5, 0.3, 0.2]),
    'Subscribes': np.random.choice(subscribes_categories, num_rows, p=[0.7, 0.3]) # 30% szans na 'Yes'
}

# Konwersja słownika na ramkę danych (DataFrame)
df = pd.DataFrame(data)

# Nazwa pliku wyjściowego
output_filename = 'subscribers.xlsx'

try:
    # Zapisanie ramki danych do pliku Excel
    df.to_excel(output_filename, index=False, engine='openpyxl')
    print(f"Plik '{output_filename}' został pomyślnie wygenerowany i zawiera {num_rows} wierszy.")
    print("Możesz teraz uruchomić główny skrypt analityczny.")
except ImportError:
    print("Błąd: Brak biblioteki 'openpyxl'.")
    print("Proszę zainstalować ją za pomocą polecenia: pip install openpyxl")
except Exception as e:
    print(f"Wystąpił nieoczekiwany błąd podczas zapisywania pliku: {e}")
