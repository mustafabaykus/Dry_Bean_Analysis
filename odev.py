import pandas as pd
import numpy as np

# 1. Veri setini yükle
df = pd.read_excel("Dry_Bean_Dataset.xlsx")

# 2. Eksik veri ekleme
df_missing = df.copy()

# %5 eksik veri 'Area' ve 'Perimeter' sütunlarına
for col in ['Area', 'Perimeter']:
    df_missing.loc[df_missing.sample(frac=0.05).index, col] = np.nan

# %35 eksik veri 'Extent' sütununa
df_missing.loc[df_missing.sample(frac=0.35).index, 'Extent'] = np.nan

# 3. Eksik verileri doldurma/silme
df_missing['Area'].fillna(df_missing['Area'].median(), inplace=True)          # Medyan ile doldurma
df_missing['Perimeter'].fillna(df_missing['Perimeter'].mean(), inplace=True)  # Ortalama ile doldurma
df_cleaned = df_missing.dropna(subset=['Extent'])                             # Extent sütunu için satır silme

# Sonuç kontrolü
print("Kalan veri satır sayısı:", df_cleaned.shape[0])
print("Eksik veriler (kalmadı):")
print(df_cleaned.isnull().sum())