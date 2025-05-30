# 🔋 Energy Predictor  
## _Smart Home, Smarter Energy_

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![ML Project](https://img.shields.io/badge/Machine_Learning-Ready-brightgreen?logo=scikit-learn)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)


**Energy Predictor** adalah proyek prediksi konsumsi energi rumah tangga berbasis _machine learning_ yang bertujuan untuk membantu mengoptimalkan penggunaan listrik secara lebih efisien dan cerdas.

---

## ✨ Apa yang Bisa Dilakukan?
- 📈 Memprediksi konsumsi energi rumah berdasarkan data sensor lingkungan seperti suhu, kelembaban, dan waktu
- 📊 Visualisasi data untuk melihat pola konsumsi energi
- 🧠 Menggunakan berbagai model Machine Learning seperti Linear Regression, Random Forest, dan Gradient Boosting
- 🧪 Evaluasi performa model menggunakan MAE, RMSE, dan R² Score
- 🔁 Struktur proyek modular yang mudah dikembangkan dan diintegrasikan

---

## 🛠️ Tech Stack

Proyek ini dibangun menggunakan:

- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Jupyter Notebook](https://jupyter.org/)

---

## 📂 Struktur Proyek

energy-predictor/

├── data/ # Dataset mentah dan hasil preprocessing

├── notebooks/ # Jupyter notebooks untuk eksplorasi dan modeling

├── models/ # Model machine learning yang disimpan

├── utils/ # Fungsi bantu (preprocessing, evaluasi, dsb)

├── requirements.txt # Dependensi proyek

├── README.md # Dokumentasi proyek


---

## 🚀 Cara Menjalankan

### 1. Clone repositori:
   ```bash
   git clone https://github.com/username/energy-predictor.git
   cd energy-predictor
   ```


### 2 Install dependencies:
```python
pip install -r requirements.txt
```
### 3 Jalankan notebook:
````python
jupyter notebook
````

### 4. 📓 Buka `[notebooks/energy_prediction.ipynb]`
untuk memulai eksplorasi dan pelatihan model.


## Langkah-Langkah


### pertama tama buka di [vscode](https://code.visualstudio.com/) atau [jupyter.org](https://jupyter.org/try)

### 1. Tampilkan data data dari file untuk dibaca
#### Code 
```python
import pandas as pd
# Baca file CSV
df = pd.read_csv("energydata_complete.csv")
# Tampilkan data awal
df.head()
```
#### Output
![App Screenshot](https://raw.githubusercontent.com/kiming-coder/revani/refs/heads/main/1.png)

### 2 Langkah ke dua periksa apakah ada nilai kosong ?

##### 1. Berdasarkan output yang diberikan, berikut adalah langkah-langkah yang dilakukan:

##### Menghapus Kolom yang Tidak Diperlukan:
```
df = df.drop(['date', 'lights'], axis=1)
```

- ##### Kolom 'date' dan 'lights' dihapus dari DataFrame karena dianggap tidak diperlukan untuk analisis lebih lanjut.
##### 2. Memeriksa Nilai Kosong (Missing Values):
```python
df.isnull().sum()
```

#### Code
```python
# Drop kolom yang tidak diperlukan
df = df.drop(['date', 'lights'], axis=1)


# Cek apakah ada nilai kosong
df.isnull().sum()
```

#### Output
![App Screenshot](https://raw.githubusercontent.com/kiming-coder/revani/refs/heads/main/2.png)

###  3 Analisis Regresi Linear untuk Prediksi Konsumsi Energi

```python
from sklearn.preprocessing import MinMaxScaler  # Seharusnya MinMaxScaler, bukan HizhiosScaler (typo)

scaler = MinMaxScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
normalized_df.head()
```
#### Output 
![App Screenshot](https://raw.githubusercontent.com/kiming-coder/revani/refs/heads/main/3.png)
- Tujuan: Menyamakan skala fitur (0-1) untuk meningkatkan performa model.
- Output: DataFrame dengan nilai ternormalisasi (contoh: `Appliances=0.046729, T1=0.32735)`.

### 4. Pisahkan fitur (X) dan target (y)
X = normalized_df.drop('Appliances', axis=1)
y = normalized_df['Appliances']

### Bagi data menjadi train (80%) dan test (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#### Output
![App Screenshot](https://raw.githubusercontent.com/kiming-coder/revani/refs/heads/main/4.png)
- Target: Variabel `Appliances` (konsumsi energi).
- Random State: 42 untuk reproduktibilitas.

### 5. Pelatihan Model Regresi Linear
```python
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```
- Output: Model terlatih `(LinearRegression())`.

### 6. Evaluasi Model
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = lr_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))
```
#### Output
![woi jngn asal copas ya dek](https://raw.githubusercontent.com/kiming-coder/revani/refs/heads/main/5.png)

- Hasil:
  - MAE: 0.0499 (rata-rata error absolut)
  - R²: 0.146 (korelasi lemah, model kurang baik).
 

### 7. Visualisasi Prediksi vs Aktual
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.legend()
plt.title("Prediksi vs Aktual Konsumsi Energi")
plt.xlabel("Sample")
plt.ylabel("Konsumsi Energi (Normalisasi)")
plt.show()
```
#### Output
![woi jangan asal jiplak orang punya revani](https://raw.githubusercontent.com/kiming-coder/revani/refs/heads/main/6.png)
- Interpretasi: Prediksi (biru) tidak mengikuti pola aktual (oranye) dengan baik, sesuai nilai R² rendah.


## 📦 Dataset
### Dataset berisi data konsumsi energi rumah tangga berdasarkan:
- ###### Tanggal dan waktu
- ###### Suhu dalam dan luar ruangan
- ###### Kelembaban
- ###### Intensitas cahaya alami
- ###### Kegiatan di beberapa ruangan (opsional)

## 🧪 Evaluasi Model
### Model yang dibangun diuji secara ketat dengan metrik:
- ###### 📉 Mean Absolute Error (MAE)
- ###### 📉 Root Mean Square Error (RMSE)
- ###### 📈 R² Score
Tujuan evaluasi ini adalah untuk mengetahui seberapa akurat model memprediksi nilai konsumsi energi dibandingkan data sebenarnya.

## 📌 Inspirasi Proyek
### Proyek ini terinspirasi dari:
- ###### Kebutuhan untuk meningkatkan efisiensi energi di rumah tangga
- ###### Penerapan nyata data science untuk keberlanjutan
- ###### Konsep rumah pintar (smart home) dan manajemen energi berbasis AI

## 🧑‍💻 Pembuat


```quote
Proyek ini dibuat oleh Revani Aprilia (23283030)
"Bangun masa depan rumah yang hemat energi, dimulai dari data 
```










