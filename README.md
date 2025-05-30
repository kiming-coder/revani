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

### 4. 📓 Buka [notebooks/energy_prediction.ipynb](notebooks/energy_prediction.ipynb) untuk memulai eksplorasi dan pelatihan model.


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

### Langkah-Langkah 3 Analisis Regresi Linear untuk Prediksi Konsumsi Energi

```python
from sklearn.preprocessing import MinMaxScaler  # Seharusnya MinMaxScaler, bukan HizhiosScaler (typo)

scaler = MinMaxScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
normalized_df.head()
