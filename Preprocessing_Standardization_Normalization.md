# Preprocessing: Standardization dan Normalization

<div align="center">

## Mengapa Data Perlu Discaling?

**Machine Learning Mission 05 — Classification**

</div>

---

# 1. Apa Itu Preprocessing?

**Preprocessing** adalah tahap menyiapkan data sebelum digunakan untuk training model Machine Learning.

Tujuan preprocessing antara lain:

- membersihkan data;
- menangani missing values;
- menangani data invalid;
- menyamakan skala feature;
- menyiapkan data agar sesuai dengan kebutuhan algoritma.

> **Ide utama:**  
> Model Machine Learning bekerja dengan angka.  
> Jika skala antar-feature sangat berbeda, hasil perhitungan model dapat menjadi tidak seimbang.

---

# 2. Mengapa Skala Feature Menjadi Masalah?

Misalkan kita memiliki dua feature:

| Feature | Rentang Nilai |
|---|---:|
| Temperature | 20 – 80 |
| Vibration | 0.1 – 2.0 |

Jika sebuah algoritma menggunakan **jarak**, maka feature dengan angka lebih besar dapat lebih dominan.

Contoh dua sample:

- Sample A = `[30, 0.2]`
- Sample B = `[60, 1.5]`

Perbedaan temperature:

```text
60 - 30 = 30
```

Perbedaan vibration:

```text
1.5 - 0.2 = 1.3
```

Secara numerik, perubahan temperature terlihat jauh lebih besar.

Padahal secara engineering, vibration juga bisa sangat penting.

---

# 3. Scaling

**Scaling** adalah proses mengubah skala feature tanpa menghilangkan pola dasar datanya.

Dua metode yang sangat umum:

1. **Standardization**
2. **Normalization**

---

# 4. Standardization

Standardization mengubah data sehingga secara umum memiliki:

- mean ≈ 0
- standard deviation ≈ 1

Rumus:

$$
z = \frac{x - \mu}{\sigma}
$$

dengan:

- $x$ = nilai asli
- $\mu$ = mean
- $\sigma$ = standard deviation
- $z$ = nilai setelah standardization

---

# 5. Contoh Standardization

Misalkan data temperature:

```text
30, 40, 50
```

Mean:

```text
40
```

Misalkan standard deviation:

```text
10
```

Untuk nilai:

```text
x = 50
```

Maka:

$$
z = \frac{50 - 40}{10}
$$

$$
z = 1
$$

Artinya nilai 50 berada sekitar **1 standard deviation di atas mean**.

---

# 6. Interpretasi Nilai Standardization

Setelah standardization:

```text
z = 0
```

berarti nilai berada di sekitar mean.

```text
z = +1
```

berarti nilai sekitar 1 standard deviation di atas mean.

```text
z = -1
```

berarti nilai sekitar 1 standard deviation di bawah mean.

---

# 7. StandardScaler di Python

Pada `scikit-learn`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Untuk training data:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

Untuk test data:

```python
X_test_scaled = scaler.transform(X_test)
```

---

# 8. Mengapa Training Menggunakan `fit_transform()`?

Pada training data:

```python
scaler.fit_transform(X_train)
```

terjadi dua proses.

### `fit()`

Scaler mempelajari:

- mean training data;
- standard deviation training data.

### `transform()`

Data training diubah menggunakan nilai mean dan standard deviation tersebut.

---

# 9. Mengapa Test Hanya `transform()`?

Test data tidak boleh digunakan untuk mempelajari parameter preprocessing.

Karena itu:

```python
X_test_scaled = scaler.transform(X_test)
```

Scaler tetap menggunakan:

- mean dari training data;
- standard deviation dari training data.

> **Prinsip penting**

```text
TRAIN → FIT + TRANSFORM
TEST  → TRANSFORM ONLY
```

Tujuannya adalah mencegah **data leakage**.

---

# 10. Normalization

Normalization biasanya mengubah data ke rentang tertentu.

Rentang yang paling umum:

```text
0 sampai 1
```

Metode yang umum digunakan adalah **Min-Max Normalization**.

Rumus:

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

---

# 11. Contoh Normalization

Misalkan:

```text
Minimum = 20
Maximum = 80
```

dan:

```text
x = 50
```

Maka:

$$
x' = \frac{50 - 20}{80 - 20}
$$

$$
x' = 0.5
$$

Jadi nilai 50 berubah menjadi:

```text
0.5
```

---

# 12. Interpretasi Min-Max Normalization

Jika menggunakan rentang 0–1:

```text
nilai minimum → 0
nilai maksimum → 1
```

Nilai di antara keduanya akan berada di antara:

```text
0 dan 1
```

Contoh:

| Nilai Asli | Nilai Normalisasi |
|---:|---:|
| 20 | 0.00 |
| 35 | 0.25 |
| 50 | 0.50 |
| 65 | 0.75 |
| 80 | 1.00 |

---

# 13. MinMaxScaler di Python

Pada `scikit-learn`:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
```

Untuk training data:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

Untuk test data:

```python
X_test_scaled = scaler.transform(X_test)
```

Prinsipnya tetap sama:

> scaler hanya belajar dari training data.

---

# 14. Standardization vs Normalization

| Aspek | Standardization | Normalization |
|---|---|---|
| Teknik | StandardScaler | MinMaxScaler |
| Hasil utama | Mean ≈ 0, Std ≈ 1 | Biasanya 0–1 |
| Formula | `(x - mean) / std` | `(x - min) / (max - min)` |
| Bergantung min-max | Tidak | Ya |
| Sensitif terhadap nilai ekstrem | Masih dapat terpengaruh | Cukup sensitif |
| Umum digunakan | Sangat umum | Sangat umum |

---

# 15. Ilustrasi Sederhana

Data asli:

```text
20, 40, 60, 80
```

### Setelah Min-Max Normalization

```text
0.00, 0.33, 0.67, 1.00
```

### Setelah Standardization

Hasil dapat berbentuk seperti:

```text
-1.34, -0.45, 0.45, 1.34
```

Keduanya mengubah skala.

Namun caranya berbeda.

---

# 16. Apakah Scaling Mengubah Makna Data?

Scaling mengubah **representasi numerik**, bukan identitas sample.

Contoh:

```text
Temperature = 50 °C
```

setelah scaling mungkin menjadi:

```text
0.42
```

atau:

```text
0.75
```

Nilai tersebut bukan lagi suhu dalam °C.

Nilai itu adalah **representasi hasil preprocessing**.

---

# 17. Mengapa Scaling Penting untuk KNN?

KNN menentukan kelas berdasarkan **jarak antar-sample**.

Salah satu ukuran jarak yang umum adalah Euclidean Distance:

$$
d = \sqrt{(x_1-y_1)^2 + (x_2-y_2)^2 + \cdots + (x_n-y_n)^2}
$$

Jika satu feature memiliki skala jauh lebih besar, maka feature tersebut dapat mendominasi jarak.

---

# 18. Contoh KNN Tanpa Scaling

Misalkan dua feature:

```text
Temperature = 20 – 80
Vibration   = 0.1 – 2.0
```

Perbedaan temperature:

```text
30
```

Perbedaan vibration:

```text
1.0
```

Pada perhitungan jarak:

```text
30² = 900
1²  = 1
```

Temperature akan jauh lebih dominan.

---

# 19. Setelah Scaling

Misalnya setelah standardization:

```text
Temperature difference = 0.8
Vibration difference   = 1.1
```

Sekarang kontribusi keduanya terhadap jarak menjadi lebih seimbang.

> Karena itu scaling sangat penting untuk algoritma berbasis jarak seperti **KNN**.

---

# 20. Algoritma yang Umumnya Membutuhkan Scaling

Scaling biasanya penting pada algoritma seperti:

- K-Nearest Neighbours (KNN);
- Support Vector Machine (SVM);
- K-Means;
- Neural Network;
- Logistic Regression pada banyak kasus.

---

# 21. Algoritma yang Relatif Tidak Sensitif terhadap Scaling

Model berbasis decision tree biasanya relatif tidak sensitif terhadap skala feature.

Contoh:

- Decision Tree;
- Random Forest;
- Gradient Boosting berbasis tree.

Karena model tersebut membuat keputusan berdasarkan **threshold**, bukan berdasarkan jarak antar-sample.

---

# 22. Kapan Menggunakan StandardScaler?

StandardScaler sering menjadi pilihan yang baik ketika:

- algoritma menggunakan jarak;
- data memiliki rentang feature yang sangat berbeda;
- kita tidak harus memaksa data ke rentang 0–1;
- distribusi data tidak memiliki batas minimum dan maksimum yang jelas.

---

# 23. Kapan Menggunakan MinMaxScaler?

MinMaxScaler sering dipilih ketika:

- kita ingin data berada pada rentang 0–1;
- batas minimum dan maksimum data cukup jelas;
- model tertentu lebih nyaman menerima input dengan rentang terbatas.

---

# 24. Mana yang Lebih Baik?

Tidak ada jawaban:

```text
StandardScaler selalu lebih baik
```

atau:

```text
MinMaxScaler selalu lebih baik
```

Pemilihan scaler bergantung pada:

- karakteristik dataset;
- algoritma;
- keberadaan outlier;
- kebutuhan aplikasi.

Cara yang baik adalah:

> **uji dan bandingkan hasilnya secara objektif.**

---

# 25. Urutan Preprocessing yang Benar

Untuk kasus Mission-05:

```text
RAW DATA
   ↓
CLEAN INVALID DATA
   ↓
INVALID → NaN
   ↓
TRAIN-TEST SPLIT
   ↓
IMPUTATION
   ↓
SCALING
   ↓
MODEL TRAINING
   ↓
EVALUATION
```

---

# 26. Mengapa Split Dilakukan Sebelum Scaling?

Karena test data harus dianggap sebagai:

> **data baru yang belum pernah dilihat sebelumnya.**

Jika scaler dihitung menggunakan seluruh dataset:

```text
TRAIN + TEST
```

maka informasi dari test set ikut memengaruhi preprocessing.

Hal ini disebut:

## Data Leakage

---

# 27. Contoh Data Leakage

### Salah

```python
scaler.fit(X)
```

kemudian baru melakukan:

```python
train_test_split(...)
```

Scaler sudah melihat seluruh data.

---

# 28. Cara yang Benar

Pertama:

```python
X_train, X_test, y_train, y_test = train_test_split(...)
```

Kemudian:

```python
scaler.fit(X_train)
```

Lalu:

```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# 29. Ringkasan

### Standardization

```text
Mean → sekitar 0
Standard deviation → sekitar 1
```

Rumus:

$$
z = \frac{x-\mu}{\sigma}
$$

### Normalization

```text
Nilai → biasanya 0 sampai 1
```

Rumus:

$$
x' = \frac{x-x_{min}}{x_{max}-x_{min}}
$$

---

# 30. Take-Home Message

> **Scaling bukan bertujuan mempercantik angka.**

Scaling dilakukan agar perbedaan skala antar-feature tidak membuat model memberikan pengaruh numerik yang tidak proporsional.

Untuk KNN:

```text
Feature scale berbeda
        ↓
Jarak dapat bias
        ↓
Scaling
        ↓
Jarak menjadi lebih seimbang
        ↓
Classification lebih masuk akal
```

---

# 31. Quick Check

### 1
Mengapa KNN sensitif terhadap skala feature?

### 2
Apa perbedaan utama StandardScaler dan MinMaxScaler?

### 3
Mengapa scaler hanya di-`fit` pada training data?

### 4
Apakah data hasil scaling masih memiliki satuan fisik yang sama?

### 5
Apakah StandardScaler selalu lebih baik daripada MinMaxScaler?

---

<div align="center">

## THINK LIKE AN ENGINEER

**Jangan hanya bertanya:**

*"Apakah model bisa berjalan?"*

**Tetapi juga tanyakan:**

*"Apakah data telah dipersiapkan dengan benar sebelum model belajar?"*

</div>
