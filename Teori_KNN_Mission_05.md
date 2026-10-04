# K-Nearest Neighbours (KNN)

## 1. Apa Itu KNN?

**K-Nearest Neighbours (KNN)** adalah algoritma Machine Learning yang menentukan kelas suatu data baru berdasarkan **kelas dari beberapa tetangga terdekat**.

Contoh:

```text
3 tetangga terdekat:

Normal
Normal
Warning
```

Karena mayoritas adalah `Normal`, maka prediction:

```text
Normal
```

---

## 2. Ide Dasar KNN

Alur KNN:

```text
Data baru
   ↓
Hitung jarak ke data training
   ↓
Pilih K tetangga terdekat
   ↓
Lihat kelas tetangga
   ↓
Pilih kelas mayoritas
```

KNN tidak membangun persamaan model yang kompleks.

Ia menyimpan data training dan membuat keputusan ketika prediction dilakukan.

---

## 3. Apa Arti K?

`K` adalah jumlah tetangga terdekat yang digunakan untuk menentukan kelas.

Contoh:

```python
KNeighborsClassifier(n_neighbors=3)
```

berarti:

```text
K = 3
```

Model menggunakan **3 tetangga terdekat**.

---

## 4. Pengaruh Nilai K

### K terlalu kecil

Contoh:

```text
K = 1
```

Keputusan sangat bergantung pada satu sample terdekat.

Akibatnya:

- sensitif terhadap noise;
- decision boundary dapat terlalu kompleks.

### K terlalu besar

Akibatnya:

- keputusan menjadi terlalu umum;
- pola lokal dapat hilang.

> Tidak ada nilai K terbaik untuk semua dataset.

---

## 5. Mengukur Jarak

KNN membutuhkan ukuran jarak antar-sample.

Salah satu yang paling umum adalah **Euclidean Distance**:

$$
d = \sqrt{(x_1-y_1)^2 + (x_2-y_2)^2 + \cdots + (x_n-y_n)^2}
$$

Semakin kecil nilai `d`, semakin dekat kedua sample.

---

## 6. Contoh Sederhana

Misalkan dua feature:

```text
Temperature
Vibration
```

Sample A:

```text
[30, 0.2]
```

Sample B:

```text
[34, 0.5]
```

Jarak Euclidean:

$$
d = \sqrt{(30-34)^2 + (0.2-0.5)^2}
$$

Sample dengan jarak paling kecil dianggap sebagai tetangga terdekat.

---

## 7. Mengapa Scaling Penting?

KNN menggunakan jarak.

Misalnya:

```text
Temperature = 20 – 80
Vibration   = 0.1 – 2.0
```

Feature `Temperature` memiliki skala angka jauh lebih besar.

Tanpa scaling, temperature dapat mendominasi perhitungan jarak.

Karena itu KNN biasanya membutuhkan:

```python
StandardScaler()
```

atau:

```python
MinMaxScaler()
```

---

## 8. Workflow KNN

```text
DATA
  ↓
CLEANING
  ↓
TRAIN-TEST SPLIT
  ↓
IMPUTATION
  ↓
SCALING
  ↓
KNN TRAINING
  ↓
PREDICTION
  ↓
EVALUATION
```

---

## 9. KNN di Python

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5
)

model.fit(
    X_train_scaled,
    y_train
)
```

Prediction:

```python
y_pred = model.predict(
    X_test_scaled
)
```

---

## 10. Prediction Probability

KNN juga dapat memberikan prediction probability:

```python
probability = model.predict_proba(
    X_test_scaled
)
```

Contoh:

```text
Fault   = 0.20
Normal  = 0.00
Warning = 0.80
```

Prediction:

```text
Warning
```

> Probability tinggi tidak berarti prediction pasti benar.

---

## 11. Kelebihan KNN

- sederhana;
- mudah dipahami;
- tidak membutuhkan model matematis yang kompleks;
- cocok untuk memperkenalkan konsep classification;
- dapat bekerja baik pada dataset dengan pola lokal yang jelas.

---

## 12. Keterbatasan KNN

- sensitif terhadap skala feature;
- sensitif terhadap pemilihan nilai K;
- prediction dapat menjadi lambat pada dataset besar;
- dapat terganggu oleh noise dan outlier;
- performa menurun jika jumlah feature terlalu banyak.

---

## 13. KNN untuk Classification

Untuk Mission-05:

```text
Input:
Water Level
Flow Rate
Current
Temperature
Vibration

        ↓

       KNN

        ↓

Normal / Warning / Fault
```

---

## 14. Take-Home Message

KNN mengambil keputusan berdasarkan:

```text
"Siapa tetangga terdekat saya?"
```

Kemudian:

```text
"Mayoritas mereka termasuk kelas apa?"
```

Hal penting yang perlu diingat:

- K menentukan jumlah tetangga;
- jarak menentukan kedekatan;
- scaling sangat penting;
- nilai K perlu diuji;
- evaluasi tidak cukup hanya dengan accuracy.

---

## Quick Check

1. Apa arti `K` pada KNN?
2. Mengapa scaling penting pada KNN?
3. Apa yang terjadi jika K terlalu kecil?
4. Apa yang terjadi jika K terlalu besar?
5. Mengapa nilai K terbaik tidak selalu sama untuk semua dataset?
