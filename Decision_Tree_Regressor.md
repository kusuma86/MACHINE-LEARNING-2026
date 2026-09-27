# 1 — Decision Tree untuk Regression

## Apa itu Decision Tree Regressor?

Decision Tree Regressor adalah metode Machine Learning yang digunakan untuk memprediksi **nilai numerik kontinu** dengan membagi data menjadi beberapa kelompok berdasarkan aturan keputusan.

Contoh target:

- konsumsi energi (kWh)
- temperatur (°C)
- flow rate (L/min)
- daya listrik (W)

### Ide utama

Model membuat serangkaian pertanyaan seperti:

```text
Apakah current_a <= 7.5?
        ↓
   Ya        Tidak
```

Setiap pertanyaan membagi data menjadi kelompok yang lebih spesifik.

---

# 2 — Perbedaan Classification dan Regression

Decision Tree dapat digunakan untuk dua jenis masalah.

| Jenis | Target | Contoh |
|---|---|---|
| Classification | kategori | Normal / Warning / Fault |
| Regression | angka kontinu | 12.4 kWh |

### Classification Tree

Leaf menghasilkan **kelas**.

```text
Normal
Fault
Warning
```

### Regression Tree

Leaf menghasilkan **nilai numerik**.

```text
8.5 kWh
11.2 kWh
13.1 kWh
```

---

# 3 — Struktur Decision Tree

Decision Tree terdiri dari:

### Root Node

Node pertama yang berisi seluruh data.

### Decision Node

Node yang membuat aturan pemisahan.

Contoh:

```text
current_a <= 7.5
```

### Branch

Cabang hasil keputusan.

```text
True / False
```

### Leaf Node

Node akhir yang menghasilkan prediction.

Contoh:

```text
Predicted Energy = 11.8 kWh
```

---

# 4 — Contoh Sederhana

Misalkan kita hanya menggunakan feature:

```text
current_a
```

untuk memprediksi:

```text
energy_kwh
```

Decision Tree dapat membentuk aturan:

```text
current_a <= 7.0
        |
   -------------
   |           |
  Ya          Tidak
   |           |
8.2 kWh    current_a <= 8.0
               |
          -------------
          |           |
         Ya          Tidak
          |           |
      11.0 kWh     13.2 kWh
```

### Intuisi

Decision Tree membagi ruang data menjadi beberapa daerah.

Setiap daerah memiliki nilai prediction sendiri.

---

# 5 — Bagaimana Tree Memilih Split?

Tree mencoba mencari feature dan nilai threshold yang menghasilkan pembagian data paling baik.

Contoh kandidat:

```text
current_a <= 7.2
current_a <= 7.5
current_a <= 7.8
```

Model mencoba beberapa split dan memilih split yang membuat nilai target pada setiap kelompok menjadi lebih seragam.

Untuk regression, salah satu ukuran yang umum digunakan adalah **Mean Squared Error (MSE)**.

---

# 6 — MSE pada Decision Tree

Untuk sebuah node:

$$
MSE =
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\bar{y})^2
$$

dengan:

- $y_i$ = nilai target sample ke-$i$
- $\bar{y}$ = rata-rata target pada node
- $n$ = jumlah sample

### Tujuan split

Tree mencari split yang membuat:

```text
MSE setelah split
```

menjadi lebih kecil.

Artinya, sample dalam masing-masing kelompok menjadi lebih mirip.

---

# 7 — Nilai Prediction pada Leaf

Pada Decision Tree Regressor, prediction pada sebuah leaf biasanya merupakan **rata-rata target** dari sample yang masuk ke leaf tersebut.

Misalnya sebuah leaf berisi:

```text
10.8
11.2
11.5
```

maka prediction:

$$
\hat{y}
=
\frac{10.8+11.2+11.5}{3}
$$

$$
\hat{y}
\approx 11.17
$$

Jadi semua sample baru yang masuk ke leaf tersebut akan mendapat prediction sekitar:

```text
11.17
```

---

# 8 — Multiple Features pada Decision Tree

Decision Tree tidak terbatas pada satu feature.

Contoh feature:

- `voltage_v`
- `current_a`
- `operating_hours`
- `ambient_temp_c`
- `power_factor`

Model dapat membuat aturan seperti:

```text
current_a <= 7.4
        ↓
operating_hours <= 7.1
        ↓
power_factor <= 0.92
        ↓
Prediction = 9.8 kWh
```

Tree bebas memilih feature yang dianggap paling membantu pada setiap node.

---

# 9 — Apa itu max_depth?

`max_depth` menentukan kedalaman maksimum tree.

Contoh:

```python
DecisionTreeRegressor(
    max_depth=3
)
```

Artinya tree hanya boleh tumbuh sampai kedalaman tertentu.

### Jika max_depth terlalu kecil

```text
Model terlalu sederhana
→ underfitting
```

### Jika max_depth terlalu besar

```text
Model terlalu kompleks
→ overfitting
```

---

# 10 — Underfitting dan Overfitting

## Underfitting

Tree terlalu sederhana.

```text
Data Training
    ↓
Tree sangat pendek
    ↓
Pola penting tidak tertangkap
```

Hasil:

- training error tinggi
- test error tinggi

## Overfitting

Tree terlalu dalam.

```text
Data Training
    ↓
Tree sangat kompleks
    ↓
Hampir menghafal data
```

Hasil:

- training error sangat kecil
- test error bisa besar

---

# 11 — Parameter Penting

Beberapa parameter yang sering digunakan:

### `max_depth`

Membatasi kedalaman tree.

### `min_samples_split`

Jumlah minimum sample agar sebuah node boleh di-split.

### `min_samples_leaf`

Jumlah minimum sample yang harus ada pada leaf.

### `random_state`

Membantu menghasilkan hasil yang konsisten saat eksperimen diulang.

Contoh:

```python
model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)
```

---

# 12 — Kelebihan Decision Tree Regressor

### Mudah dipahami

Aturannya dapat divisualisasikan.

### Tidak membutuhkan scaling

Decision Tree tidak bergantung pada jarak atau besar skala feature.

Artinya:

```text
Voltage ≈ 220
Power Factor ≈ 0.9
```

tetap dapat diproses tanpa StandardScaler.

### Dapat menangkap hubungan nonlinear

Decision Tree tidak memaksa hubungan berbentuk garis lurus.

### Dapat menangani interaksi antar-feature

Feature yang berbeda dapat digunakan pada node yang berbeda.

---

# 13 — Kekurangan Decision Tree Regressor

### Mudah overfitting

Tree yang terlalu dalam dapat menghafal data training.

### Prediction berbentuk potongan

Decision Tree menghasilkan prediction yang cenderung bersifat bertingkat.

```text
Region A → 8.2
Region B → 10.7
Region C → 12.9
```

### Sensitif terhadap perubahan data

Perubahan kecil pada data training dapat menghasilkan struktur tree yang berbeda.

### Kurang stabil dibanding ensemble

Karena hanya menggunakan satu tree.

---

# 14 — Linear Regression vs Decision Tree

| Aspek | Linear Regression | Decision Tree Regressor |
|---|---|---|
| Hubungan | linear | nonlinear |
| Bentuk model | persamaan | aturan bercabang |
| Scaling | tidak wajib | tidak wajib |
| Interpretasi | coefficient | tree / rule |
| Overfitting | relatif lebih rendah | lebih mudah |
| Prediksi | halus / kontinu | bertingkat |

### Linear Regression

$$
\hat{y}=b+w_1x_1+w_2x_2+\cdots+w_nx_n
$$

### Decision Tree

```text
if condition:
    prediction A
else:
    prediction B
```

---

# 15 — Evaluasi Decision Tree Regressor

Metric yang digunakan sama seperti regression lainnya.

## MAE

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
$$

## MSE

$$
MSE =
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

## RMSE

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

Semakin kecil error, semakin dekat prediction terhadap actual.

---

# 16 — Feature Importance

Decision Tree dapat memberikan nilai **feature importance**.

Contoh:

```text
current_a           0.45
operating_hours     0.30
voltage_v           0.15
power_factor        0.07
ambient_temp_c      0.03
```

Interpretasi:

Feature dengan nilai lebih besar lebih banyak berkontribusi terhadap keputusan tree.

### Catatan penting

Feature importance:

- bukan bukti sebab-akibat;
- tidak selalu stabil pada dataset kecil;
- harus dibaca bersama konteks engineering.

---

# 17 — Dari Decision Tree ke Random Forest

Satu Decision Tree dapat tidak stabil.

Solusinya:

```text
Tree 1
Tree 2
Tree 3
...
Tree N
```

Kemudian hasilnya digabungkan.

Inilah ide dasar:

# Random Forest Regressor

Pada regression:

```text
Prediction Tree 1
Prediction Tree 2
Prediction Tree 3
       ↓
     Average
       ↓
Final Prediction
```

Jadi Random Forest merupakan pengembangan dari banyak Decision Tree.

---

# 18 — Alur Decision Tree Regressor

```text
RAW DATA
   ↓
CLEANING
   ↓
FEATURE + TARGET
   ↓
TRAIN / TEST SPLIT
   ↓
DECISION TREE REGRESSOR
   ↓
FIND BEST SPLIT
   ↓
BUILD TREE
   ↓
PREDICTION
   ↓
ACTUAL vs PREDICTED
   ↓
MAE / MSE / RMSE
```

---

# 19 — Ringkasan

Decision Tree Regressor:

1. digunakan untuk target numerik kontinu;
2. membagi data menggunakan aturan keputusan;
3. mencari split yang membuat target dalam node semakin seragam;
4. leaf menghasilkan nilai prediction;
5. mampu menangkap hubungan nonlinear;
6. tidak membutuhkan feature scaling;
7. mudah divisualisasikan;
8. dapat mengalami overfitting jika tree terlalu dalam.

### Ide utama

> **Decision Tree Regressor memprediksi angka dengan membagi data menjadi beberapa wilayah berdasarkan aturan pada feature.**
