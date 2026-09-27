<!--
GitHub-compatible version.
Rumus menggunakan sintaks matematika yang disederhanakan agar lebih stabil
saat dirender langsung pada halaman Markdown GitHub.
-->

# 1 — Multiple Feature Regression

## Apa itu Multiple Feature Regression?

Multiple Feature Regression digunakan ketika sebuah **target numerik** diprediksi menggunakan **lebih dari satu feature**.

### Contoh pada sistem kelistrikan

Feature:

- Voltage
- Current
- Operating Hours
- Temperature
- Power Factor

Target:

- **Energy Consumption (kWh)**

Secara umum:

$$
\hat{y}=b+w_1x_1+w_2x_2+\cdots+w_nx_n
$$

---

# 2 — Persamaan Multiple Linear Regression

Untuk $n$ buah feature:

**Persamaan Multiple Linear Regression:**

$$
\hat{y}=b+\sum_{j=1}^{n} w_jx_j
$$

atau:

$$
\hat{y}=b+w_1x_1+w_2x_2+w_3x_3+\cdots+w_nx_n
$$

### Keterangan

| Simbol | Arti |
|---|---|
| $\hat{y}$ | nilai hasil prediksi |
| $x_j$ | feature ke-$j$ |
| $w_j$ | weight / coefficient feature ke-$j$ |
| $b$ | bias / intercept |
| $n$ | jumlah feature |

### Ide utama

Model berusaha mencari nilai:

$$
w_1,w_2,\ldots,w_n,b
$$

yang membuat hasil prediction sedekat mungkin dengan nilai actual.

---

# 3 — Contoh pada Sistem Kelistrikan

Misalkan kita ingin memprediksi konsumsi energi:

$$
\hat{E}=b+w_1V+w_2I+w_3H+w_4T+w_5PF
$$

dengan:

Keterangan:

- $E$ = Energy Consumption (kWh)
- $V$ = Voltage (V)
- $I$ = Current (A)
- $H$ = Operating Hours (hour)
- $T$ = Ambient Temperature (°C)
- $PF$ = Power Factor

Misalnya setelah training diperoleh:

$$
\hat{E}=-12+0.02V+1.25I+1.60H+0.05T+3.2PF
$$

Untuk kondisi:

$$
V=220,\quad
I=8,\quad
H=7,\quad
T=30,\quad
PF=0.9
$$

maka:

$$
\hat{E}=-12+0.02(220)+1.25(8)+1.60(7)+0.05(30)+3.2(0.9)
$$

> Angka coefficient pada contoh ini hanya digunakan untuk menjelaskan konsep.

---

# 4 — Apa Arti Weight?

Perhatikan:

$$
\hat{y}=b+w_1x_1+w_2x_2+\cdots
$$

Weight menunjukkan bagaimana perubahan suatu feature berhubungan dengan perubahan prediction, **ketika feature lainnya dianggap tetap**.

Misalnya:

$$
w_{\text{current}}=1.25
$$

Secara sederhana dapat dibaca:

> Ketika current meningkat 1 A, prediction energy meningkat sekitar 1.25 kWh, dengan feature lainnya tetap.

Jika:

$$
w_j>0
$$

maka peningkatan feature cenderung meningkatkan prediction.

Jika:

$$
w_j<0
$$

maka peningkatan feature cenderung menurunkan prediction.

Jika:

$$
w_j\approx0
$$

maka kontribusi linear feature tersebut terhadap prediction relatif kecil.

### Catatan penting

**Coefficient besar tidak otomatis berarti feature paling penting**, karena masing-masing feature dapat memiliki skala dan satuan berbeda.

---

# 5 — Bias / Intercept

Persamaan:

$$
\hat{y}=b+w_1x_1+\cdots+w_nx_n
$$

Nilai:

$$
b
$$

disebut **bias** atau **intercept**.

Secara matematis, bias merupakan prediction model ketika:

$$
x_1=x_2=\cdots=x_n=0
$$

sehingga:

$$
\hat{y}=b
$$

Namun secara engineering, kondisi semua feature sama dengan nol belum tentu merupakan kondisi yang benar-benar terjadi.

Jadi bias terutama berfungsi sebagai **konstanta model**.

---

# 6 — Actual, Predicted, dan Error

Untuk setiap sample:

$$
y_i=\text{actual value}
$$

sedangkan:

$$
\hat{y}_i=\text{predicted value}
$$

Error atau residual:

**Residual / error:**

$$
e_i=y_i-\hat{y}_i
$$

### Contoh

$$
y_i=14.2\text{ kWh}
$$

$$
\hat{y}_i=13.8\text{ kWh}
$$

maka:

$$
e_i=14.2-13.8=0.4\text{ kWh}
$$

---

# 7 — Bagaimana Model Belajar?

Linear Regression mencari coefficient yang membuat total error sekecil mungkin.

Pendekatan yang umum digunakan adalah **Least Squares**.

Model meminimalkan:

**Least Squares objective:**

$$
\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

dengan:

$$
m=\text{jumlah sample}
$$

Karena:

$$
\hat{y}_i=
b+w_1x_{i1}+w_2x_{i2}+\cdots+w_nx_{in}
$$

maka model sebenarnya mencari:

$$
b,w_1,w_2,\ldots,w_n
$$

yang menghasilkan **sum of squared errors** paling kecil.

---

# 8 — Bentuk Matriks

Multiple Linear Regression dapat ditulis:

**Bentuk matriks:**

$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b
$$

Untuk beberapa sample, bentuk ringkasnya adalah:

$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b
$$

dengan:

- $\mathbf{X}$ = matriks feature
- $\mathbf{w}$ = vektor weight
- $b$ = bias / intercept
- $\hat{\mathbf{y}}$ = vektor hasil prediction

Secara sederhana:

**Dataset → Matrix X → Weight → Prediction**

Setiap kolom pada $\mathbf{X}$ mewakili satu feature.

---

# 9 — Evaluasi Model: MAE

## MAE — Mean Absolute Error

$$
MAE =
\frac{1}{m}
\sum_{i=1}^{m}
|y_i-\hat{y}_i|
$$

### Interpretasi

MAE menunjukkan rata-rata besar kesalahan prediction.

Contoh:

Jika:

$$
MAE=0.5\text{ kWh}
$$

maka secara rata-rata prediction meleset sekitar **0.5 kWh**.

---

# 10 — Evaluasi Model: MSE dan RMSE

## MSE — Mean Squared Error

$$
MSE =
\frac{1}{m}
\sum_{i=1}^{m}
(y_i-\hat{y}_i)^2
$$

Error besar mendapatkan penalti lebih kuat karena dikuadratkan.

Satuan MSE:

$$
\text{kWh}^2
$$

## RMSE — Root Mean Squared Error

$$
RMSE =
\sqrt{
\frac{1}{m}
\sum_{i=1}^{m}
(y_i-\hat{y}_i)^2
}
$$

Keunggulan RMSE:

$$
\text{satuan RMSE}=\text{satuan target}
$$

Jika target dalam kWh:

$$
RMSE\rightarrow \text{kWh}
$$

---

# 11 — Konsep Utama yang Perlu Diingat

```text
Multiple Features
       ↓
       X
       ↓
Multiple Linear Regression
       ↓
Learn w1, w2, ..., wn and b
       ↓
       ŷ
       ↓
Actual vs Predicted
       ↓
MAE / MSE / RMSE
```

### Pesan utama

> **Multiple regression tidak mencari satu hubungan input-output, tetapi mencari kombinasi beberapa feature yang bersama-sama dapat menjelaskan atau memprediksi target.**

---

# 12 — Ringkasan

Pada Multiple Feature Regression:

1. Kita memiliki lebih dari satu feature.
2. Model mempelajari coefficient untuk setiap feature.
3. Semua feature dikombinasikan untuk menghasilkan prediction.
4. Prediction dibandingkan dengan actual value.
5. Error dievaluasi menggunakan MAE, MSE, dan RMSE.
6. Hasil model tetap harus ditafsirkan dalam konteks engineering.

### Persamaan utama

**Persamaan Multiple Linear Regression:**

$$
\hat{y}=b+\sum_{j=1}^{n} w_jx_j
$$

### Evaluasi

$$
MAE,\qquad MSE,\qquad RMSE
$$
