# Mission-06B — Multiple Linear Regression on ESP32
## Predicting Machine Power from RPM and Load (MicroPython)

> **Praktikum Machine Learning — Embedded ML tanpa sensor**  
> Perangkat: ESP32 + kabel USB + Thonny/MicroPython. Tidak memerlukan sensor, Wi-Fi, breadboard, atau library `numpy`/`sklearn` di ESP32.  
> **Estimasi waktu:** 80–100 menit.  
> **Prasyarat:** sudah mencoba simple linear regression, memahami train/test, MAE, dan koefisien regresi.

## 1. Tujuan pembelajaran

Setelah praktikum, mahasiswa mampu:

1. Menjelaskan perbedaan simple dan multiple linear regression.
2. Menggunakan parameter hasil training untuk inference pada ESP32.
3. Menghitung parameter regresi dua fitur **langsung pada ESP32** menggunakan least squares.
4. Membandingkan MAE **test set** model satu fitur dan dua fitur.
5. Menguji pengaruh fitur tambahan dan mengenali keterbatasan data serta komputasi pada microcontroller.

## 2. Kasus: Prediksi konsumsi daya mesin

Sebuah mesin hipotetis memiliki dua variabel masukan:

- $x_1$: kecepatan motor (**RPM**).
- $x_2$: beban mesin (**%**, 0–100).
- $y$: daya listrik (**W**), sebagai nilai target.

Seluruh nilai pada praktikum ini adalah **data sintetik**, bukan pengukuran suatu motor nyata. Anggap hubungan daya, putaran, dan beban dalam rentang data sebagai pendekatan linear untuk latihan; model ini **bukan hukum fisika universal** tentang motor.

Persamaan multiple linear regression:

$$
\hat{y} = b_0 + b_1x_1 + b_2x_2
$$

Keterangan:

- $\hat{y}$: daya hasil prediksi (W).
- $b_0$: intercept (W).
- $b_1$: koefisien RPM (W/RPM), ketika beban tetap.
- $b_2$: koefisien beban (W per satu poin persentase), ketika RPM tetap.

> **Catatan:** Koefisien regresi menggambarkan asosiasi dalam dataset, bukan bukti kausal. Intercept juga tidak harus bermakna secara fisik apabila RPM = 0 dan beban = 0 berada di luar rentang data training.

## 3. Dataset

Gunakan dataset *training* berikut (12 sampel) dan *test* berikut (5 sampel). Jangan mencampurkan test set ke proses training.

**Training set**

| RPM | Beban (%) | Daya (W) |
|---:|---:|---:|
| 800 | 20 | 122 |
| 900 | 50 | 170 |
| 1000 | 30 | 153 |
| 1100 | 70 | 205 |
| 1200 | 40 | 182 |
| 1300 | 80 | 236 |
| 1400 | 25 | 190 |
| 1500 | 60 | 236 |
| 1600 | 45 | 225 |
| 1700 | 85 | 276 |
| 1800 | 35 | 239 |
| 1900 | 75 | 284 |

**Test set (tidak digunakan saat training)**

| RPM | Beban (%) | Daya aktual (W) |
|---:|---:|---:|
| 950 | 40 | 164 |
| 1250 | 65 | 213 |
| 1450 | 35 | 203 |
| 1750 | 55 | 258 |
| 1850 | 90 | 298 |

## 4. Persiapan

1. Sambungkan ESP32 ke komputer melalui USB.
2. Pada Thonny, pilih interpreter **MicroPython (ESP32)** dan port serial yang sesuai.
3. Buat berkas baru, misalnya `mission06b.py`, lalu jalankan dengan **F5**.
4. Apabila kode mengandung `input()`, ketik nilainya pada **Shell Thonny**. Gunakan `Ctrl+C` untuk menghentikan program.
5. Jalankan **setiap eksperimen secara terpisah**, atau simpan fungsi dasar pada satu berkas dan tambahkan bagian eksperimen secara berurutan.

> **Penting:** Kita menjalankan perhitungan di ESP32. Laptop dan Thonny hanya bertindak sebagai editor dan terminal serial, bukan sebagai mesin inference.

---

## Eksperimen A — Inference dengan parameter yang sudah tersedia (±15 menit)

Pada tahap ini ESP32 **tidak melakukan training**. Parameter model dua fitur diasumsikan telah dihitung di komputer dari *training set* di atas.

```python
# MicroPython — ESP32
# Parameter referensi dari least squares pada training set
b0 = 28.57969737
b1 = 0.09530851
b2 = 1.02609079


def predict_power(rpm, load):
    return b0 + b1 * rpm + b2 * load


rpm = 1450
load = 35
print("RPM:", rpm)
print("Load (%):", load)
print("Predicted power (W):", round(predict_power(rpm, load), 2))
```

**Tugas A:**

1. Uji `(RPM, beban) = (950, 40)`, `(1450, 35)`, dan `(1850, 90)`.
2. Catat hasil prediksinya.
3. Apa yang berubah ketika hanya RPM dinaikkan, sementara beban tetap?
4. Jelaskan mengapa langkah ini disebut **inference**, bukan training.

---

## Eksperimen B — Training multiple linear regression di ESP32 (±30 menit)

Model mencari tiga parameter $b_0$, $b_1$, dan $b_2$ dengan meminimalkan *sum of squared errors*:

$$
\mathrm{SSE} = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Secara matematis, parameter bisa diperoleh melalui *normal equations*:

$$
(X^T X)\boldsymbol{b} = X^T\boldsymbol{y}
$$

Untuk menjaga skala angka saat menghitung, kode memakai fitur $u = \mathrm{RPM}/1000$ dan $v = \mathrm{beban}/100$ **secara konsisten** ketika training dan inference. Pemakaian skala tetap ini bukan `StandardScaler` yang di-*fit* pada dataset.

Kode berikut adalah **least squares sungguhan**, bukan koefisien yang disetel manual. Karena MicroPython standar tidak memiliki `numpy`, sistem persamaan linear 3×3 diselesaikan dengan eliminasi Gauss ber-*partial pivoting*.

```python
# TRAINING DI ESP32 — MicroPython tanpa numpy
TRAIN = [
    (800, 20, 122), (900, 50, 170), (1000, 30, 153),
    (1100, 70, 205), (1200, 40, 182), (1300, 80, 236),
    (1400, 25, 190), (1500, 60, 236), (1600, 45, 225),
    (1700, 85, 276), (1800, 35, 239), (1900, 75, 284)
]

TEST = [
    (950, 40, 164), (1250, 65, 213), (1450, 35, 203),
    (1750, 55, 258), (1850, 90, 298)
]


def solve_linear(a, b):
    # Gauss-Jordan + partial pivoting untuk matriks kecil
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-10:
            raise ValueError("Fitur tidak independen / matriks singular")
        scale = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= scale
        for row in range(n):
            if row != col:
                factor = m[row][col]
                for j in range(col, n + 1):
                    m[row][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def fit_multiple(data):
    # x = [1, rpm/1000, load/100]
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for rpm, load, power in data:
        x = [1.0, rpm / 1000.0, load / 100.0]
        for i in range(3):
            xty[i] += x[i] * power
            for j in range(3):
                xtx[i][j] += x[i] * x[j]
    return solve_linear(xtx, xty)


def predict_multiple(coef, rpm, load):
    return coef[0] + coef[1] * (rpm / 1000.0) + coef[2] * (load / 100.0)


def mae(data, predict_fn):
    total = 0.0
    for rpm, load, actual in data:
        total += abs(actual - predict_fn(rpm, load))
    return total / len(data)


coef = fit_multiple(TRAIN)
print("Coefficients for scaled features:", coef)
print("Test predictions:")
for rpm, load, actual in TEST:
    pred = predict_multiple(coef, rpm, load)
    print("RPM", rpm, "Load", load, "Actual", actual,
          "Pred", round(pred, 2))

print("Test MAE (W):", round(mae(
    TEST, lambda rpm, load: predict_multiple(coef, rpm, load)
), 4))
```

**Perhatian saat membaca koefisien:** karena fitur pada kode dibagi 1000 dan 100, koefisien yang dicetak tidak sama secara numerik dengan $b_1$ dan $b_2$ dalam satuan asli. Untuk mengembalikannya ke persamaan yang memakai RPM dan beban (%):

$$
 b_1 = \frac{\text{coef}[1]}{1000}, \qquad
 b_2 = \frac{\text{coef}[2]}{100}
$$

**Tugas B:**

1. Catat ketiga parameter hasil training.
2. Cocokkan prediksi *test set* dengan Eksperimen A (selisih kecil akibat presisi numerik dapat terjadi).
3. Laporkan MAE pada data uji, bukan hanya data training.
4. Mengapa regresi dengan dua fitur memerlukan paling sedikit tiga parameter?

---

## Eksperimen C — Bandingkan model satu fitur vs dua fitur (±15 menit)

Sekarang kita membandingkan model **RPM saja** dengan model **RPM + beban** pada *test set yang sama*. Tambahkan kode berikut **setelah Eksperimen B**.

```python
# Simple linear regression: daya = a + b * (rpm/1000)
def fit_rpm_only(data):
    n = len(data)
    xs = [row[0] / 1000.0 for row in data]
    ys = [row[2] for row in data]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y)
                    for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("Nilai RPM tidak bervariasi")
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return intercept, slope


simple_coef = fit_rpm_only(TRAIN)


def predict_simple(rpm, load):
    # load sengaja tidak digunakan oleh model satu fitur
    return simple_coef[0] + simple_coef[1] * (rpm / 1000.0)


simple_mae = mae(TEST, predict_simple)
multiple_mae = mae(TEST, lambda rpm, load:
                   predict_multiple(coef, rpm, load))
print("MAE RPM only (W):", round(simple_mae, 4))
print("MAE RPM + load (W):", round(multiple_mae, 4))
```

**Tugas C:**

1. Model mana yang menghasilkan MAE lebih kecil pada data uji?
2. Apakah penambahan fitur **selalu** meningkatkan generalisasi? Mengapa tidak?
3. Jelaskan mengapa membandingkan error pada data training saja bisa menyesatkan.

> Dataset praktikum ini memang dibuat agar fitur beban memberikan informasi tambahan. Jangan menyimpulkan bahwa dua fitur **selalu** lebih baik untuk setiap dataset.

---

## Eksperimen D — Prediksi interaktif tanpa sensor (±10 menit)

Setelah menjalankan Eksperimen B, tambahkan blok berikut pada akhir file untuk menerima input dari **Shell Thonny**.

```python
while True:
    try:
        rpm = float(input("RPM (800-1900, atau Ctrl+C): "))
        load = float(input("Load % (0-100): "))
        if not (800 <= rpm <= 1900 and 0 <= load <= 100):
            print("Input di luar rentang yang disarankan.")
            continue
        power = predict_multiple(coef, rpm, load)
        print("Predicted power:", round(power, 2), "W")
        print("--------------------------")
    except ValueError:
        print("Masukkan angka yang valid.")
    except KeyboardInterrupt:
        print("Program dihentikan.")
        break
```

**Tugas D:** pilih RPM tetap (misalnya 1500) lalu ubah beban (20%, 50%, 80%). Apa pola perubahan prediksi? Ulangi dengan beban tetap dan RPM berubah.

---

## Eksperimen E — Studi outlier dan batas model (±10–15 menit)

Coba tambahkan satu data **outlier sintetik** pada training set (jalankan dari kode Eksperimen B, sebelum blok `while True`):

```python
train_outlier = TRAIN + [(1400, 50, 500)]
coef_outlier = fit_multiple(train_outlier)
print("MAE test sebelum outlier:", round(mae(
    TEST, lambda r, l: predict_multiple(coef, r, l)), 4))
print("MAE test setelah outlier:", round(mae(
    TEST, lambda r, l: predict_multiple(coef_outlier, r, l)), 4))
```

**Pertanyaan E:** mengapa outlier pada *training set* dapat memengaruhi parameter dan MAE test? Mengapa outlier tidak boleh otomatis dibuang tanpa alasan domain yang jelas?

## 5. Hasil acuan praktikum

Berikut hasil acuan dari least squares pada dataset yang diberikan, dengan presisi tampilan dibulatkan:

| Model | Parameter dengan satuan fitur asli | MAE test |
|---|---|---:|
| RPM saja | $\hat{y} \approx 45.5676 + 0.1216783\,\mathrm{RPM}$ | **13.0012 W** |
| RPM + beban | $\hat{y} \approx 28.5797 + 0.0953085\,\mathrm{RPM} + 1.0260908\,\mathrm{beban}$ | **2.5003 W** |

Nilai dapat berbeda sedikit karena pembulatan dan presisi *floating point*. Acuan ini dihitung menggunakan training set saja; test set hanya dipakai untuk menghitung MAE.

## 6. Lembar hasil praktikum

| Percobaan | Hasil/observasi mahasiswa |
|---|---|
| A — Prediksi RPM=1450, beban=35 | ... W |
| B — $b_0$ | ... |
| B — $b_1$ dalam W/RPM | ... |
| B — $b_2$ dalam W/poin % | ... |
| C — MAE RPM saja | ... W |
| C — MAE RPM + beban | ... W |
| D — Pengaruh beban pada RPM tetap | ... |
| E — Dampak outlier pada MAE test | ... |

## 7. Pertanyaan diskusi dan laporan

1. Jelaskan perbedaan **training di ESP32** (Eksperimen B) dan **inference di ESP32** (Eksperimen A).
2. Mengapa tambahan fitur beban membuat hasil lebih baik **pada dataset ini**? Apakah hasil ini cukup untuk membuat klaim umum?
3. Bagaimana jika RPM dan beban sangat berkorelasi satu sama lain? Apa dampaknya pada interpretasi koefisien?
4. Apa keterbatasan regresi linear saat memprediksi kondisi di luar rentang training (ekstrapolasi)?
5. Menurut Anda, apakah ESP32 selalu harus melakukan training, atau cukup menjalankan inference? Jelaskan trade-off-nya.

