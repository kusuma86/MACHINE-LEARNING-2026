# Praktikum Edge Machine Learning pada Raspberry Pi Zero 2 W

**Topik:** Random Forest, inferensi model, dan dashboard web lokal  
**Perangkat:** Raspberry Pi Zero 2 W  
**Sistem operasi:** Raspberry Pi OS Lite / Debian 13 (Trixie), ARM64  
**Bahasa pemrograman:** Python 3  
**Pustaka:** NumPy, SciPy, scikit-learn, joblib, Flask, dan Waitress

## Tujuan

Setelah menyelesaikan praktikum ini, mahasiswa mampu:

1. Membuat dan mengaktifkan *Python virtual environment* pada Raspberry Pi.
2. Menginstal pustaka Machine Learning di dalam virtual environment.
3. Melatih dan menyimpan model klasifikasi Random Forest sederhana.
4. Memuat model tersimpan dan melakukan inferensi tanpa melatih ulang model.
5. Membuat aplikasi web lokal yang menampilkan hasil prediksi dan waktu inferensi.
6. Mengakses aplikasi Raspberry Pi menggunakan browser laptop melalui jaringan lokal.

> **Catatan:** Dataset dalam praktikum ini adalah data sintetik berukuran sangat kecil. Tujuannya untuk mempelajari alur kerja (*workflow*), **bukan** menghasilkan model yang siap digunakan untuk diagnosis kerusakan motor nyata. Pada tahap pengembangan lebih lanjut, model sebaiknya dilatih dan dievaluasi menggunakan dataset yang lebih besar di laptop, kemudian dipindahkan ke Raspberry Pi untuk inferensi.

---

## 1. Persiapan

Pastikan:

- Raspberry Pi telah menjalankan Raspberry Pi OS Lite dan dapat diakses melalui terminal atau SSH.
- Raspberry Pi memiliki koneksi internet untuk instalasi pustaka.
- Laptop dan Raspberry Pi terhubung ke **jaringan lokal yang sama** ketika mengakses dashboard.
- Perintah dijalankan dari terminal Raspberry Pi, bukan terminal laptop.

Periksa versi Python:

```bash
python3 --version
```

Instal paket pendukung virtual environment:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

## 2. Membuat dan mengaktifkan virtual environment

Buat direktori proyek:

```bash
mkdir -p ~/edge-ml
cd ~/edge-ml
```

Buat virtual environment dengan nama `.venv`:

```bash
python3 -m venv .venv
```

Aktifkan:

```bash
source .venv/bin/activate
```

Verifikasi bahwa Python yang aktif berasal dari proyek:

```bash
which python
```

Contoh hasil (nama pengguna dapat berbeda):

```text
/home/otomasi/edge-ml/.venv/bin/python
```

Terminal biasanya menampilkan awalan `(.venv)` setelah aktivasi.

> **Catatan:** Folder `.venv` tersembunyi karena namanya diawali titik. Gunakan `ls -la` untuk melihatnya di terminal. Jika memakai WinSCP, *refresh* tampilan file setelah membuat file baru.
>
> Jika membuka terminal/SSH baru, aktifkan kembali environment dengan `cd ~/edge-ml` lalu `source .venv/bin/activate`.

## 3. Menginstal pustaka

Pastikan virtual environment sudah aktif, kemudian jalankan:

```bash
python -m pip install --no-cache-dir --only-binary=:all: scikit-learn joblib flask waitress
```

`scikit-learn` akan memasang dependensi seperti NumPy dan SciPy. Opsi `--only-binary=:all:` mencegah kompilasi paket dari *source* pada Raspberry Pi. Jika paket binary tidak tersedia, instalasi akan berhenti dan perlu diperiksa kompatibilitas sistem/Python-nya.

Verifikasi instalasi:

```bash
python -c "import numpy, scipy, sklearn, joblib, flask, waitress; print('NumPy:', numpy.__version__); print('SciPy:', scipy.__version__); print('scikit-learn:', sklearn.__version__); print('Semua pustaka berhasil diimpor')"
```

Nomor versi bisa berbeda, asalkan tidak muncul pesan kesalahan.

## 4. Melatih dan menyimpan model Random Forest

Dalam direktori `~/edge-ml`, buat file:

```bash
nano test_rf.py
```

Isi dengan kode berikut:

```python
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Fitur berurutan: [arus (A), getaran (m/s^2), RPM]
# Dataset sintetik untuk pengenalan alur ML.
X = np.array([
    [0.20, 0.5, 100],
    [0.22, 0.6, 102],
    [0.19, 0.4, 98],
    [0.21, 0.5, 101],

    [0.55, 1.2, 82],
    [0.60, 1.4, 79],
    [0.58, 1.3, 80],
    [0.57, 1.2, 81],

    [0.35, 4.5, 95],
    [0.38, 5.0, 92],
    [0.36, 4.8, 94],
    [0.37, 4.6, 93],
])

y = [
    "Normal", "Normal", "Normal", "Normal",
    "Load", "Load", "Load", "Load",
    "Mechanical", "Mechanical", "Mechanical", "Mechanical",
]

# Latih Random Forest sederhana.
model = RandomForestClassifier(
    n_estimators=20,
    max_depth=4,
    n_jobs=1,
    random_state=42,
)
model.fit(X, y)

# Simpan model pada folder yang sama dengan file Python ini.
model_path = Path(__file__).resolve().parent / "model_rf.joblib"
joblib.dump(model, model_path)

# Muat kembali model sebagai simulasi deployment.
model = joblib.load(model_path)

# Data uji: [arus, getaran, RPM].
sample = np.array([[0.23, 0.7, 99]])

# Satu prediksi untuk warm-up.
model.predict(sample)

start = time.perf_counter()
prediction = model.predict(sample)
latency_ms = (time.perf_counter() - start) * 1000

print("===== EDGE ML TEST =====")
print("Model      : Random Forest")
print("Prediction :", prediction[0])
print(f"Latency    : {latency_ms:.3f} ms")
print("Status     : Success")
print("Model file :", model_path)
```

Simpan file di `nano` menggunakan **Ctrl+O**, **Enter**, lalu **Ctrl+X**.

Jalankan:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_rf.py
```

Contoh keluaran:

```text
===== EDGE ML TEST =====
Model      : Random Forest
Prediction : Normal
Latency    : ... ms
Status     : Success
Model file : /home/otomasi/edge-ml/model_rf.joblib
```

Periksa bahwa file model sudah dibuat:

```bash
ls -lh model_rf.joblib
```

**Penjelasan:** `model.fit()` melatih model, `joblib.dump()` menyimpannya, dan `joblib.load()` memuat model yang sudah tersimpan. Nilai latency dapat berubah antarperangkat dan antarpercobaan.

## 5. Inferensi menggunakan model tersimpan

Pada tahap ini model **tidak dilatih ulang**. Kita cukup memuat `model_rf.joblib` lalu menjalankan `predict()`.

Buat file:

```bash
nano inference_rf.py
```

Masukkan kode berikut:

```python
import time
from pathlib import Path

import joblib
import numpy as np

# Muat model hasil training di Langkah 4.
model_path = Path(__file__).resolve().parent / "model_rf.joblib"
model = joblib.load(model_path)

# Urutan fitur harus sama seperti saat training:
# [arus (A), getaran (m/s^2), RPM].
X = np.array([[0.23, 0.7, 99]])

# Warm-up: hasil 10 prediksi awal tidak dicatat.
for _ in range(10):
    model.predict(X)

latencies_ms = []
for _ in range(100):
    start = time.perf_counter()
    prediction = model.predict(X)
    latencies_ms.append((time.perf_counter() - start) * 1000)

print("===== EDGE INFERENCE =====")
print("Prediction :", prediction[0])
print(f"Average    : {np.mean(latencies_ms):.3f} ms")
print(f"Median     : {np.median(latencies_ms):.3f} ms")
print(f"P95        : {np.percentile(latencies_ms, 95):.3f} ms")
print(f"Minimum    : {np.min(latencies_ms):.3f} ms")
print(f"Maximum    : {np.max(latencies_ms):.3f} ms")
```

Jalankan:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python inference_rf.py
```

Program menjalankan 100 prediksi untuk memperlihatkan variasi waktu inferensi. **Average** adalah rata-rata, **Median** adalah nilai tengah, dan **P95** adalah nilai di bawah atau sama dengan yang dicapai sekitar 95% pengukuran. Pengukuran hanya mencakup `predict()`, bukan waktu memulai Python atau memuat model.

## 6. Membuat aplikasi web dengan Flask

Sekarang kita akan menggunakan model yang sama untuk menerima input dari browser dan mengembalikan hasil prediksi.

Buat file:

```bash
nano app.py
```

Masukkan kode berikut:

```python
import time
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Model dimuat satu kali ketika aplikasi mulai berjalan.
model_path = Path(__file__).resolve().parent / "model_rf.joblib"
model = joblib.load(model_path)

HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Edge ML Laboratory</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 520px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.5;
        }
        input, button {
            width: 100%;
            padding: 10px;
            margin: 6px 0 14px;
            box-sizing: border-box;
        }
        button { cursor: pointer; }
        .result {
            background: #f1f5f9;
            padding: 14px;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <h2>Edge ML Laboratory</h2>
    <p>Raspberry Pi Zero 2 W — Random Forest</p>

    <form method="POST">
        <label>Arus (A)</label>
        <input type="number" step="any" name="current" value="0.23" required>

        <label>Getaran (m/s²)</label>
        <input type="number" step="any" name="vibration" value="0.7" required>

        <label>RPM</label>
        <input type="number" step="any" name="rpm" value="99" required>

        <button type="submit">Predict</button>
    </form>

    {% if prediction is not none %}
    <div class="result">
        <h3>Prediction: {{ prediction }}</h3>
        {% if latency is not none %}
        <p>Inference latency: {{ latency }} ms</p>
        {% endif %}
    </div>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    latency = None

    if request.method == "POST":
        try:
            current = float(request.form["current"])
            vibration = float(request.form["vibration"])
            rpm = float(request.form["rpm"])

            # Urutan fitur harus sesuai data training.
            X = np.array([[current, vibration, rpm]])
            if not np.isfinite(X).all():
                raise ValueError("Input harus berupa angka berhingga")

            start = time.perf_counter()
            prediction = str(model.predict(X)[0])
            latency = round((time.perf_counter() - start) * 1000, 3)

        except (ValueError, KeyError):
            prediction = "Input tidak valid"

    return render_template_string(
        HTML,
        prediction=prediction,
        latency=latency,
    )
```

**Cara kerja:** Halaman HTML ditampilkan melalui Flask. Ketika tombol **Predict** ditekan, browser mengirim tiga nilai fitur ke Raspberry Pi. Raspberry Pi memanggil model Random Forest, lalu mengirimkan hasil prediksi dan waktu inferensi untuk ditampilkan di halaman web.

## 7. Menjalankan web server dengan Waitress

Pastikan masih berada di `~/edge-ml` dan virtual environment aktif. Jalankan:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
waitress-serve --listen=0.0.0.0:5000 --threads=2 app:app
```

Biarkan terminal ini tetap berjalan selama aplikasi digunakan.

Pada **terminal SSH kedua**, periksa alamat IP Raspberry Pi:

```bash
hostname -I
```

Misalnya alamat IP Raspberry Pi adalah `192.168.1.100`. Buka browser di laptop dan kunjungi:

```text
http://192.168.1.100:5000
```

**Ganti `192.168.1.100` dengan alamat IP Raspberry Pi yang sebenarnya.** Laptop harus berada di jaringan lokal yang sama, dan jaringan harus mengizinkan komunikasi antarklien.

Lakukan pengujian:

1. Buka halaman **Edge ML Laboratory** dari laptop.
2. Isi arus `0.23`, getaran `0.7`, dan RPM `99`.
3. Klik **Predict**.
4. Amati hasil klasifikasi dan *inference latency* yang ditampilkan.
5. Coba ubah nilainya untuk melihat perubahan hasil prediksi.

Untuk **menghentikan server**, kembali ke terminal Waitress dan tekan **Ctrl+C**.

> Server dijalankan **manual**, bukan otomatis saat boot. Ini memudahkan penggunaan Raspberry Pi secara bergantian dalam praktikum. Aplikasi ini ditujukan untuk jaringan lokal yang tepercaya; **jangan membuka port 5000 langsung ke internet** karena belum ada autentikasi.

---

## 8. Struktur proyek akhir

```text
edge-ml/
├── .venv/                # Virtual environment (tersembunyi)
├── test_rf.py            # Training dan penyimpanan model
├── model_rf.joblib       # File model Random Forest
├── inference_rf.py       # Inferensi dan pengukuran latency
└── app.py                # Flask + antarmuka web
```

**Urutan menjalankan praktikum dari awal:**

```bash
cd ~/edge-ml
source .venv/bin/activate

# Jalankan sekali untuk menghasilkan model
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_rf.py

# Uji model tersimpan
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python inference_rf.py

# Jalankan aplikasi web lokal
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
waitress-serve --listen=0.0.0.0:5000 --threads=2 app:app
```

## 9. Catatan untuk mahasiswa

- **Training** menggunakan `model.fit()`; **inferensi** menggunakan `model.predict()`.
- File `.joblib` memungkinkan model digunakan kembali tanpa training ulang. Muat hanya model dari sumber yang tepercaya karena format ini dapat menjalankan kode saat dibaca.
- Urutan fitur **arus, getaran, RPM** harus konsisten pada saat training maupun inferensi.
- `OPENBLAS_NUM_THREADS=1` dan `OMP_NUM_THREADS=1` membatasi thread pada pustaka komputasi tertentu agar pengujian awal lebih terkendali.
- Model dari 12 data sintetik **tidak cocok untuk keputusan operasional nyata**. Fokus praktikum adalah alur deployment Machine Learning pada perangkat edge.
- Pada praktikum berikutnya, training dapat dipindahkan ke laptop, sedangkan Raspberry Pi hanya menyimpan model dan menjalankan inferensi serta dashboard web.

**Hasil akhir:** mahasiswa berhasil membangun aplikasi Edge ML yang dapat diakses dari browser laptop, sementara seluruh proses inferensi model Random Forest berlangsung pada Raspberry Pi Zero 2 W.
