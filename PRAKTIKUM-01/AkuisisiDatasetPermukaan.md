# MODUL PRAKTIKUM 01

## Akuisisi dan Penyusunan Dataset Citra untuk Klasifikasi Permukaan Lingkungan

**Mata kuliah:** Machine Learning  
**Durasi:** 1 pertemuan (180 menit)  
**Bentuk kerja:** Kelompok, **2 mahasiswa per kelompok** (jika jumlah peserta ganjil, satu kelompok dapat beranggotakan 3 mahasiswa)  
**Lokasi:** Lingkungan kampus  
**Luaran utama:** Dataset citra berlabel, metadata `.csv`, dokumentasi singkat, dan berkas `.zip`  
**Batasan:** Praktikum ini **tidak mencakup pelatihan ataupun evaluasi model Machine Learning**.

---

## A. Pendahuluan

Dalam penerapan *Machine Learning*, model hanya dapat belajar dari data yang tersedia. Karena itu, kualitas data yang dikumpulkan, kejelasan label, keragaman sampel, dan keterlacakan sumber data berpengaruh terhadap kualitas model yang akan dibangun. Dataset yang tampak besar belum tentu berkualitas apabila banyak gambar duplikat, label salah, atau seluruh foto berasal dari satu titik yang sama.

Pada praktikum ini, mahasiswa akan membuat dataset citra secara mandiri menggunakan kamera smartphone. Objek yang diamati adalah **enam jenis permukaan lingkungan kampus**: **rumput, paving, aspal, tanah, batu, dan semen**. Dataset ini direncanakan untuk digunakan pada praktikum berikutnya dalam membangun model klasifikasi citra.

Kegiatan ini juga terkait dengan aplikasi rekayasa, misalnya pengenalan jenis medan pada robot bergerak, inspeksi lingkungan, dan sistem persepsi berbasis kamera. Namun, dataset yang diperoleh dari foto smartphone belum otomatis layak untuk aplikasi robot nyata; kemampuan generalisasi perlu diuji pada tahap berikutnya.

### Tujuan pembelajaran

Setelah praktikum, mahasiswa diharapkan mampu:

1. Menjelaskan perbedaan **citra**, **fitur visual**, **kelas/label**, dan **dataset**.
2. Menetapkan definisi enam kelas permukaan secara konsisten.
3. Merancang dan menjalankan prosedur pengambilan gambar menggunakan smartphone.
4. Mengumpulkan gambar yang bervariasi tanpa menghasilkan banyak duplikat atau gambar yang tidak relevan.
5. Menyusun folder dataset, memberikan label, dan membuat metadata yang dapat ditelusuri.
6. Melakukan pemeriksaan dasar terhadap jumlah, keseimbangan, dan kualitas gambar.

---

## B. Alat dan Bahan

| No. | Alat/bahan | Keterangan |
|---|---|---|
| 1 | Smartphone berkamera | Minimal satu unit per mahasiswa; gunakan kamera utama jika memungkinkan |
| 2 | Laptop/komputer | Untuk menyalin, mengelompokkan, dan memeriksa gambar |
| 3 | Aplikasi kamera bawaan | Tidak memerlukan aplikasi berbayar atau pemrograman |
| 4 | File manager | Untuk membuat folder dan mengganti nama file |
| 5 | Aplikasi spreadsheet/editor teks | Excel, Google Sheets, LibreOffice Calc, atau editor teks untuk `metadata.csv` |
| 6 | Lingkungan kampus | Area dengan permukaan rumput, paving, aspal, tanah, batu, dan semen |
| 7 | Koneksi internet (opsional) | Untuk mengunggah hasil; pengambilan foto dapat dilakukan tanpa internet |

**Keselamatan:** Pengambilan foto dilakukan hanya pada lokasi yang aman dan diizinkan. Jangan memotret di tengah jalan aktif, memanjat, memasuki area terlarang, atau menghalangi pengguna jalan. Hindari menyertakan wajah orang, pelat kendaraan, atau informasi pribadi dalam gambar.

---

## C. Deskripsi Kasus dan Kelas

**Permasalahan:** Menyiapkan dataset untuk model yang nantinya menerima sebuah foto permukaan dan memprediksi salah satu dari enam kelas.

| Kode label | Nama kelas | Kriteria yang digunakan |
|---|---|---|
| `rumput` | Rumput | Area didominasi helaian rumput yang tumbuh di permukaan tanah |
| `paving` | Paving | Blok paving/bata beton tersusun, dengan pola sambungan antarblok terlihat |
| `aspal` | Aspal | Lapisan permukaan jalan atau halaman berbahan aspal |
| `tanah` | Tanah | Permukaan tanah terbuka; tidak didominasi rumput atau kerikil |
| `batu` | Batu | Permukaan yang didominasi batu alam, kerikil, atau bebatuan |
| `semen` | Semen | Permukaan cor beton/semen yang relatif kontinu, tanpa pola sambungan paving |

**Aturan pelabelan:**

- Foto diberi label berdasarkan **material permukaan dominan**, bukan berdasarkan lokasi atau warna semata.
- Permukaan campuran (misalnya rumput dan paving masing-masing menempati sebagian besar gambar) **tidak digunakan**; cari area yang lebih jelas.
- Hindari foto dengan objek lain yang dominan, misalnya sepatu, kendaraan, manusia, atau pot tanaman.
- Jika sulit membedakan aspal, semen, dan batu, diskusikan berdasarkan definisi kelas di atas. Foto yang tetap ambigu dikeluarkan dari dataset.
- Istilah `semen` digunakan sebagai nama kelas praktikum untuk permukaan beton/cor yang kontinu, bukan untuk membedakan komposisi bahan secara laboratoris.

---

## D. Target Dataset dan Strategi Akuisisi

### 1. Jumlah data

**Target setiap kelompok: 180 gambar valid**, terdiri dari 30 gambar untuk masing-masing dari enam kelas.

| Kelas | Mahasiswa A | Mahasiswa B | Total |
|---|---:|---:|---:|
| Rumput | 15 | 15 | 30 |
| Paving | 15 | 15 | 30 |
| Aspal | 15 | 15 | 30 |
| Tanah | 15 | 15 | 30 |
| Batu | 15 | 15 | 30 |
| Semen | 15 | 15 | 30 |
| **Total** | **90** | **90** | **180** |

> **Catatan:** Jumlah ini adalah target operasional praktikum, **bukan jaminan bahwa dataset cukup untuk menghasilkan model klasifikasi yang akurat**. Prioritaskan gambar valid dan keragaman sumber daripada mengejar jumlah dengan foto hampir identik. Bila suatu kelas tidak dapat ditemukan secara aman, laporkan kendalanya kepada dosen; **jangan membuat label fiktif**.

### 2. Keragaman sumber gambar

Untuk **setiap kelas**, usahakan memperoleh gambar dari **minimal 3 lokasi/petak permukaan yang berbeda**, misalnya `rumput_L01`, `rumput_L02`, dan `rumput_L03`. Target pembagian sederhana adalah sekitar **10 gambar per lokasi**. Lokasi tidak harus jauh, tetapi harus berupa petak/area yang berbeda, bukan 30 foto hampir identik dari satu titik kamera.

Kedua mahasiswa harus memotret **semua kelas**, bukan membagi kelas menurut pemilik smartphone. Hal ini mengurangi risiko kelas tertentu selalu berkaitan dengan satu kamera tertentu.

### 3. Protokol pemotretan

1. Gunakan kamera utama pada mode foto biasa dan **jangan gunakan filter kecantikan, efek warna, atau penyuntingan AI**.
2. Pegang smartphone dengan stabil; pastikan permukaan tampak jelas dan tidak buram.
3. Ambil foto dengan **permukaan memenuhi sebagian besar bingkai** (disarankan sekitar 80% atau lebih).
4. Utamakan arah kamera **menghadap ke bawah / hampir tegak lurus permukaan**. Beberapa variasi sudut ringan diperbolehkan, tetapi perlakukan seluruh kelas secara serupa.
5. Ambil gambar dari **bagian permukaan yang berbeda**. Jangan membuat variasi hanya dengan menekan tombol kamera berulang-ulang tanpa berpindah posisi.
6. Usahakan mengambil gambar pada kondisi terang/teduh yang wajar; hindari foto terlalu gelap, overexposed, atau tertutup bayangan tubuh sendiri.
7. Jangan mengunduh gambar dari internet dan jangan memakai gambar buatan AI. **Semua foto wajib hasil pengambilan langsung oleh mahasiswa.**
8. **Jangan menggabungkan beberapa foto menjadi kolase**, serta jangan memperbanyak data melalui *copy-paste*, rotasi, atau *augmentation* pada pertemuan ini.
9. Bila tersedia opsi penyimpanan, gunakan `.jpg`/`.jpeg`. Jika kamera menyimpan `.heic`, konversikan ke `.jpg` sebelum dikumpulkan tanpa mengubah label atau isi gambar.

**Tips:** Ambil beberapa foto cadangan (misalnya 10–15% lebih banyak), kemudian seleksi hingga diperoleh 180 gambar valid. Mengambil ratusan gambar tambahan tidak diwajibkan.

---

## E. Langkah Kerja (180 Menit)

| Tahap | Waktu | Aktivitas | Bukti/hasil |
|---|---:|---|---|
| 1. Orientasi dan perencanaan | 20 menit | Pahami enam kelas, kriteria foto valid, keselamatan, dan pembagian tugas | Rencana area pengambilan gambar |
| 2. Survei singkat lokasi | 15 menit | Temukan contoh keenam kelas dan tentukan minimal 3 petak/lokasi per kelas bila memungkinkan | Daftar `location_id` |
| 3. Akuisisi gambar | 80 menit | Kedua mahasiswa memotret seluruh kelas sesuai protokol | Foto mentah dari smartphone |
| 4. Seleksi dan organisasi | 35 menit | Hapus foto buram, ambigu, duplikat; beri nama; pindahkan ke folder per kelas | 180 gambar valid dan berlabel |
| 5. Metadata dan validasi | 20 menit | Lengkapi `metadata.csv`, cek jumlah file, label, dan lokasi | CSV konsisten dengan folder |
| 6. Pengemasan dan pengumpulan | 10 menit | Buat ringkasan, kompres folder ke ZIP, kumpulkan | Berkas ZIP final |
| **Total** | **180 menit** | | |

### Langkah 1 — Pembentukan kelompok dan rencana pengambilan foto

1. Bentuk pasangan mahasiswa.
2. Tentukan identitas kelompok, misalnya `K05A_G01`.
3. Pahami definisi enam kelas pada Bagian C.
4. Bagi tanggung jawab **jumlah foto**, bukan pembagian kelas: masing-masing mahasiswa menargetkan 15 gambar per kelas.
5. Tentukan area yang aman untuk mencari berbagai jenis permukaan.

### Langkah 2 — Survei permukaan

1. Temukan keenam kelas di lingkungan kampus.
2. Untuk setiap kelas, identifikasi hingga tiga petak/lokasi berbeda yang aman dijangkau.
3. Catat kode lokasi, misalnya `R01`, `R02`, `R03` untuk rumput; `P01` untuk paving; dan seterusnya. Kode yang sama hanya digunakan untuk area fisik yang sama.
4. Bila salah satu jenis permukaan sulit ditemukan, segera konsultasikan kepada dosen dan catat keterbatasannya.

### Langkah 3 — Pengambilan citra

1. Setiap mahasiswa mengambil foto untuk seluruh enam kelas.
2. Pastikan area utama dalam foto adalah permukaan yang sesuai label.
3. Berpindahlah di dalam satu petak dan antarpetak untuk memperoleh variasi tekstur yang nyata.
4. Gunakan metode yang relatif konsisten antar kelas; hindari misalnya memotret aspal dari sangat jauh tetapi rumput dari jarak sangat dekat.
5. Lakukan pemeriksaan cepat pada layar smartphone untuk mendeteksi gambar kabur atau salah sasaran.

### Langkah 4 — Seleksi, penamaan, dan folder

Pilih gambar yang layak. Hapus dari dataset final gambar yang buram, tidak jelas labelnya, terpotong oleh objek lain, atau terlalu mirip dengan gambar lain.

Gunakan penamaan **unik dan konsisten**, misalnya:

- `rumput_001.jpg`
- `rumput_002.jpg`
- `paving_001.jpg`
- `aspal_001.jpg`

Buat struktur folder berikut:

```text
K05A_G01_DatasetPermukaan/
├── images/
│   ├── rumput/
│   │   ├── rumput_001.jpg
│   │   └── ...
│   ├── paving/
│   ├── aspal/
│   ├── tanah/
│   ├── batu/
│   └── semen/
├── metadata.csv
└── README.md
```

**Penting:** Jangan membuat folder `train`, `validation`, atau `test` pada pertemuan pertama. Pembagian dataset untuk pembelajaran model akan dilakukan di praktikum berikutnya berdasarkan informasi sumber/lokasi.

### Langkah 5 — Membuat metadata CSV

Buat satu baris untuk **setiap foto**. Kolom wajib:

| Kolom | Makna | Contoh |
|---|---|---|
| `filename` | Path relatif dari folder utama dataset | `images/rumput/rumput_001.jpg` |
| `label` | Salah satu dari enam nama kelas | `rumput` |
| `location_id` | Kode petak/area asal foto | `R01` |
| `photographer_id` | Kode mahasiswa pemotret, tanpa data pribadi | `A` |
| `lighting` | Kondisi pencahayaan sederhana | `terang` atau `teduh` |
| `angle` | Sudut pengambilan secara kategoris | `atas` atau `miring` |

Contoh isi `metadata.csv`:

```csv
filename,label,location_id,photographer_id,lighting,angle
images/rumput/rumput_001.jpg,rumput,R01,A,terang,atas
images/rumput/rumput_002.jpg,rumput,R02,B,teduh,atas
images/paving/paving_001.jpg,paving,P01,A,terang,miring
images/aspal/aspal_001.jpg,aspal,A01,B,teduh,atas
images/tanah/tanah_001.jpg,tanah,T01,A,terang,atas
images/batu/batu_001.jpg,batu,B01,B,terang,atas
images/semen/semen_001.jpg,semen,S01,A,teduh,atas
```

**Catatan:** Contoh di atas hanya menunjukkan format; file sebenarnya harus mencakup **seluruh 180 foto final** (satu baris per foto). Kolom `filename` harus benar-benar menunjuk ke file yang ada. Jika memakai spreadsheet, ekspor dalam format **CSV UTF-8**.

### Langkah 6 — Validasi dataset

Lakukan pemeriksaan berikut sebelum pengumpulan:

- [ ] Ada **enam folder kelas** dengan nama persis sesuai ketentuan.
- [ ] Masing-masing kelas berisi **30 foto valid** (target total 180 foto).
- [ ] Setiap mahasiswa telah berkontribusi pada **semua kelas**.
- [ ] Tiap kelas memiliki sumber dari beberapa petak/lokasi berbeda (target minimal 3 lokasi).
- [ ] Tidak ada gambar dari internet, gambar AI, duplikat, kolase, atau hasil *augmentation*.
- [ ] Tidak ada gambar yang jelas-jelas salah label atau didominasi objek nonpermukaan.
- [ ] `metadata.csv` memuat **satu baris per foto** dan tidak ada nama file yang berulang.
- [ ] Seluruh path pada `filename` sesuai dengan file di folder `images/`.
- [ ] `README.md` berisi identitas kelompok, prosedur singkat, dan kendala yang ditemukan.

### Langkah 7 — Menyusun README dan ZIP

Isi minimum `README.md`:

```markdown
# Dataset Permukaan — K05A_G01

Anggota: [Nama/NIM mahasiswa A], [Nama/NIM mahasiswa B]
Tanggal akuisisi: [tanggal]
Lokasi umum: Lingkungan kampus

## Ringkasan
- Kelas: rumput, paving, aspal, tanah, batu, semen
- Total gambar valid: [jumlah]
- Jumlah gambar per kelas: [rincian]

## Prosedur
[Jelaskan singkat alat, arah kamera, lokasi/petak, dan pembagian kerja.]

## Kendala dan keterbatasan
[Jelaskan kelas sulit ditemukan, foto yang dibuang, kondisi cahaya, dan sebagainya.]
```

Kompres folder utama menjadi **`K05A_G01_DatasetPermukaan.zip`** (sesuaikan kode kelompok), lalu unggah sesuai kanal dan tenggat yang ditetapkan dosen.

---

## F. Tugas Mahasiswa

### Tugas 1 — Akuisisi data (kelompok)

Buat dataset citra dari enam jenis permukaan di lingkungan kampus dengan target **180 gambar valid per kelompok**, masing-masing kelas sebanyak **30 gambar**. Kedua mahasiswa harus ikut memotret semua kelas. Utamakan beberapa petak sumber dan hindari foto yang hampir identik.

### Tugas 2 — Pelabelan dan metadata (kelompok)

Susun enam folder gambar dan satu `metadata.csv` sesuai format. Lakukan pemeriksaan silang: **Mahasiswa A memeriksa sebagian foto dan label Mahasiswa B, dan sebaliknya**. Perbaiki foto yang salah label atau tidak memenuhi kriteria.

### Tugas 3 — Analisis kualitas dataset (kelompok)

Dalam `README.md`, tambahkan bagian **Analisis Kualitas Dataset** sepanjang sekitar **150–250 kata** yang membahas:

1. Kelas mana yang paling mudah dan paling sulit dikumpulkan? Mengapa?
2. Dua contoh kondisi yang membuat label foto ambigu.
3. Perbedaan gambar dari dua lokasi/petak dalam kelas yang sama.
4. Langkah kelompok untuk menghindari foto duplikat dan bias pengambilan gambar.

### Tugas 4 — Refleksi individu

Setiap mahasiswa menuliskan **jawaban singkat (3–5 kalimat per pertanyaan)** untuk dua pertanyaan berikut, disimpan pada bagian akhir `README.md` dengan nama masing-masing:

1. Mengapa 30 foto yang diambil dari satu titik hampir sama tidak selalu lebih baik daripada 30 foto dari beberapa lokasi?
2. Apa risiko jika seluruh foto `rumput` dibuat dengan smartphone A, sementara seluruh foto `aspal` dibuat dengan smartphone B?

---

## G. Luaran yang Dikumpulkan

**Satu file ZIP per kelompok**, berisi:

1. Folder `images/` dengan enam subfolder kelas.
2. File `metadata.csv` yang sesuai dengan seluruh gambar.
3. File `README.md` berisi identitas, prosedur, analisis kualitas, kendala, dan refleksi individu.

**Kriteria selesai:** Dataset dapat dibuka, jumlah gambar dan metadata dapat diperiksa, serta semua gambar memiliki label dan sumber/lokasi yang dapat ditelusuri. Bila target tidak tercapai karena keterbatasan lokasi yang nyata, laporkan **jumlah aktual dan alasannya**; tidak diperbolehkan mengisi kekurangan dengan gambar palsu atau duplikat.

---

## H. Penilaian

| Komponen | Bobot | Indikator utama |
|---|---:|---|
| Kelengkapan dan keberagaman citra | 35% | Enam kelas, keseimbangan jumlah, beberapa lokasi, gambar jelas dan tidak duplikat |
| Ketepatan label dan metadata | 30% | Label sesuai, seluruh path valid, kode sumber jelas, CSV lengkap |
| Organisasi dan dokumentasi | 15% | Struktur folder rapi, README menjelaskan prosedur dan kendala |
| Kontribusi dan refleksi individu | 20% | Kedua mahasiswa berkontribusi dan memahami risiko bias dataset |
| **Total** | **100%** | |

> **Catatan untuk mahasiswa:** Penilaian tidak didasarkan pada akurasi Machine Learning, karena pada pertemuan ini **belum dilakukan pelatihan model**.

---

## I. Pertanyaan Diskusi Penutup

1. Apakah dataset dengan jumlah gambar yang sama pada setiap kelas sudah pasti berkualitas baik? Jelaskan.
2. Mengapa `location_id` perlu dicatat ketika mengumpulkan gambar permukaan?
3. Jika dua kelas memiliki warna yang hampir sama, karakteristik visual apa selain warna yang mungkin membantu membedakannya?
4. Bagaimana kemungkinan pengaruh bayangan, air, atau sudut kamera terhadap hasil klasifikasi di kemudian hari?
5. Apa perbedaan antara **gambar baru** dan **gambar yang benar-benar independen** dari gambar latih?

**Pengingat untuk pertemuan berikutnya:** Simpan dataset asli dan jangan menghapus metadata sumber/lokasi. Data tersebut akan berguna saat pembagian data latih dan data uji agar evaluasi model tidak terlalu optimistis.
