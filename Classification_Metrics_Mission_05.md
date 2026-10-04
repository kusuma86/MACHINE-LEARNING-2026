# Classification Metrics

## Mengapa Model Perlu Dievaluasi?

Setelah model melakukan classification, kita perlu menjawab:

> **Seberapa baik model membuat keputusan?**

Tidak cukup hanya melihat apakah model bisa menghasilkan prediction.

Kita perlu melihat:

- berapa banyak prediction yang benar;
- berapa banyak yang salah;
- jenis kesalahan apa yang terjadi;
- apakah suatu kelas sering terlewat;
- apakah model terlalu sering memberikan alarm palsu.

---

# 1. Contoh Sederhana

Misalkan kita memiliki model yang memprediksi:

```text
Positif
Negatif
```

Contoh kasus:

> Apakah sebuah email termasuk Spam?

Kita memiliki 10 email.

| Email | Actual | Predicted |
|---|---|---|
| 1 | Spam | Spam |
| 2 | Spam | Spam |
| 3 | Spam | Not Spam |
| 4 | Spam | Spam |
| 5 | Not Spam | Not Spam |
| 6 | Not Spam | Spam |
| 7 | Not Spam | Not Spam |
| 8 | Not Spam | Not Spam |
| 9 | Not Spam | Not Spam |
| 10 | Not Spam | Spam |

Prediction tidak sempurna.

Ada prediction yang benar dan ada yang salah.

---

# 2. Confusion Matrix

Confusion matrix menunjukkan hubungan antara:

- **Actual**
- **Predicted**

Untuk binary classification:

| Actual \ Predicted | Spam | Not Spam |
|---|---:|---:|
| Spam | 3 | 1 |
| Not Spam | 2 | 4 |

Artinya:

```text
3 Spam diprediksi Spam
1 Spam diprediksi Not Spam
2 Not Spam diprediksi Spam
4 Not Spam diprediksi Not Spam
```

---

# 3. TP, TN, FP, FN

Jika kita anggap:

```text
Positive = Spam
Negative = Not Spam
```

maka:

| Istilah | Arti |
|---|---|
| TP | True Positive |
| TN | True Negative |
| FP | False Positive |
| FN | False Negative |

Dari contoh:

```text
TP = 3
TN = 4
FP = 2
FN = 1
```

---

# 4. True Positive (TP)

**True Positive** berarti:

```text
Actual    = Positive
Predicted = Positive
```

Contoh:

```text
Actual    = Spam
Predicted = Spam
```

Prediction benar.

---

# 5. True Negative (TN)

**True Negative** berarti:

```text
Actual    = Negative
Predicted = Negative
```

Contoh:

```text
Actual    = Not Spam
Predicted = Not Spam
```

Prediction benar.

---

# 6. False Positive (FP)

**False Positive** berarti:

```text
Actual    = Negative
Predicted = Positive
```

Contoh:

```text
Actual    = Not Spam
Predicted = Spam
```

Model memberikan **alarm positif yang salah**.

False Positive sering disebut:

```text
False Alarm
```

---

# 7. False Negative (FN)

**False Negative** berarti:

```text
Actual    = Positive
Predicted = Negative
```

Contoh:

```text
Actual    = Spam
Predicted = Not Spam
```

Model gagal mendeteksi kondisi positif.

---

# 8. Accuracy

Accuracy menunjukkan proporsi seluruh prediction yang benar.

Rumus:

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

Dari contoh:

```text
TP = 3
TN = 4
FP = 2
FN = 1
```

Maka:

$$
Accuracy = \frac{3+4}{3+4+2+1}
$$

$$
Accuracy = \frac{7}{10}
$$

$$
Accuracy = 0.70
$$

Jadi:

```text
Accuracy = 70%
```

---

# 9. Mengapa Accuracy Tidak Cukup?

Misalkan dataset:

```text
95 Not Spam
5 Spam
```

Model selalu menjawab:

```text
Not Spam
```

Maka:

```text
95 prediction benar
5 prediction salah
```

Accuracy:

```text
95%
```

Terlihat sangat bagus.

Tetapi:

```text
Spam yang berhasil ditemukan = 0
```

Artinya model sama sekali tidak berguna untuk mendeteksi Spam.

> **Accuracy tinggi tidak selalu berarti model bagus.**

---

# 10. Precision

Precision menjawab:

> Dari semua yang **diprediksi Positive**, berapa yang benar-benar Positive?

Rumus:

$$
Precision = \frac{TP}{TP + FP}
$$

Dari contoh:

```text
TP = 3
FP = 2
```

Maka:

$$
Precision = \frac{3}{3+2}
$$

$$
Precision = 0.60
$$

Jadi:

```text
Precision = 60%
```

Interpretasi:

> Dari semua email yang diprediksi Spam, 60% benar-benar Spam.

---

# 11. Recall

Recall menjawab:

> Dari semua data yang **sebenarnya Positive**, berapa yang berhasil ditemukan?

Rumus:

$$
Recall = \frac{TP}{TP + FN}
$$

Dari contoh:

```text
TP = 3
FN = 1
```

Maka:

$$
Recall = \frac{3}{3+1}
$$

$$
Recall = 0.75
$$

Jadi:

```text
Recall = 75%
```

Interpretasi:

> Dari seluruh email Spam, 75% berhasil ditemukan model.

---

# 12. Precision vs Recall

### Precision

Fokus pada:

```text
Prediction Positive
```

Pertanyaan:

> Ketika model mengatakan Positive, seberapa sering model benar?

---

### Recall

Fokus pada:

```text
Actual Positive
```

Pertanyaan:

> Dari seluruh Positive yang ada, berapa banyak yang berhasil ditemukan?

---

# 13. Cara Mudah Mengingat

### Precision

```text
Yang saya katakan Positive,
berapa yang benar?
```

### Recall

```text
Dari semua Positive yang ada,
berapa yang berhasil saya temukan?
```

---

# 14. Specificity

Specificity mengukur kemampuan model mengenali kelas Negative dengan benar.

Rumus:

$$
Specificity = \frac{TN}{TN + FP}
$$

Dari contoh:

```text
TN = 4
FP = 2
```

Maka:

$$
Specificity = \frac{4}{4+2}
$$

$$
Specificity \approx 0.67
$$

Jadi:

```text
Specificity ≈ 67%
```

Interpretasi:

> Dari semua email Not Spam, sekitar 67% berhasil dikenali dengan benar.

---

# 15. F1-Score

Kadang precision tinggi tetapi recall rendah.

Atau sebaliknya.

F1-score digunakan untuk menggabungkan keduanya.

Rumus:

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

Dengan:

```text
Precision = 0.60
Recall    = 0.75
```

Maka:

$$
F1 = 2 \times \frac{0.60 \times 0.75}{0.60 + 0.75}
$$

$$
F1 \approx 0.67
$$

---

# 16. Ringkasan Metric Binary Classification

| Metric | Pertanyaan Utama |
|---|---|
| Accuracy | Berapa persen seluruh prediction yang benar? |
| Precision | Dari semua Positive prediction, berapa yang benar? |
| Recall | Dari semua Actual Positive, berapa yang ditemukan? |
| Specificity | Dari semua Actual Negative, berapa yang dikenali? |
| F1-score | Seberapa seimbang precision dan recall? |

---

# 17. Confusion Matrix di Python

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)
```

Misalnya hasil:

```text
[[4 2]
 [1 3]]
```

Interpretasi bergantung pada urutan label.

Untuk binary classification, bentuk umumnya:

```text
[[TN FP]
 [FN TP]]
```

Jadi:

```text
TN = 4
FP = 2
FN = 1
TP = 3
```

---

# 18. Menampilkan Confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)
```

Confusion matrix membantu melihat:

- prediction benar;
- False Positive;
- False Negative;
- pola kesalahan model.

---

# 19. Menghitung Metric di Python

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
```

Kemudian:

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)
```

---

# 20. Classification Report

Scikit-learn dapat menampilkan beberapa metric sekaligus:

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

Output biasanya berisi:

```text
precision
recall
f1-score
support
```

---

# 21. Apa Itu Support?

**Support** adalah jumlah sample aktual pada suatu kelas.

Contoh:

```text
Spam     = 4
Not Spam = 6
```

Maka support:

```text
Spam     → 4
Not Spam → 6
```

Support bukan ukuran performa.

Support hanya menunjukkan:

> berapa banyak data aktual pada setiap kelas.

---

# 22. Multiclass Classification

Untuk lebih dari dua kelas:

```text
Cat
Dog
Rabbit
```

confusion matrix dapat berbentuk:

| Actual \ Predicted | Cat | Dog | Rabbit |
|---|---:|---:|---:|
| Cat | 8 | 1 | 1 |
| Dog | 2 | 7 | 1 |
| Rabbit | 0 | 2 | 8 |

Nilai diagonal:

```text
8
7
8
```

adalah prediction yang benar.

Nilai di luar diagonal menunjukkan kesalahan.

---

# 23. Precision dan Recall pada Multiclass

Pada multiclass classification, setiap kelas dapat dianggap secara bergantian sebagai:

```text
Positive
```

sementara kelas lain dianggap:

```text
Negative
```

Karena itu setiap kelas memiliki:

```text
precision
recall
f1-score
```

masing-masing.

---

# 24. Macro Average

`macro average` menghitung metric untuk setiap kelas, kemudian mengambil rata-rata.

Contoh:

```text
F1 Cat    = 0.80
F1 Dog    = 0.60
F1 Rabbit = 0.70
```

Maka:

$$
F1_{macro} = \frac{0.80 + 0.60 + 0.70}{3}
$$

$$
F1_{macro} = 0.70
$$

Semua kelas memiliki bobot yang sama.

---

# 25. Weighted Average

`weighted average` juga menghitung metric per kelas.

Tetapi rata-ratanya mempertimbangkan jumlah sample pada setiap kelas.

Kelas dengan support lebih besar memberikan pengaruh lebih besar.

---

# 26. Macro vs Weighted

| Average | Karakteristik |
|---|---|
| Macro | Semua kelas dianggap sama penting |
| Weighted | Mempertimbangkan jumlah sample tiap kelas |

Jika dataset tidak seimbang, keduanya dapat menghasilkan nilai yang berbeda.

---

# 27. Mana Metric yang Harus Digunakan?

Tidak ada satu metric yang selalu paling baik.

Gunakan:

### Accuracy

Jika:

- kelas relatif seimbang;
- jenis kesalahan memiliki dampak yang relatif sama.

### Precision

Jika:

> False Positive ingin dikurangi.

### Recall

Jika:

> False Negative ingin dikurangi.

### F1-score

Jika:

> precision dan recall sama-sama penting.

---

# 28. Jangan Hanya Melihat Satu Angka

Model A:

```text
Accuracy = 92%
```

belum tentu lebih baik daripada Model B:

```text
Accuracy = 89%
```

Kita juga perlu melihat:

```text
Precision
Recall
F1-score
Confusion Matrix
```

terutama pada kelas yang penting.

---

# 29. Take-Home Message

Evaluasi classification bukan hanya:

```text
"Berapa accuracy?"
```

Tetapi juga:

```text
Prediction mana yang salah?
```

```text
Apakah Positive sering terlewat?
```

```text
Apakah model terlalu sering memberi False Positive?
```

```text
Bagaimana precision dan recall setiap kelas?
```

---

# 30. Ringkasan

```text
CONFUSION MATRIX
       ↓
TP / TN / FP / FN
       ↓
ACCURACY
PRECISION
RECALL
SPECIFICITY
F1-SCORE
       ↓
INTERPRETASI MODEL
```

---

# Quick Check

### 1
Apa perbedaan False Positive dan False Negative?

### 2
Jika ingin mengurangi Positive yang terlewat, metric apa yang perlu diperhatikan?

### 3
Jika ingin mengurangi False Alarm, metric apa yang perlu diperhatikan?

### 4
Mengapa accuracy tinggi belum tentu berarti model bagus?

### 5
Apa fungsi confusion matrix?

### 6
Apa perbedaan macro average dan weighted average?
