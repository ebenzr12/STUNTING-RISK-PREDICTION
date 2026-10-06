# Cek Stunting Balita

Aplikasi web untuk membantu kader posyandu dan orang tua memeriksa status tinggi badan balita
menurut standar WHO, melihat posisinya di kurva pertumbuhan, dan membandingkannya dengan
prediksi model machine learning.

**Demo:** _tempel link Streamlit di sini_

![Tampilan aplikasi](docs/screenshot.png)

## Fitur
- **Periksa satu anak**: Z-score TB/U (WHO), status stunting, dan kurva pertumbuhan dengan titik anak.
- **Unggah CSV**: periksa banyak balita sekaligus, lihat ringkasan, dan unduh hasilnya.
- **Pantau pertumbuhan**: lintasan beberapa pengukuran satu anak, dengan peringatan bila Z-score turun.
- **Peringatan data tidak wajar**: nilai dengan |Z| > 6 ditandai agar diukur ulang (kriteria WHO).
- **Pembanding ML dan zona abu-abu**: model terbaik (KNN/Random Forest) menjadi pembanding Z-score, dan kasus dengan keyakinan rendah ditandai "ukur ulang".

## Cara kerja
Status utama dihitung dari Z-score `((tinggi / M) ** L - 1) / (S * L)` memakai tabel WHO
(L = 1 untuk TB/U), dengan ambang Permenkes No. 2 Tahun 2020: < -3 SD sangat pendek,
-3 s.d. < -2 SD pendek, -2 s.d. +3 SD normal, > +3 SD tinggi.

## Data dan metode
- Dataset: *Stunting Toddler (Balita) Detection (121K rows)* (Pradana, Kaggle), fitur umur, jenis kelamin, tinggi badan.
- 81.574 baris duplikat dibuang **sebelum** membagi data latih dan uji (tersisa 39.425 baris).
- Label dataset divalidasi terhadap Z-score WHO: cocok 99,51%.
- Evaluasi: stratified 5-fold CV dan data uji 20%. Random Forest dan KNN sama-sama sekitar 99% (akurasi CV).
- Analisis lengkap (EDA, perbandingan 5 model, klasifikasi ordinal, conformal prediction, uji galat pengukuran) ada di `Stunting_ML.ipynb`.

## Keterbatasan
- Dataset tampak hasil simulasi dari standar WHO, bukan pengukuran lapangan (938 baris punya |Z| > 6). Akurasi tinggi
  berarti model mempelajari ulang aturan Z-score, jadi Z-score WHO dijadikan rujukan dan model hanya pembanding.
- Selisih ukur berbaring vs berdiri (sekitar 0,7 cm di usia 24 bulan) tidak dikoreksi.
- Alat bantu skrining, bukan diagnosis medis.

## Struktur
```
├── .streamlit/config.toml
├── artifacts/            # model.pkl & metrics.json (hasil train_model.py)
├── data/
│   ├── data_balita.csv   # dataset latih
│   └── who_lhfa.csv      # tabel LMS WHO TB/U 0-60 bulan
├── Stunting_ML.ipynb     # analisis ML lengkap
├── app.py                # aplikasi Streamlit
├── train_model.py        # latih & bandingkan model
├── who.py                # Z-score, status, kurva WHO
└── requirements.txt
```

## Menjalankan
```bash
pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

## Sumber
- WHO Child Growth Standards, tabel length/height-for-age 0-5 tahun.
- Permenkes No. 2 Tahun 2020 tentang Standar Antropometri Anak.
- Pradana, R. P. *Stunting Toddler (Balita) Detection (121K rows)*. Kaggle.
