# 🍃 AirthMind: Sistem Cerdas Pemantau Kualitas Udara Dalam Ruangan

**🚀 Live Demo: [Coba Aplikasi AirthMind di Sini!](https://airthmind-mf5fzyxpgapjcju4d5hehu.streamlit.app/)**

**AirthMind** adalah proyek *Data Science* & *Machine Learning* yang bertujuan untuk memantau, mengklasifikasi, dan memprediksi kualitas udara di dalam ruangan (Indoor Air Quality) berdasarkan pembacaan data sensor IoT (Internet of Things). Lewat aplikasi ini, kita tidak hanya mengetahui status kesehatan udara saat ini, tetapi juga bisa meramalkan tingkat PM2.5 beberapa jam ke depan guna mengambil tindakan pencegahan sedini mungkin.

## 📈 Alur Kerja & Proses Proyek (Metodologi)

Proyek ini dirancang dari hulu ke hilir dengan tahapan sebagai berikut:

1. **Pemahaman Data (Exploratory Data Analysis / EDA)**
   - Menggunakan kumpulan data (*dataset*) dari sensor IoT yang mencatat berbagai variabel seperti Suhu, Kelembapan, Karbon Dioksida (CO2), *Particulate Matter* (PM2.5 & PM10), TVOC (bahan kimia), Karbon Monoksida (CO), dan status ventilasi ruangan.
   - Melakukan observasi korelasi antarkomponen gas untuk melihat penyebab utama memburuknya udara.

2. **Prapemrosesan Data (Data Preprocessing)**
   - Menangani tipe-tipe data mentah dan menerapkan algoritma **StandardScaler** (rekayasa numerik) agar semua nilai rentang angka sensor dapat seimbang/merata saat masuk ke dalam model komputasi.
   - Mengubah kolom target kualitatif (seperti Sehat, Sedang, Berbahaya) menjadi angka (*Encoding*) menggunakan algoritma **LabelEncoder**.

3. **Pembuatan Model Klasifikasi (XGBoost)**
   - Melatih model Prediktor Cuaca Ruangan menggunakan **eXtreme Gradient Boosting (XGBoost)**, sebuah sistem Machine Learning unggulan untuk data berbasis tabel.
   - Model ini bertugas untuk membaca nilai ke-8 sensor saat ini secara instan lalu menggolongkannya ke salah satu *Health Zone* (sehat, waspada, atau bahaya).

4. **Pembuatan Model Peramalan Waktu (Prophet)**
   - Algoritma prediksi linier tidak cukup untuk mengestimasi kepekatan udara esok hari. Oleh karena itu, diterapkan algoritma *Time-Series Forecasting* khusus dari Meta yaitu **Prophet**.
   - Model Prophet ini difungsikan untuk "melihat masa depan": memetakan tren pergerakan debu halus (PM2.5) selama **6 jam ke depan**.

5. **Deployment & Pembuatan Dashboard (Streamlit)**
   - Seluruh program prediksi otak (*scaler*, *label encoder*, *xgboost*, dan *prophet*) dibungkus menjadi file bawaan (`.pkl`) dengan pustaka *Joblib*.
   - Halaman antarmuka pengguna interaktif diciptakan dengan **Streamlit**, lalu diunggah ke internet (*Streamlit Community Cloud*) untuk memudahkan presentasi portofolio atau interaksi publik.

## 🚀 Fitur Unggulan Aplikasi
- **Simulasi Sensor Real-time:** Bisa memodifikasi tingkat polusi menggunakan *slider* intuitif di menu *sidebar*.
- **Klasifikasi Instan:** Deteksi langsung kondisi ruangan (misal: "Healthy" atau "Hazardous").
- **Alarm Aturan Khusus (*Rule-based Feedback*):** Jika hanya ada satu gas berlebih saja (misal CO terlalu tinggi), aplikasi langsung menyala merah memperingatkan adanya ancaman.
- **Grafik Ramalan 6 Jam Kedepan:** Visualisasi data *forecasting* untuk PM2.5 yang dinamis.

## 📂 Struktur Repositori
- `app.py`: *Script* UI utama untuk menjalankan panel antarmuka aplikasi Streamlit.
- `requirements.txt`: Daftar pustaka yang esensial untuk di-deploy ke Cloud.
- `models/`: Folder repositori otak mesin kecerdasan buatan (*scaler*, *encoder*, model klasifikasi, & prediksi waktu).
- `data/`: Letak file data mentah (`.csv`) dari perangkat IoT.
- `notebooks/`: Arsip *Jupyter Notebook* (`.ipynb`) berisi kode bereksperimen, penelitian historis, proses EDA, dan pelatihan awal AI.
