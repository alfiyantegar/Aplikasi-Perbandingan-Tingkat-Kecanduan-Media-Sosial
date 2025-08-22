import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -----------------------------
# Pengaturan Aplikasi
# -----------------------------
st.set_page_config(page_title="Cek Kecanduan Medsos", layout="wide")
DB_PATH = "riwayat_prediksi.db"  # Tempat menyimpan riwayat prediksi

# -----------------------------
# Membuat Tempat Penyimpanan Riwayat
# -----------------------------
def buat_tempat_riwayat():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS riwayat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        waktu TEXT,
        cara_input TEXT,
        data_masuk TEXT,
        hasil_rf REAL,
        hasil_lr REAL
    )
    """)
    conn.commit()
    conn.close()

# Hapus database lama jika ada ketidaksesuaian kolom
def perbaiki_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS riwayat")
    conn.commit()
    conn.close()
    buat_tempat_riwayat()

# Panggil fungsi untuk memastikan skema database benar
perbaiki_database()

def simpan_riwayat(data_masuk, hasil_rf, hasil_lr, cara_input="manual"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT INTO riwayat (waktu, cara_input, data_masuk, hasil_rf, hasil_lr)
        VALUES (?,?,?,?,?)
    """, (waktu, cara_input, json.dumps(data_masuk), hasil_rf, hasil_lr))
    conn.commit()
    conn.close()

def lihat_riwayat(jumlah=10):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM riwayat ORDER BY id DESC LIMIT ?", conn, params=(jumlah,))
    conn.close()
    return df

# -----------------------------
# Melatih Sistem dan Hitung Metrik
# -----------------------------
def latih_sistem_dan_evaluasi(data):
    X = data.drop(columns=["Student_ID", "Addicted_Score"], errors="ignore")  # Data yang digunakan
    y = data["Addicted_Score"]  # Hasil yang diprediksi

    # Pisahkan data angka dan teks
    kolom_angka = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    kolom_teks = X.select_dtypes(include=["object"]).columns.tolist()

    # Siapkan data untuk sistem
    pengolah_data = ColumnTransformer(
        transformers=[
            ("angka", StandardScaler(), kolom_angka),
            ("teks", OneHotEncoder(handle_unknown="ignore", sparse_output=False), kolom_teks)
        ]
    )

    # Buat dua sistem prediksi
    sistem_rf = Pipeline([
        ("pengolah", pengolah_data),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    sistem_lr = Pipeline([
        ("pengolah", pengolah_data),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # Bagi data untuk latihan dan uji
    X_latih, X_uji, y_latih, y_uji = train_test_split(X, y, test_size=0.2, random_state=42)
    
    n_total = len(X)
    n_train = len(X_latih)
    n_test  = len(X_uji)
    
    st.session_state.split_info = {
    "Total Data": n_total,
    "Data Latih (80%)": n_train,
    "Data Uji (20%)": n_test
}
    # Latih sistem
    sistem_rf.fit(X_latih, y_latih)
    sistem_lr.fit(X_latih, y_latih)

    # Prediksi pada data uji
    y_pred_rf = sistem_rf.predict(X_uji)
    y_pred_lr = sistem_lr.predict(X_uji)

    # Hitung metrik evaluasi
    metrik_rf = {
        "MSE": round(mean_squared_error(y_uji, y_pred_rf), 2),
        "RMSE": round(mean_squared_error(y_uji, y_pred_rf, squared=False), 2),
        "MAE": round(mean_absolute_error(y_uji, y_pred_rf), 2),
        "R²": round(r2_score(y_uji, y_pred_rf), 2)
    }
    metrik_lr = {
        "MSE": round(mean_squared_error(y_uji, y_pred_lr), 2),
        "RMSE": round(mean_squared_error(y_uji, y_pred_lr, squared=False), 2),
        "MAE": round(mean_absolute_error(y_uji, y_pred_lr), 2),
        "R²": round(r2_score(y_uji, y_pred_lr), 2)
    }

    # Ekstrak fitur terpenting dari Random Forest
    feature_names = []
    if kolom_angka:
        feature_names.extend(kolom_angka)
    if kolom_teks:
        ohe = sistem_rf.named_steps["pengolah"].named_transformers_["teks"]
        ohe_feature_names = ohe.get_feature_names_out(kolom_teks)
        feature_names.extend(ohe_feature_names)
    
    importances = sistem_rf.named_steps["model"].feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    return sistem_rf, sistem_lr, metrik_rf, metrik_lr, feature_importance

# -----------------------------
# Halaman Utama
# -----------------------------
st.title("🌟 Cek Risiko Kecanduan Media Sosial")
st.markdown("""
Selamat datang di aplikasi **Cek Kecanduan Medsos**!  
Aplikasi ini membantu kamu tahu seberapa besar risiko seseorang kecanduan media sosial, seperti Instagram, TikTok, atau lainnya, berdasarkan kebiasaan sehari-hari.  

**Apa yang bisa kamu lakukan?**  
- Lihat contoh data yang digunakan.  
- Masukkan data untuk memprediksi risiko kecanduan.  
- Cek riwayat prediksi sebelumnya.  
- Lihat performa sistem prediksi dan faktor yang paling memengaruhi hasil di tab Analisis.  

Pilih salah satu tab di bawah ini untuk mulai!
""")

tab_data, tab_prediksi, tab_riwayat, tab_analisis = st.tabs(["📊 Lihat Data", "🔍 Prediksi", "📜 Riwayat", "📈 Analisis Performa Model"])

# -----------------------------
# Tab: Lihat Data
# -----------------------------
with tab_data:
    st.subheader("📊 Lihat Data")
    st.info("Di sini kamu bisa melihat contoh data yang digunakan atau mengunggah data baru dalam bentuk file CSV.")

    # Cek apakah data sudah ada
    if "data" not in st.session_state:
        try:
            data_awal = pd.read_csv("Students Social Media Addiction.csv")
            st.session_state.data = data_awal
            # Latih ulang model dan simpan metrik serta fitur terpenting
            st.session_state.sistem_rf, st.session_state.sistem_lr, st.session_state.metrik_rf, st.session_state.metrik_lr, st.session_state.feature_importance = latih_sistem_dan_evaluasi(data_awal)
        except FileNotFoundError:
            st.error("Data awal tidak ditemukan. Silakan unggah file CSV terlebih dahulu.")
            st.session_state.data = None

    # Opsi unggah file baru
    file_baru = st.file_uploader("Unggah file CSV (opsional)", type="csv")
    if file_baru:
        data_baru = pd.read_csv(file_baru)
        st.session_state.data = data_baru
        # Latih ulang model dan simpan metrik serta fitur terpenting saat dataset baru diunggah
        st.session_state.sistem_rf, st.session_state.sistem_lr, st.session_state.metrik_rf, st.session_state.metrik_lr, st.session_state.feature_importance = latih_sistem_dan_evaluasi(data_baru)
        st.success("✅ Data baru berhasil dimuat dan dilatih, Segera Cek Tab Analisis Performa Model!")

    if st.session_state.data is not None:
        st.write("**Contoh 5 baris data:**")
        st.dataframe(st.session_state.data.head())

# -----------------------------
# Tab: Prediksi
# -----------------------------
with tab_prediksi:
    st.subheader("🔍 Prediksi Risiko Kecanduan")
    st.info("Kamu bisa memasukkan data satu per satu (manual) atau mengunggah file CSV untuk memeriksa banyak data sekaligus.")

    pilihan = st.radio("Pilih cara memasukkan data:", ["Isi Manual", "Unggah File CSV"])

    if pilihan == "Isi Manual":
        if st.session_state.data is None:
            st.error("Data belum dimuat. Silakan unggah file CSV di tab 'Lihat Data'.")
        else:
            data = st.session_state.data
            # Pastikan Addicted_Score tidak masuk ke kolom input
            kolom_masuk = [c for c in data.columns if c not in ["Student_ID", "Addicted_Score"]]

            st.write("Masukkan informasi berikut:")
            data_masuk = {}
            for kol in kolom_masuk:
                if kol == "Age":
                    data_masuk[kol] = st.number_input(
                        "Usia (dalam tahun, misalnya 18 atau 20)", 
                        min_value=18.0, 
                        max_value=24.0, 
                        value=20.0, 
                        step=1.0
                    )
                elif kol == "Gender":
                    data_masuk[kol] = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
                elif kol == "Academic_Level":
                    data_masuk[kol] = st.selectbox("Jenjang Pendidikan", ["Sarjana", "Pascasarjana"])
                elif kol == "Country":
                    data_masuk[kol] = st.selectbox("Negara", ["Bangladesh", "Lainnya"])
                elif kol == "Most_Used_Platform":
                    data_masuk[kol] = st.selectbox("Media Sosial yang Paling Sering Digunakan", ["Instagram", "TikTok", "YouTube", "Lainnya"])
                elif kol == "Affects_Academic_Performance":
                    data_masuk[kol] = st.selectbox("Apakah media sosial mengganggu prestasi belajar?", ["Ya", "Tidak"])
                elif kol == "Relationship_Status":
                    data_masuk[kol] = st.selectbox("Status Hubungan", ["Dalam Hubungan", "Lajang"])
                elif kol == "Conflicts_Over_Social_Media":
                    data_masuk[kol] = st.number_input(
                        "Seberapa sering bertengkar karena media sosial? (0 = tidak pernah, 5 = sangat sering)",
                        min_value=0.0,
                        max_value=5.0,
                        value=0.0,
                        step=0.1
                    )
                elif kol == "Avg_Daily_Usage_Hours":
                    data_masuk[kol] = st.number_input(
                        "Rata-rata jam penggunaan media sosial per hari (misalnya 4.9 jam)",
                        min_value=1.5,
                        max_value=8.5,
                        value=4.9,
                        step=0.1
                    )
                elif kol == "Sleep_Hours_Per_Night":
                    data_masuk[kol] = st.number_input(
                        "Jam tidur per malam (misalnya 6.9 jam)",
                        min_value=3.8,
                        max_value=9.6,
                        value=6.9,
                        step=0.1
                    )
                elif kol == "Mental_Health_Score":
                    data_masuk[kol] = st.number_input(
                        "Skor kesehatan mental (4 = buruk, 9 = sangat baik)",
                        min_value=4.0,
                        max_value=9.0,
                        value=6.2,
                        step=0.1
                    )
                else:  # Untuk kolom lain
                    if data[kol].dtype == "object":
                        pilihan = data[kol].unique().tolist()
                        data_masuk[kol] = st.selectbox(f"Pilih {kol}", pilihan)
                    else:
                        data_masuk[kol] = st.number_input(
                            f"Masukkan {kol} (min: {data[kol].min():.1f}, maks: {data[kol].max():.1f})",
                            min_value=float(data[kol].min()),
                            max_value=float(data[kol].max()),
                            value=float(data[kol].mean())
                        )

            if st.button("🔎 Lihat Hasil Prediksi"):
                sistem_rf, sistem_lr = st.session_state.sistem_rf, st.session_state.sistem_lr
                hasil_rf = sistem_rf.predict(pd.DataFrame([data_masuk]))[0]
                hasil_lr = sistem_lr.predict(pd.DataFrame([data_masuk]))[0]

                st.success(f"**Hasil Prediksi (Skala 0-10):**")
                st.write(f"🌳 Metode Random Forest: **{hasil_rf:.1f}** (Semakin tinggi, semakin berisiko kecanduan)")
                st.write(f"📈 Metode Regresi Logistik: **{hasil_lr:.1f}** (Semakin tinggi, semakin berisiko kecanduan)")
                
                # Beri saran sederhana berdasarkan hasil
                if hasil_rf > 75 or hasil_lr > 75:
                    st.warning("⚠️ Skor tinggi! Coba kurangi waktu penggunaan media sosial dan perhatikan kebiasaan sehari-hari.")
                elif hasil_rf > 50 or hasil_lr > 50:
                    st.info("ℹ️ Skor sedang. Perhatikan pola penggunaan media sosial agar tetap seimbang.")
                else:
                    st.success("✅ Skor rendah. Kebiasaan media sosial tampaknya masih terkendali!")

                simpan_riwayat(data_masuk, hasil_rf, hasil_lr, "Manual")

    else:  # Unggah File CSV
        file_csv = st.file_uploader("Unggah file CSV untuk prediksi banyak data", type="csv")
        if file_csv and st.session_state.data is not None:
            data_masuk = pd.read_csv(file_csv)
            sistem_rf, sistem_lr = st.session_state.sistem_rf, st.session_state.sistem_lr

            # Prediksi untuk semua data
            hasil_rf = sistem_rf.predict(data_masuk.drop(columns=["Student_ID", "Addicted_Score"], errors="ignore"))
            hasil_lr = sistem_lr.predict(data_masuk.drop(columns=["Student_ID", "Addicted_Score"], errors="ignore"))

            data_masuk["Hasil Metode Random Forest"] = hasil_rf
            data_masuk["Hasil Metode Regresi Logistik"] = hasil_lr

            st.write("**Hasil Prediksi untuk Data yang Diunggah:**")
            st.dataframe(data_masuk)

            # Simpan riwayat
            for _, row in data_masuk.iterrows():
                data_in = row.drop(labels=["Hasil Metode Random Forest", "Hasil Metode Regresi Logistik"]).to_dict()
                simpan_riwayat(data_in, row["Hasil Metode Random Forest"], row["Hasil Metode Regresi Logistik"], "CSV")

# -----------------------------
# Tab: Riwayat
# -----------------------------
with tab_riwayat:
    st.subheader("📜 Riwayat Prediksi")
    st.info("Lihat semua prediksi yang pernah kamu lakukan di sini.")

    riwayat = lihat_riwayat(10)
    if not riwayat.empty:
        riwayat["data_masuk"] = riwayat["data_masuk"].apply(lambda x: json.loads(x))
        st.write("**10 Prediksi Terakhir:**")
        st.dataframe(riwayat[["waktu", "cara_input", "data_masuk", "hasil_rf", "hasil_lr"]])
    else:
        st.warning("Belum ada riwayat prediksi. Coba lakukan prediksi di tab 'Prediksi'!")

# -----------------------------
# Tab: Analisis Performa Model
# -----------------------------
with tab_analisis:
    st.subheader("📈 Analisis Performa Model")
    st.info("Di sini kamu bisa melihat seberapa baik sistem prediksi bekerja dan faktor apa yang paling memengaruhi hasil prediksi.")
    if "split_info" in st.session_state:
        st.write("### 📦 Ringkasan Pembagian Data (80% : 20%)")
    split_df = pd.DataFrame(list(st.session_state.split_info.items()), columns=["Keterangan", "Jumlah Baris"])
    st.dataframe(split_df, use_container_width=True)

    if st.session_state.data is not None and "metrik_rf" in st.session_state:
        
        # Buat tabel metrik
        metrik_df = pd.DataFrame({
            "Metode": ["Random Forest", "Regresi Logistik"],
            "MSE": [st.session_state.metrik_rf["MSE"], st.session_state.metrik_lr["MSE"]],
            "RMSE": [st.session_state.metrik_rf["RMSE"], st.session_state.metrik_lr["RMSE"]],
            "MAE": [st.session_state.metrik_rf["MAE"], st.session_state.metrik_lr["MAE"]],
            "R²": [st.session_state.metrik_rf["R²"], st.session_state.metrik_lr["R²"]]
        })
        st.write("**Tabel Performa Model:**")
        st.write("**Penjelasan:**")
        st.write("- **MSE**: Rata-rata kuadrat kesalahan prediksi. Semakin kecil, semakin akurat.")
        st.write("- **RMSE**: Akar dari MSE, menunjukkan besar kesalahan prediksi. Semakin kecil, semakin baik.")
        st.write("- **MAE**: Rata-rata kesalahan absolut. Semakin kecil, semakin baik.")
        st.write("- **R²**: Menunjukkan seberapa baik model menjelaskan data (0 sampai 1). Semakin mendekati 1, semakin baik.")
        st.dataframe(metrik_df)

        # Visualisasi metrik
        st.write("**Grafik Perbandingan Performa Model:**")
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        # Grafik MSE
        axes[0].bar(metrik_df["Metode"], metrik_df["MSE"], color=["#4CAF50", "#2196F3"])
        axes[0].set_title("MSE")
        axes[0].set_ylim(0, max(metrik_df["MSE"]) * 1.2)
        # Grafik RMSE
        axes[1].bar(metrik_df["Metode"], metrik_df["RMSE"], color=["#4CAF50", "#2196F3"])
        axes[1].set_title("RMSE")
        axes[1].set_ylim(0, max(metrik_df["RMSE"]) * 1.2)
        # Grafik MAE
        axes[2].bar(metrik_df["Metode"], metrik_df["MAE"], color=["#4CAF50", "#2196F3"])
        axes[2].set_title("MAE")
        axes[2].set_ylim(0, max(metrik_df["MAE"]) * 1.2)
        # Grafik R²
        axes[3].bar(metrik_df["Metode"], metrik_df["R²"], color=["#4CAF50", "#2196F3"])
        axes[3].set_title("R²")
        axes[3].set_ylim(0, 1.2)
        plt.tight_layout()
        st.pyplot(fig)

        # Visualisasi fitur terpenting (hanya untuk Random Forest)
        st.write("**Grafik Fitur Terpenting (Metode Random Forest):**")
        st.write("Grafik ini menunjukkan faktor mana yang paling memengaruhi prediksi risiko kecanduan media sosial. Semakin tinggi batang, semakin besar pengaruhnya.")
        feature_names, importances = zip(*st.session_state.feature_importance[:10])  # Ambil 10 fitur teratas
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_names, importances, color="#4CAF50")
        ax.set_xlabel("Tingkat Kepentingan")
        ax.set_title("Fitur yang Paling Memengaruhi Prediksi")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Belum ada data untuk analisis. Silakan unggah data di tab 'Lihat Data'.")
