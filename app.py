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
st.set_page_config(
    page_title="Perbandingan Metode Prediksi Kecanduan Media Sosial",
    layout="wide"
)
DB_PATH = "riwayat_prediksi.db"

# -----------------------------
# Util & Database
# -----------------------------
def _get_ohe():
    """OneHotEncoder kompatibel lintas-versi sklearn."""
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

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

# ⚠️ Kalau kamu tidak ingin riwayat terhapus tiap rerun, hapus pemanggilan fungsi ini.
def perbaiki_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS riwayat")
    conn.commit()
    conn.close()
    buat_tempat_riwayat()

perbaiki_database()  # hapus baris ini jika ingin menyimpan riwayat jangka panjang

def simpan_riwayat(data_masuk, hasil_rf, hasil_lr, cara_input="manual"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT INTO riwayat (waktu, cara_input, data_masuk, hasil_rf, hasil_lr)
        VALUES (?,?,?,?,?)
    """, (waktu, cara_input, json.dumps(data_masuk), float(hasil_rf), float(hasil_lr)))
    conn.commit()
    conn.close()

def lihat_riwayat(jumlah=10):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM riwayat ORDER BY id DESC LIMIT ?",
        conn, params=(jumlah,)
    )
    conn.close()
    return df

# -----------------------------
# Latih model & hitung metrik
# -----------------------------
def latih_sistem_dan_evaluasi(data: pd.DataFrame):
    if "Addicted_Score" not in data.columns:
        st.error("Kolom target 'Addicted_Score' tidak ditemukan di dataset.")
        raise ValueError("Kolom 'Addicted_Score' wajib ada.")

    # X & y
    X = data.drop(columns=["Student_ID", "Addicted_Score"], errors="ignore")
    y = data["Addicted_Score"]

    # Tipe fitur
    kolom_angka = X.select_dtypes(include=[np.number]).columns.tolist()
    kolom_teks = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessor
    ohe = _get_ohe()
    pengolah_data = ColumnTransformer(
        transformers=[
            ("angka", StandardScaler(), kolom_angka),
            ("teks", ohe, kolom_teks)
        ],
        remainder="drop"
    )

    # Pipelines
    sistem_rf = Pipeline([
        ("pengolah", pengolah_data),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    sistem_lr = Pipeline([
        ("pengolah", pengolah_data),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # Split
    X_latih, X_uji, y_latih, y_uji = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Info split
    st.session_state.split_info = {
        "Total Data": len(X),
        "Data Latih (80%)": len(X_latih),
        "Data Uji (20%)": len(X_uji)
    }

    # Fit
    sistem_rf.fit(X_latih, y_latih)
    sistem_lr.fit(X_latih, y_latih)

    # Prediksi uji
    y_pred_rf = sistem_rf.predict(X_uji)
    y_pred_lr = sistem_lr.predict(X_uji)

    # Metrik – TANPA squared=False (kompatibel semua versi)
    mse_rf = mean_squared_error(y_uji, y_pred_rf)
    mse_lr = mean_squared_error(y_uji, y_pred_lr)

    metrik_rf = {
        "MSE": round(mse_rf, 2),
        "RMSE": round(float(np.sqrt(mse_rf)), 2),
        "MAE": round(mean_absolute_error(y_uji, y_pred_rf), 2),
        "R²": round(r2_score(y_uji, y_pred_rf), 2)
    }
    metrik_lr = {
        "MSE": round(mse_lr, 2),
        "RMSE": round(float(np.sqrt(mse_lr)), 2),
        "MAE": round(mean_absolute_error(y_uji, y_pred_lr), 2),
        "R²": round(r2_score(y_uji, y_pred_lr), 2)
    }

    # Feature importance RF
    feature_names = []
    if kolom_angka:
        feature_names.extend(kolom_angka)
    if kolom_teks:
        try:
            ohe_names = sistem_rf.named_steps["pengolah"] \
                .named_transformers_["teks"] \
                .get_feature_names_out(kolom_teks)
        except AttributeError:
            # sklearn lama
            ohe_names = sistem_rf.named_steps["pengolah"] \
                .named_transformers_["teks"] \
                .get_feature_names(kolom_teks)
        feature_names.extend(list(ohe_names))

    importances = sistem_rf.named_steps["model"].feature_importances_
    # Jaga-jaga mismatch panjang
    if len(importances) != len(feature_names):
        feature_names = [f"fitur_{i}" for i in range(len(importances))]

    feature_importance = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    return sistem_rf, sistem_lr, metrik_rf, metrik_lr, feature_importance

# -----------------------------
# UI
# -----------------------------
st.title("📊 Aplikasi Perbandingan Metode Prediksi Kecanduan Media Sosial")

st.markdown("""
Aplikasi ini berfokus pada **analisis dataset terbaru** agar penelitian bisa terus diperbarui dari tahun ke tahun.  
Unggah dataset baru, kemudian sistem akan melatih ulang **Random Forest** dan **Regresi Logistik** untuk membandingkan performanya.

**Tab Utama**
- 📊 **Lihat Data** – Unggah/lihat dataset dan latih model.
- 📈 **Analisis Performa Model** – Lihat metrik (MSE, RMSE, MAE, R²) serta fitur terpenting.
- 🔍 **Prediksi** – Uji prediksi (input manual atau CSV kecil).
- 📜 **Riwayat** – Arsip prediksi tersimpan (SQLite).
""")

tab_data, tab_prediksi, tab_riwayat, tab_analisis = st.tabs(
    ["📊 Lihat Data", "🔍 Prediksi", "📜 Riwayat", "📈 Analisis Performa Model"]
)

# -----------------------------
# Tab: Lihat Data
# -----------------------------
with tab_data:
    st.subheader("📊 Lihat Data")
    st.info("Unggah file CSV untuk melatih ulang model. Pastikan kolom target bernama **Addicted_Score**.")

    if "data" not in st.session_state:
        try:
            data_awal = pd.read_csv("Students Social Media Addiction.csv")
            st.session_state.data = data_awal
            st.session_state.sistem_rf, st.session_state.sistem_lr, \
            st.session_state.metrik_rf, st.session_state.metrik_lr, \
            st.session_state.feature_importance = latih_sistem_dan_evaluasi(data_awal)
        except FileNotFoundError:
            st.warning("Data awal tidak ditemukan. Silakan unggah CSV.")
            st.session_state.data = None

    file_baru = st.file_uploader("Unggah CSV (opsional)", type="csv")
    if file_baru:
        data_baru = pd.read_csv(file_baru)
        st.session_state.data = data_baru
        st.session_state.sistem_rf, st.session_state.sistem_lr, \
        st.session_state.metrik_rf, st.session_state.metrik_lr, \
        st.session_state.feature_importance = latih_sistem_dan_evaluasi(data_baru)
        st.success("✅ Data baru berhasil dimuat & model sudah dilatih. Cek tab **Analisis Performa Model**.")

    if st.session_state.data is not None:
        st.write("**Contoh 5 baris data:**")
        st.dataframe(st.session_state.data.head(), use_container_width=True)

# -----------------------------
# Tab: Prediksi
# -----------------------------
with tab_prediksi:
    st.subheader("🔍 Prediksi (Uji Coba)")
    st.info("Masukkan data manual atau unggah CSV kecil untuk uji prediksi. Skor prediksi berada pada **skala 0–10** (semakin tinggi → risiko lebih besar).")

    pilihan_cara = st.radio("Pilih cara memasukkan data:", ["Isi Manual", "Unggah File CSV"])

    if pilihan_cara == "Isi Manual":
        if st.session_state.data is None or "sistem_rf" not in st.session_state:
            st.error("Model belum dilatih. Unggah dataset di tab **Lihat Data** terlebih dahulu.")
        else:
            data = st.session_state.data
            kolom_masuk = [c for c in data.columns if c not in ["Student_ID", "Addicted_Score"]]

            st.write("**Isi form berikut sesuai variabel pada dataset:**")
            data_masuk = {}
            for kol in kolom_masuk:
                if data[kol].dtype == "object":
                    # Pilihan dari nilai unik + opsi lain
                    opsi = sorted([str(x) for x in data[kol].dropna().unique().tolist()])
                    data_masuk[kol] = st.selectbox(f"{kol}", opsi)
                else:
                    minv = float(np.nanmin(data[kol].values)) if data[kol].notna().any() else 0.0
                    maxv = float(np.nanmax(data[kol].values)) if data[kol].notna().any() else 10.0
                    defaultv = float(np.nanmean(data[kol].values)) if data[kol].notna().any() else 0.0
                    data_masuk[kol] = st.number_input(f"{kol} (min: {minv:.2f}, maks: {maxv:.2f})",
                                                      value=defaultv)

            if st.button("🔎 Jalankan Prediksi"):
                df_in = pd.DataFrame([data_masuk])
                rf = st.session_state.sistem_rf.predict(df_in)[0]
                lr = st.session_state.sistem_lr.predict(df_in)[0]

                st.success("**Hasil Prediksi (Skala 0–10):**")
                st.write(f"🌳 Random Forest: **{rf:.2f}**")
                st.write(f"📈 Regresi Logistik: **{lr:.2f}**")

                # Kategori risiko skala 0–10
                batas = max(rf, lr)
                if batas > 7:
                    st.warning("⚠️ Risiko **tinggi**. Pertimbangkan pengurangan durasi penggunaan dan perbaikan sleep hygiene.")
                elif batas >= 5:
                    st.info("ℹ️ Risiko **sedang**. Pantau pola penggunaan agar tetap seimbang.")
                else:
                    st.success("✅ Risiko **rendah**. Kebiasaan masih terkendali.")

                simpan_riwayat(data_masuk, rf, lr, "Manual")

    else:
        file_csv = st.file_uploader("Unggah CSV untuk prediksi batch", type="csv")
        if file_csv and ("sistem_rf" in st.session_state):
            df_in = pd.read_csv(file_csv)
            X_pred = df_in.drop(columns=["Student_ID", "Addicted_Score"], errors="ignore")
            rf = st.session_state.sistem_rf.predict(X_pred)
            lr = st.session_state.sistem_lr.predict(X_pred)

            out = df_in.copy()
            out["Pred_RF"] = rf
            out["Pred_LogReg"] = lr

            st.write("**Hasil Prediksi:**")
            st.dataframe(out, use_container_width=True)

            # Simpan tiap baris ke riwayat
            for _, row in out.iterrows():
                data_in = row.drop(labels=["Pred_RF", "Pred_LogReg"]).to_dict()
                simpan_riwayat(data_in, row["Pred_RF"], row["Pred_LogReg"], "CSV")

# -----------------------------
# Tab: Riwayat
# -----------------------------
with tab_riwayat:
    st.subheader("📜 Riwayat Input & Prediksi")
    st.info("Riwayat disimpan di SQLite. (Catatan: saat ini riwayat dihapus saat aplikasi pertama kali dijalankan karena fungsi *perbaiki_database()*.)")

    riwayat = lihat_riwayat(50)
    if not riwayat.empty:
        # tampilkan ringkas + expandable detail
        riwayat_tampil = riwayat.copy()
        riwayat_tampil["data_masuk"] = riwayat_tampil["data_masuk"].apply(lambda x: json.loads(x))
        st.dataframe(riwayat_tampil[["waktu", "cara_input", "hasil_rf", "hasil_lr"]], use_container_width=True)

        with st.expander("Lihat detail data_masuk"):
            st.json(riwayat_tampil["data_masuk"].tolist())
    else:
        st.warning("Belum ada riwayat.")

# -----------------------------
# Tab: Analisis Performa Model
# -----------------------------
with tab_analisis:
    st.subheader("📈 Analisis Performa Model")
    st.info("Bandingkan akurasi, error, dan fitur terpenting dari Random Forest vs Regresi Logistik.")

    if "split_info" in st.session_state:
        st.write("### 📦 Ringkasan Pembagian Data (80% : 20%)")
        split_df = pd.DataFrame(list(st.session_state.split_info.items()),
                                columns=["Keterangan", "Jumlah Baris"])
        st.dataframe(split_df, use_container_width=True)

    if ("data" in st.session_state) and ("metrik_rf" in st.session_state):
        metrik_df = pd.DataFrame({
            "Metode": ["Random Forest", "Regresi Logistik"],
            "MSE": [st.session_state.metrik_rf["MSE"], st.session_state.metrik_lr["MSE"]],
            "RMSE": [st.session_state.metrik_rf["RMSE"], st.session_state.metrik_lr["RMSE"]],
            "MAE": [st.session_state.metrik_rf["MAE"], st.session_state.metrik_lr["MAE"]],
            "R²": [st.session_state.metrik_rf["R²"], st.session_state.metrik_lr["R²"]]
        })
        st.write("**Tabel Performa Model:**")
        st.dataframe(metrik_df, use_container_width=True)

        st.markdown("""
**Keterangan Metrik**
- **MSE**: Rata-rata kuadrat kesalahan. Makin kecil → makin akurat.
- **RMSE**: Akar MSE (satuan sama dengan target). Makin kecil → makin baik.
- **MAE**: Rata-rata selisih absolut. Makin kecil → makin baik.
- **R²**: Proporsi variasi target yang dijelaskan model (0–1). Makin mendekati 1 → makin baik.
        """)

        # Grafik perbandingan metrik
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].bar(metrik_df["Metode"], metrik_df["MSE"])
        axes[0].set_title("MSE")

        axes[1].bar(metrik_df["Metode"], metrik_df["RMSE"])
        axes[1].set_title("RMSE")

        axes[2].bar(metrik_df["Metode"], metrik_df["MAE"])
        axes[2].set_title("MAE")

        axes[3].bar(metrik_df["Metode"], metrik_df["R²"])
        axes[3].set_title("R²")
        axes[3].set_ylim(0, 1.05)

        plt.tight_layout()
        st.pyplot(fig)

        # Feature importance RF
        st.write("**Fitur Terpenting (Random Forest):**")
        fi = st.session_state.feature_importance[:10]  # top 10
        if fi:
            feature_names, importances = zip(*fi)
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.barh(feature_names, importances)
            ax2.set_xlabel("Tingkat Kepentingan")
            ax2.set_title("Top 10 Fitur Paling Berpengaruh")
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("Feature importance tidak tersedia.")
    else:
        st.warning("Model belum dilatih. Silakan unggah dataset di tab **Lihat Data**.")
