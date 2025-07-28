import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import shap

# Mengatur konfigurasi halaman
st.set_page_config(page_title="Prediktor Kecanduan Media Sosial", layout="wide")

# Memuat dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Students Social Media Addiction.csv")
    return df

# Pra-pemrosesan data
def preprocess_data(df):
    # Menentukan fitur dan target
    X = df.drop(columns=["Student_ID", "Addicted_Score"])
    y = df["Addicted_Score"]
    
    # Menentukan kolom numerik dan kategorikal
    numeric_features = ["Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", 
                       "Mental_Health_Score", "Conflicts_Over_Social_Media"]
    categorical_features = ["Gender", "Academic_Level", "Country", "Most_Used_Platform", 
                          "Affects_Academic_Performance", "Relationship_Status"]
    
    # Membuat pipeline pra-pemrosesan
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ])
    
    return X, y, preprocessor

# Melatih model
@st.cache_resource
def train_model(X, y, _preprocessor):
    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Membuat pipeline
    model = Pipeline(steps=[
        ("preprocessor", _preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Melatih model
    model.fit(X_train, y_train)
    
    # Mengevaluasi model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation (5-fold)
    cv_mse = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error").mean()
    cv_mae = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error").mean()
    cv_rmse = np.sqrt(cv_mse)
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
    
    return model, mse, mae, rmse, r2, cv_mse, cv_mae, cv_rmse, cv_r2, X_train, X_test, y_test, y_pred

# Aplikasi utama
def main():
    st.title("Prediktor Kecanduan Media Sosial")
    st.write("Aplikasi ini memprediksi skor kecanduan media sosial mahasiswa berdasarkan berbagai faktor.")
    
    # Memuat data
    df = load_data()
    X, y, preprocessor = preprocess_data(df)
    
    # Melatih model
    model, mse, mae, rmse, r2, cv_mse, cv_mae, cv_rmse, cv_r2, X_train, X_test, y_test, y_pred = train_model(X, y, preprocessor)
    
    # Sidebar untuk navigasi
    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox("Pilih halaman", ["Dashboard", "Prediksi", "Analisis Data", "Interpretasi Model"])
    
    if page == "Dashboard":
        st.header("Dashboard: Ikhtisar Proyek")
        st.markdown("""
        ### Informasi Proyek
        **Judul**: Prediksi Tingkat Kecanduan Media Sosial Mahasiswa Menggunakan Machine Learning dan Visualisasi Interaktif Berbasis Streamlit

        **Tujuan**:  
        Proyek ini bertujuan untuk memprediksi tingkat kecanduan media sosial di kalangan mahasiswa menggunakan machine learning dan menyediakan aplikasi web interaktif untuk memvisualisasikan wawasan data serta memberikan rekomendasi yang dipersonalisasi. Aplikasi ini membantu mahasiswa, pendidik, dan konselor memahami faktor-faktor yang berkontribusi pada kecanduan media sosial dan mendorong kebiasaan digital yang lebih sehat.

        **Deskripsi Dataset**:  
        Dataset *Students Social Media Addiction.csv* berisi informasi tentang penggunaan media sosial mahasiswa dan dampaknya terhadap performa akademik, kesehatan mental, dan gaya hidup. Fitur utama meliputi:
        - **Usia**: Usia mahasiswa (15–30 tahun).
        - **Jenis Kelamin**: Pria atau Wanita.
        - **Tingkat Akademik**: SMA, Sarjana, atau Pascasarjana.
        - **Negara**: Negara tempat tinggal.
        - **Rata-rata Jam Penggunaan Harian**: Jam rata-rata penggunaan media sosial per hari.
        - **Platform Paling Sering Digunakan**: Platform media sosial utama.
        - **Mempengaruhi Performa Akademik**: Apakah media sosial memengaruhi performa akademik (Ya/Tidak).
        - **Jam Tidur per Malam**: Rata-rata jam tidur per malam.
        - **Skor Kesehatan Mental**: Skor kesehatan mental yang dilaporkan sendiri (1–10).
        - **Status Hubungan**: Lajang, Berpacaran, atau Rumit.
        - **Konflik Akibat Media Sosial**: Jumlah konflik yang disebabkan oleh media sosial (0–5).
        - **Skor Kecanduan**: Variabel target, mewakili tingkat kecanduan (1–10).

        ### Penjelasan Rentang Nilai Fitur Utama
        **Konflik Akibat Media Sosial (0-5)**:  
        Mengukur jumlah konflik atau pertengkaran akibat penggunaan media sosial (misalnya, dengan teman, keluarga, atau pasangan).  
        - **0**: Tidak ada konflik.  
        - **1**: Konflik sangat jarang (misalnya, salah paham kecil).  
        - **2**: Konflik ringan atau sesekali (misalnya, adu argumen ringan).  
        - **3**: Konflik sedang (pertengkaran lebih sering).  
        - **4**: Konflik sering (mengganggu hubungan sosial).  
        - **5**: Konflik sangat sering atau parah (dampak serius pada hubungan).  
        Nilai yang lebih tinggi menunjukkan dampak negatif media sosial yang lebih besar pada hubungan interpersonal.

        **Skor Kesehatan Mental (1-10)**:  
        Skor yang dilaporkan sendiri untuk menilai kondisi kesehatan mental, seperti stres, kecemasan, atau kebahagiaan.  
        - **1-2**: Kesehatan mental sangat buruk (stres berat atau depresi).  
        - **3-4**: Kesehatan mental buruk (sering cemas atau sedih).  
        - **5-6**: Kesehatan mental sedang (cukup stabil, kadang stres).  
        - **7-8**: Kesehatan mental baik (umumnya bahagia dan tenang).  
        - **9-10**: Kesehatan mental sangat baik (optimis dan sejahtera).  
        Skor yang lebih rendah menunjukkan kesehatan mental yang lebih buruk, yang mungkin terkait dengan penggunaan media sosial berlebihan.

        **Skor Kecanduan (1-10)**:  
        Mengukur tingkat kecanduan media sosial berdasarkan perilaku, seperti frekuensi penggunaan atau dampak negatif pada kehidupan sehari-hari.  
        - **1-3**: Risiko kecanduan sangat rendah (penggunaan terkontrol).  
        - **4-5**: Risiko kecanduan rendah (penggunaan cukup teratur).  
        - **6-7**: Risiko kecanduan sedang (mengganggu produktivitas atau tidur).  
        - **8-10**: Risiko kecanduan tinggi (dampak negatif signifikan pada kehidupan).  
        Skor yang lebih tinggi menunjukkan kecanduan yang lebih serius, digunakan untuk memberikan rekomendasi dalam aplikasi.

        **Metodologi**:  
        - **Pra-pemrosesan Data**: Menangani nilai yang hilang, mengkodekan variabel kategorikal (OneHotEncoder), dan menskalakan fitur numerik (StandardScaler).
        - **Model**: Random Forest Regressor untuk memprediksi `Skor Kecanduan`.
        - **Evaluasi**: Metrik meliputi Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan R² Score.
        - **Visualisasi**: Plot interaktif (histogram, heatmap, scatter plot) dan interpretasi model menggunakan SHAP dan kepentingan fitur.
        - **Aplikasi**: Dibangun dengan Streamlit untuk prediksi dan eksplorasi data yang ramah pengguna.

        **Fitur Utama Aplikasi**:  
        - **Prediksi**: Masukkan data pribadi untuk memprediksi skor kecanduan dan menerima rekomendasi.  
        - **Analisis Data**: Visualisasikan tren dan korelasi dalam dataset.  
        - **Interpretasi Model**: Pahami faktor-faktor utama yang memengaruhi prediksi menggunakan plot SHAP dan kepentingan fitur.  
        - **Dashboard**: Ikhtisar proyek dan tujuannya.

        **Dikembangkan oleh**: Alfiyan Tegar Budi Satria untuk Tujuan Skripsi di Universitas Duta Bangsa  
        **Tanggal**: 18 Juni 2025
        """)

    elif page == "Prediksi":
        st.header("Prediksi Skor Kecanduan Media Sosial Anda")
        
        # Input pengguna
        col1, col2 = st.columns(2)
        with col1:
            usia = st.number_input("Usia", min_value=15, max_value=30, value=20)
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
            tingkat_akademik = st.selectbox("Tingkat Akademik", ["SMA", "Sarjana", "Pascasarjana"])
            negara = st.selectbox("Negara", df["Country"].unique())
            rata_rata_jam_penggunaan = st.slider("Rata-rata Jam Penggunaan Harian", 0.0, 10.0, 4.0)
        
        with col2:
            platform_utama = st.selectbox("Platform Paling Sering Digunakan", df["Most_Used_Platform"].unique())
            memengaruhi_akademik = st.selectbox("Mempengaruhi Performa Akademik?", ["Ya", "Tidak"])
            jam_tidur = st.slider("Jam Tidur per Malam", 0.0, 12.0, 6.0)
            skor_kesehatan_mental = st.slider("Skor Kesehatan Mental (1-10)", 1, 10, 7)
            status_hubungan = st.selectbox("Status Hubungan", ["Lajang", "Berpacaran", "Rumit"])
            konflik = st.slider("Konflik Akibat Media Sosial (0-5)", 0, 5, 2)
        
        # Membuat dataframe input
        input_data = pd.DataFrame({
            "Age": [usia],
            "Gender": [jenis_kelamin],
            "Academic_Level": [tingkat_akademik],
            "Country": [negara],
            "Avg_Daily_Usage_Hours": [rata_rata_jam_penggunaan],
            "Most_Used_Platform": [platform_utama],
            "Affects_Academic_Performance": [memengaruhi_akademik],
            "Sleep_Hours_Per_Night": [jam_tidur],
            "Mental_Health_Score": [skor_kesehatan_mental],
            "Relationship_Status": [status_hubungan],
            "Conflicts_Over_Social_Media": [konflik]
        })
        
        # Membuat prediksi
        if st.button("Prediksi"):
            prediksi = model.predict(input_data)[0]
            st.success(f"Skor Kecanduan yang Diprediksi: {prediksi:.2f}")
            
            # Memberikan rekomendasi
            if prediksi >= 8:
                st.warning("Risiko kecanduan tinggi! Pertimbangkan untuk mengurangi penggunaan media sosial dan mencari saran profesional.")
            elif prediksi >= 6:
                st.info("Risiko kecanduan sedang. Cobalah menetapkan batas waktu penggunaan media sosial.")
            else:
                st.success("Risiko kecanduan rendah. Pertahankan gaya hidup seimbang!")
        
    elif page == "Analisis Data":
        st.header("Analisis Data")
        
        # Metrik Evaluasi Model
        st.subheader("Metrik Evaluasi Model")
        st.write(f"**Mean Squared Error (MSE)**: {mse:.4f}")
        st.write(f"**Mean Absolute Error (MAE)**: {mae:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE)**: {rmse:.4f}")
        st.write(f"**R² Score**: {r2:.4f}")
        st.write("Metrik ini mengevaluasi performa model Random Forest pada data pengujian. MSE, MAE, dan RMSE yang lebih rendah menunjukkan akurasi yang lebih baik, sementara R² yang lebih tinggi (mendekati 1) menunjukkan kecocokan model yang baik.")
        
        # Metrik Cross-Validation
        st.subheader("Hasil Cross-Validation (5-Fold)")
        st.write(f"**Rata-rata MSE (Cross-Validation)**: {cv_mse:.4f}")
        st.write(f"**Rata-rata MAE (Cross-Validation)**: {cv_mae:.4f}")
        st.write(f"**Rata-rata RMSE (Cross-Validation)**: {cv_rmse:.4f}")
        st.write(f"**Rata-rata R² (Cross-Validation)**: {cv_r2:.4f}")
        st.write("Cross-validation menguji performa model pada subset data yang berbeda untuk memastikan model tidak *overfitting* dan dapat digeneralisasi dengan baik.")
        
        # Plot Residual
        st.subheader("Plot Residual")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals, ax=ax)
        ax.axhline(y=0, color="r", linestyle="--")
        ax.set_xlabel("Nilai Prediksi")
        ax.set_ylabel("Residual (Nilai Aktual - Prediksi)")
        ax.set_title("Plot Residual: Prediksi vs. Residual")
        st.pyplot(fig)
        st.write("Plot ini menunjukkan selisih antara nilai prediksi dan aktual. Titik-titik yang mendekati garis nol menunjukkan prediksi yang akurat.")
        
        # Plot Kepentingan Fitur
        st.subheader("Kepentingan Fitur (Random Forest)")
        feature_importance = model.named_steps["regressor"].feature_importances_
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        importance_df = pd.DataFrame({"Fitur": feature_names, "Kepentingan": feature_importance})
        importance_df = importance_df.sort_values(by="Kepentingan", ascending=False).head(10)  # Top 10 fitur
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Kepentingan", y="Fitur", data=importance_df, ax=ax)
        ax.set_title("10 Fitur Teratas Berdasarkan Kepentingan")
        st.pyplot(fig)
        st.write("Plot ini menunjukkan fitur yang paling berpengaruh dalam memprediksi skor kecanduan berdasarkan model Random Forest.")
        
        # Visualisasi distribusi Skor Kecanduan
        st.subheader("Distribusi Skor Kecanduan")
        fig, ax = plt.subplots()
        sns.histplot(df["Addicted_Score"], kde=True, ax=ax)
        ax.set_title("Distribusi Skor Kecanduan")
        st.pyplot(fig)
        
        # Visualisasi heatmap korelasi
        st.subheader("Heatmap Korelasi")
        numeric_cols = ["Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", 
                       "Mental_Health_Score", "Conflicts_Over_Social_Media", "Addicted_Score"]
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Heatmap Korelasi Fitur Numerik")
        st.pyplot(fig)
        
        # Scatter plot
        st.subheader("Jam Penggunaan Harian vs. Skor Kecanduan")
        fig, ax = plt.subplots()
        sns.scatterplot(x="Avg_Daily_Usage_Hours", y="Addicted_Score", hue="Gender", data=df, ax=ax)
        ax.set_title("Jam Penggunaan Harian vs. Skor Kecanduan Berdasarkan Jenis Kelamin")
        st.pyplot(fig)
        
    elif page == "Interpretasi Model":
        st.header("Interpretasi Model")
        
        # Nilai SHAP
        st.subheader("Kepentingan Fitur (SHAP)")
        explainer = shap.TreeExplainer(model.named_steps["regressor"])
        X_transformed = model.named_steps["preprocessor"].transform(X_train)
        shap_values = explainer.shap_values(X_transformed)
        
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_transformed, feature_names=model.named_steps["preprocessor"].get_feature_names_out())
        st.pyplot(fig)
        st.write("Plot ini menunjukkan dampak setiap fitur terhadap prediksi. Warna merah menunjukkan nilai fitur yang tinggi, dan biru menunjukkan nilai fitur yang rendah.")

if __name__ == "__main__":
    main()