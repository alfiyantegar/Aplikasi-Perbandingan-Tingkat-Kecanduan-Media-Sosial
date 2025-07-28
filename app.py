import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

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
import sqlite3
import json
import datetime

#########################
# Database Helpers       #
#########################

def init_db():
    conn = sqlite3.connect("predictions.db")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_json TEXT,
            predicted_score REAL
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(df_row: pd.DataFrame, score: float):
    conn = sqlite3.connect("predictions.db")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (timestamp, input_json, predicted_score) VALUES (?, ?, ?)",
        (
            datetime.datetime.now().isoformat(timespec="seconds"),
            df_row.to_json(orient="records"),
            score,
        ),
    )
    conn.commit()
    conn.close()


def load_history() -> pd.DataFrame:
    conn = sqlite3.connect("predictions.db")
    df_hist = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df_hist

init_db()

###############################
# Streamlit Config & Styling  #
###############################

st.set_page_config(page_title="Prediktor Kecanduan Media Sosial", layout="wide", page_icon="📱")

# Global CSS tweaks
st.markdown(
    """
    <style>
    /* hero */
    .hero{background:linear-gradient(135deg,#6366f1 0%,#7c3aed 35%,#d946ef 100%);color:#fff;padding:3rem 1.5rem;border-radius:1.25rem;text-align:center;margin-bottom:2rem;box-shadow:0 6px 20px rgba(0,0,0,.15)}
    /* card */
    .card{background:#ffffff;border-radius:1rem;box-shadow:0 4px 12px rgba(0,0,0,.1);padding:1.5rem;margin-bottom:1.5rem}
    /* hide hamburger */
    header[data-testid="stHeader"] {visibility:hidden;}
    /* nicer tabs */
    div[role="tablist"] > button {font-size:1.05rem;padding:0.75rem 1.25rem;border-radius:0.5rem 0.5rem 0 0;color:#4b5563;font-weight:600}
    div[role="tablist"] > button[aria-selected="true"]{background:#6366f1;color:#fff}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv("Students Social Media Addiction.csv")

############################
# Pre‑processing & Model   #
############################

def preprocess(df: pd.DataFrame):
    X = df.drop(columns=["Student_ID", "Addicted_Score"], errors="ignore")
    y = df["Addicted_Score"]

    num = [
        "Age",
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
        "Conflicts_Over_Social_Media",
    ]
    cat = [
        "Gender",
        "Academic_Level",
        "Country",
        "Most_Used_Platform",
        "Affects_Academic_Performance",
        "Relationship_Status",
    ]

    ct = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat),
    ])
    return X, y, ct

@st.cache_resource(show_spinner=True)
def train(X, y, _ct):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([
        ("pre", _ct),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    m = {
        "mse": mean_squared_error(y_te, y_pred),
        "mae": mean_absolute_error(y_te, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_te, y_pred)),
        "r2": r2_score(y_te, y_pred),
    }
    m["cv_mse"] = -cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error").mean()
    m["cv_mae"] = -cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error").mean()
    m["cv_rmse"] = np.sqrt(m["cv_mse"])
    m["cv_r2"] = cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()
    return pipe, m

####################
# Main Application #
####################

def main():
    # Sidebar only for dataset upload
    uploaded_file = st.sidebar.file_uploader(
        "📂 Upload CSV (opsional)",
        type=["csv"],
        help="Dataset harus memiliki kolom 'Addicted_Score' sebagai target prediksi.",
    )

    df = load_data(uploaded_file)
    if "Addicted_Score" not in df.columns:
        st.error("Dataset harus memiliki kolom 'Addicted_Score'.")
        return

    X, y, ct = preprocess(df)
    model, metrics = train(X, y, ct)

    # Tabs navigation across the top
    tab_dash, tab_pred, tab_analysis = st.tabs(["🏠 Dashboard", "🔮 Prediksi", "📊 Analisis Data"])

    ################ Dashboard ################
    with tab_dash:
        st.markdown(
            '<div class="hero"><h1>📱 Prediktor Kecanduan Media Sosial</h1><p>Prediksi tingkat kecanduan mahasiswa & dapatkan wawasan personal untuk membentuk kebiasaan digital yang lebih sehat.</p></div>',
            unsafe_allow_html=True,
        )
        st.subheader("Mengapa aplikasi ini?")
        st.markdown(
            """
            - **Insight cepat**: dapatkan skor kecanduan dengan beberapa input saja.
            - **Riwayat tersimpan**: lacak perubahan perilaku dari waktu ke waktu.
            - **Dataset kustom**: unggah CSV Anda, model langsung menyesuaikan.
            """
        )
        st.success("Beralih ke tab 🔮 Prediksi untuk mencoba!")

    ################ Prediction ###############
    with tab_pred:
        st.header("Prediksi Skor Kecanduan")
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Usia", 15, 30, 20)
                gender = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
                academic = st.selectbox("Tingkat Akademik", ["SMA", "Sarjana", "Pascasarjana"])
                country = st.selectbox("Negara", df["Country"].unique())
                usage = st.slider("Jam Penggunaan Harian", 0.0, 10.0, 4.0, 0.5, help="Rata‑rata jam menggunakan media sosial per hari")
            with c2:
                platform = st.selectbox("Platform Utama", df["Most_Used_Platform"].unique())
                affects = st.selectbox("Mempengaruhi Akademik?", ["Ya", "Tidak"])
                sleep = st.slider("Jam Tidur", 0.0, 12.0, 6.0, 0.5, help="Rata‑rata jam tidur per malam")
                mental = st.slider("Skor Kesehatan Mental", 1, 10, 7, help="1 = sangat buruk, 10 = sangat baik")
                relation = st.selectbox("Status Hubungan", ["Lajang", "Berpacaran", "Rumit"])
                conflict = st.slider("Konflik (0‑5)", 0, 5, 2, help="0 = tidak pernah, 5 = sangat sering")
            pred_btn = st.form_submit_button("Prediksi")

        if pred_btn:
            new_df = pd.DataFrame({
                "Age": [age],
                "Gender": [gender],
                "Academic_Level": [academic],
                "Country": [country],
                "Avg_Daily_Usage_Hours": [usage],
                "Most_Used_Platform": [platform],
                "Affects_Academic_Performance": [affects],
                "Sleep_Hours_Per_Night": [sleep],
                "Mental_Health_Score": [mental],
                "Relationship_Status": [relation],
                "Conflicts_Over_Social_Media": [conflict],
            })
            score = model.predict(new_df)[0]
            st.success(f"Skor Kecanduan Diprediksi: {score:.2f}")
            if score >= 8:
                st.warning("Risiko kecanduan **tinggi**! Kurangi penggunaan dan pertimbangkan bantuan profesional.")
            elif score >= 6:
                st.info("Risiko kecanduan **sedang**. Tetapkan batas waktu penggunaan.")
            else:
                st.success("Risiko kecanduan **rendah**. Pertahankan kebiasaan baik!")
            st.caption("Skor <6 = rendah, 6‑7.99 = sedang, ≥8 = tinggi")
            save_prediction(new_df, float(score))

        st.subheader("Riwayat Prediksi")
        hist = load_history()
        if hist.empty:
            st.info("Belum ada riwayat.")
        else:
            def flat(r):
                rec = json.loads(r["input_json"])[0]
                rec.update({"predicted_score": r["predicted_score"], "timestamp": r["timestamp"]})
                return pd.Series(rec)
            st.dataframe(hist.apply(flat, axis=1), use_container_width=True)

    ################ Analysis ################
    with tab_analysis:
        st.header("Analisis Data")
        st.subheader("Ukuran & Pratinjau Dataset")
        st.write(f"Baris: {df.shape[0]} | Kolom: {df.shape[1]}")
        st.dataframe(df.head())

        st.subheader("Performa Model (Hold‑out)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", f"{metrics['mse']:.2f}")
        col2.metric("MAE", f"{metrics['mae']:.2f}")
        col3.metric("RMSE", f"{metrics['rmse']:.2f}")
        col4.metric("R²", f"{metrics['r2']:.2f}")

        st.subheader("Cross‑Validation (5‑Fold)")
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("CV MSE", f"{metrics['cv_mse']:.2f}")
        col6.metric("CV MAE", f"{metrics['cv_mae']:.2f}")
        col7.metric("CV RMSE", f"{metrics['cv_rmse']:.2f}")
        col8.metric("CV R²", f"{metrics['cv_r2']:.2f}")

        st.subheader("Distribusi Skor Kecanduan")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.histplot(df["Addicted_Score"], kde=True, ax=ax1)
        st.pyplot(fig1)

        st.subheader("Feature Importance (Random Forest)")
        importances = model.named_steps["rf"].feature_importances_
        feature_names = model.named_steps["pre"].get_feature_names_out()
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(15)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x="importance", y="feature", data=imp_df, ax=ax2)
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("")
        st.pyplot(fig2)


if __name__ == "__main__":
    main()