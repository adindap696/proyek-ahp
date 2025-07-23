import streamlit as st
import numpy as np
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Setup koneksi ke Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)


def ahp_weights(matrix):
    # Normalisasi matriks
    norm_matrix = matrix / np.sum(matrix, axis=0)
    
    # Hitung bobot prioritas (rata-rata dari setiap baris matriks ternormalisasi)
    weights = np.mean(norm_matrix, axis=1)
    
    # Hitung Aw (hasil perkalian matriks A dan vektor bobot)
    Aw = np.dot(matrix, weights)
    
    # Rasio Aw / w
    ratio = Aw / weights
    
    # Lambda maks adalah rata-rata dari rasio
    lamda_max = np.mean(ratio)
    
    return weights, lamda_max

def consistency_ratio(matrix, weights, lamda_max):
    n = matrix.shape[0]
    CI = (lamda_max - n) / (n - 1)
    RI_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_dict.get(n, 1.49)
    CR = 0 if RI == 0 else CI / RI
    return CI, RI, CR

def normalize_matrix(matrix):
    col_sum = np.sum(matrix, axis=0)
    norm_matrix = matrix / col_sum
    return norm_matrix

def classify_risk(total_score):
    if total_score < 1.0:
        return "Rendah"
    elif total_score < 2.34:
        return "Sedang"
    else:
        return "Tinggi"

# Aplikasi Streamlit
st.title("AHP untuk Klasifikasi Kerawanan Tanah Longsor")
st.header("1. Input Matriks Perbandingan Berpasangan Kriteria")

criteria = ["Curah Hujan", "Kemiringan Lereng", "Jenis Tanah", "Penggunaan Lahan", "Jenis Batuan"]
n = len(criteria)

st.write("Masukkan nilai perbandingan berpasangan antar kriteria (skala 1-9).")
st.write("Isikan nilai pada sel atas diagonal, sisanya akan terisi otomatis.")

matrix = np.ones((n, n))
for i in range(n):
    for j in range(i + 1, n):
        val = st.number_input(f"{criteria[i]} vs {criteria[j]}", min_value=1.0, max_value=9.0, value=1.0, step=1.0, key=f"{i}_{j}")
        matrix[i, j] = val
        matrix[j, i] = 1 / val

st.subheader("Matriks Perbandingan Berpasangan")

# Hitung jumlah per kolom
col_sums = matrix.sum(axis=0)

# Buat DataFrame dan tambahkan baris jumlah kolom
df_matrix = pd.DataFrame(matrix, index=criteria, columns=criteria)
df_matrix.loc['Jumlah'] = col_sums

# Tampilkan tabel
st.dataframe(df_matrix.style.format("{:.3f}"))

norm_matrix = normalize_matrix(matrix)
st.subheader("Matriks Normalisasi")

# Buat DataFrame dari matriks normalisasi
df_norm = pd.DataFrame(norm_matrix, index=criteria, columns=criteria)

# Hitung jumlah per kolom dan tambahkan sebagai baris terakhir
df_norm.loc['Jumlah'] = df_norm.sum(axis=0)

# Tampilkan tabel
st.dataframe(df_norm.style.format("{:.3f}"))

weights, lamda_max = ahp_weights(matrix)

st.subheader("Bobot Kriteria")

# Buat DataFrame bobot
df_weights = pd.DataFrame(weights, index=criteria, columns=["Bobot"])

# Tambahkan baris total bobot
df_weights.loc["Total"] = df_weights["Bobot"].sum()

# Tampilkan tabel
st.dataframe(df_weights.style.format("{:.3f}"))

CI, RI, CR = consistency_ratio(matrix, weights, lamda_max)

st.subheader("Perhitungan Konsistensi")
st.write(f"Nilai λ maks ≈ {lamda_max:.3f}")
st.write(f"Consistency Index (CI): {CI:.3f}")
st.write(f"Random Index (RI): {RI:.3f}")
st.write(f"Consistency Ratio (CR): {CR:.3f}")

if CR <= 0.1:
    st.success("Konsistensi matriks diterima (CR ≤ 0.1).")
else:
    st.error("Konsistensi matriks tidak diterima (CR > 0.1). Silakan perbaiki input perbandingan.")
    st.stop()

st.header("2. Input Skor Kriteria Kerawanan untuk Alternatif")

skor_ranges = {
    "Kemiringan Lereng": {
        "0 - 8%": 1,
        "8 - 15%": 2,
        "15 - 25%": 3,
        "25 - 40%": 4,
        "> 40%": 5
    },
    "Curah Hujan": {
        "< 1000 mm": 1,
        "1000 - 1500 mm": 2,
        "1500 - 2500 mm": 3,
        "2500 - 3500 mm": 4,
        "> 3500 mm": 5
    },
    "Jenis Tanah": {
        "Alluvial": 1,
        "Latosol": 2,
        "Mediteran": 3,
        "Grumosol, Andosol": 4,
        "Litosol, Organosol": 5
    },
    "Penggunaan Lahan": {
        "Daerah perkotaan dan pemukiman": 1,
        "Hutan": 2,
        "Lahan perkebunan": 3,
        "Lahan pertanian": 4,
        "Padang rumput, semak belukar": 5
    },
    "Jenis Batuan": {
        "Aluvial": 1,
        "Kapur": 2,
        "Granit": 3,
        "Sedimen": 4,
        "Basal, Vulkanik": 5
    }
}

# Setelah memilih skor kriteria
scores = []
for c in criteria:
    pilihan = st.selectbox(f"Pilih kondisi untuk {c}", options=list(skor_ranges[c].keys()))
    skor = skor_ranges[c][pilihan]
    scores.append(skor)

st.subheader("Skor Kriteria yang Dipilih")
df_scores = pd.DataFrame([scores], columns=criteria)
st.dataframe(df_scores)

# Hitung skor total
total_score = np.dot(weights, scores)
st.subheader("Hasil Perhitungan Skor Total")
st.write(f"Skor Total: {total_score:.3f}")

risk_level = classify_risk(total_score)
st.subheader("Klasifikasi Tingkat Kerawanan Tanah Longsor")
st.write(f"Tingkat Kerawanan: **{risk_level}**")