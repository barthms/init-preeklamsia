import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Preeklamsia (Data Sintetis)",
    page_icon="🩺",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent.parent   # = proyek-DASI-DALIS/
CSV_PATH = BASE_DIR / "dataset" / "dummy" / "dataset_sintetis_preeklamsia_1000_bersih_tanpa_null.csv"
# CSV_PATH = "../dataset/dummy/dataset_sintetis_preeklamsia_1000_bersih_tanpa_null.csv"
PLACEHOLDER = ["TIDAK_DIPERIKSA", "BELUM_DINILAI", "TIDAK_ADA", "TIDAK_BERLAKU"]
NUMERIC_COLS = [
    "hemoglobin_g_dl", "blood_glucose_mg_dl", "platelet_count",
    "creatinine_mg_dl", "ast_u_l", "alt_u_l", "tfu_cm",
]
DATE_COLS = ["visit_date", "delivery_date"]
WARNA = {"Non-preeklamsia": "#4C78A8", "Preeklamsia": "#E45756"}


# ---------------------------------------------------------------
# 2. LOAD & BERSIHKAN DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Nilai penanda "tidak diperiksa" dll. diubah jadi NaN agar tidak merusak grafik
    df = df.replace(PLACEHOLDER, np.nan)
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def pe_rate(data: pd.DataFrame, by: str) -> pd.DataFrame:
    """Hitung jumlah pasien & persentase preeklamsia per kelompok."""
    t = (
        data.groupby(by)
        .agg(total=("patient_id", "count"),
             pe=("final_diagnosis", lambda s: (s == "Preeklamsia").sum()))
        .reset_index()
    )
    t["persen_pe"] = t["pe"] / t["total"] * 100
    return t


df = load_data(CSV_PATH)

# ---------------------------------------------------------------
# 3. SIDEBAR FILTER
# ---------------------------------------------------------------
st.sidebar.header("Filter Data")

faskes = st.sidebar.multiselect(
    "Fasilitas kesehatan",
    sorted(df["facility_name"].unique()),
    default=sorted(df["facility_name"].unique()),
)
diagnosis = st.sidebar.multiselect(
    "Diagnosis akhir",
    sorted(df["final_diagnosis"].unique()),
    default=sorted(df["final_diagnosis"].unique()),
)
risiko = st.sidebar.multiselect(
    "Kategori risiko skrining",
    ["Rendah", "Sedang", "Tinggi"],
    default=["Rendah", "Sedang", "Tinggi"],
)
umur_min, umur_max = int(df["maternal_age"].min()), int(df["maternal_age"].max())
umur = st.sidebar.slider("Usia ibu (tahun)", umur_min, umur_max, (umur_min, umur_max))

mask = (
    df["facility_name"].isin(faskes)
    & df["final_diagnosis"].isin(diagnosis)
    & df["screening_risk_category"].isin(risiko)
    & df["maternal_age"].between(*umur)
)
dff = df[mask]  # level kunjungan (1 baris = 1 kunjungan ANC)

if dff.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()

# Level pasien (1 baris = 1 pasien) -> untuk menghindari penghitungan ganda
pasien = dff.drop_duplicates("patient_id")

# ---------------------------------------------------------------
# 4. JUDUL & KPI
# ---------------------------------------------------------------
st.title("🩺 Dashboard Preeklamsia")
st.caption("Data sintetis — Kabupaten Toba. Untuk keperluan pembelajaran, bukan data pasien nyata.")

n_pe = int((pasien["final_diagnosis"] == "Preeklamsia").sum())
k1, k2, k3, k4 = st.columns(4)
k1.metric("Kunjungan ANC", f"{len(dff):,}")
k2.metric("Jumlah pasien", f"{len(pasien):,}")
k3.metric("Pasien preeklamsia", f"{n_pe:,}")
k4.metric("Proporsi preeklamsia", f"{n_pe / len(pasien) * 100:.1f}%")

st.divider()

# ---------------------------------------------------------------
# 5. TAB VISUALISASI
# ---------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Ringkasan", "Profil Ibu", "Klinis", "Luaran", "Data"]
)

# ---- TAB 1: RINGKASAN ----
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        dist = pasien["final_diagnosis"].value_counts().reset_index()
        dist.columns = ["Diagnosis", "Jumlah"]
        fig = px.pie(dist, names="Diagnosis", values="Jumlah", hole=0.45,
                     color="Diagnosis", color_discrete_map=WARNA,
                     title="Distribusi diagnosis akhir (per pasien)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        t = pe_rate(pasien, "facility_name")
        fig = px.bar(t, x="facility_name", y="persen_pe",
                     text=t["persen_pe"].round(1).astype(str) + "%",
                     title="Proporsi preeklamsia per fasilitas kesehatan",
                     labels={"facility_name": "Fasilitas", "persen_pe": "% preeklamsia"})
        st.plotly_chart(fig, width="stretch")

    t = pe_rate(pasien, "domicile_district").sort_values("persen_pe", ascending=False)
    fig = px.bar(t, x="domicile_district", y="persen_pe",
                 text=t["persen_pe"].round(1).astype(str) + "%",
                 title="Proporsi preeklamsia per kecamatan domisili",
                 labels={"domicile_district": "Kecamatan", "persen_pe": "% preeklamsia"})
    st.plotly_chart(fig, width="stretch")

# ---- TAB 2: PROFIL IBU ----
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(pasien, x="maternal_age", color="final_diagnosis", nbins=20,
                           barmode="overlay", opacity=0.7, color_discrete_map=WARNA,
                           title="Distribusi usia ibu",
                           labels={"maternal_age": "Usia ibu", "final_diagnosis": "Diagnosis"})
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.box(pasien, x="final_diagnosis", y="prepregnancy_bmi",
                     color="final_diagnosis", color_discrete_map=WARNA,
                     title="IMT sebelum hamil",
                     labels={"final_diagnosis": "Diagnosis", "prepregnancy_bmi": "IMT"})
        st.plotly_chart(fig, width="stretch")

    pilihan = {
        "Pendidikan": "education_level",
        "Pekerjaan": "occupation",
        "Status sosial ekonomi": "socioeconomic_status",
        "Paritas (nullipara)": "nulliparous_status",
    }
    label = st.selectbox("Lihat proporsi preeklamsia berdasarkan:", list(pilihan))
    t = pe_rate(pasien, pilihan[label])
    fig = px.bar(t, x=pilihan[label], y="persen_pe",
                 text=t["persen_pe"].round(1).astype(str) + "%",
                 labels={pilihan[label]: label, "persen_pe": "% preeklamsia"},
                 title=f"Proporsi preeklamsia menurut {label.lower()}")
    st.plotly_chart(fig, width="stretch")

    faktor = {
        "previous_preeclampsia": "Riwayat PE sebelumnya",
        "family_history_preeclampsia": "Riwayat PE keluarga",
        "chronic_hypertension": "Hipertensi kronis",
        "preexisting_diabetes": "Diabetes sebelum hamil",
        "gestational_diabetes": "Diabetes gestasional",
        "kidney_disease": "Penyakit ginjal",
        "sle": "SLE",
    }
    rows = []
    for col, nama in faktor.items():
        for grp, g in pasien.groupby("final_diagnosis"):
            rows.append({"Faktor risiko": nama, "Diagnosis": grp,
                         "Persen": (g[col] == "Ya").mean() * 100})
    fig = px.bar(pd.DataFrame(rows), x="Faktor risiko", y="Persen", color="Diagnosis",
                 barmode="group", color_discrete_map=WARNA,
                 title="Prevalensi faktor risiko: preeklamsia vs non-preeklamsia (%)")
    st.plotly_chart(fig, width="stretch")

# ---- TAB 3: KLINIS ----
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(dff, x="final_diagnosis", y="systolic_bp", color="final_diagnosis",
                     color_discrete_map=WARNA, title="Tekanan darah sistolik per kunjungan",
                     labels={"final_diagnosis": "Diagnosis", "systolic_bp": "Sistolik (mmHg)"})
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.scatter(dff, x="systolic_bp", y="diastolic_bp", color="final_diagnosis",
                         color_discrete_map=WARNA, opacity=0.6,
                         title="Sistolik vs diastolik",
                         labels={"systolic_bp": "Sistolik", "diastolic_bp": "Diastolik",
                                 "final_diagnosis": "Diagnosis"})
        fig.add_hline(y=90, line_dash="dash", line_color="gray")
        fig.add_vline(x=140, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, width="stretch")

    # Tren tekanan darah menurut usia kehamilan
    tren = dff.copy()
    tren["Usia kehamilan"] = pd.cut(
        tren["gestational_age_weeks"], bins=[0, 12, 20, 28, 34, 42],
        labels=["<=12", "13-20", "21-28", "29-34", "35+ minggu"],
    )
    tren = (tren.groupby(["Usia kehamilan", "final_diagnosis"], observed=True)["systolic_bp"]
            .mean().reset_index())
    fig = px.line(tren, x="Usia kehamilan", y="systolic_bp", color="final_diagnosis",
                  markers=True, color_discrete_map=WARNA,
                  title="Rata-rata sistolik menurut usia kehamilan",
                  labels={"systolic_bp": "Rata-rata sistolik", "final_diagnosis": "Diagnosis"})
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        t = dff["screening_risk_category"].value_counts().reindex(["Rendah", "Sedang", "Tinggi"]).reset_index()
        t.columns = ["Kategori", "Jumlah"]
        fig = px.bar(t, x="Kategori", y="Jumlah", title="Kategori risiko skrining (kunjungan)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        t = dff["proteinuria_result"].fillna("Tidak diperiksa").value_counts().reset_index()
        t.columns = ["Hasil", "Jumlah"]
        fig = px.bar(t, x="Hasil", y="Jumlah", title="Hasil pemeriksaan proteinuria")
        st.plotly_chart(fig, width="stretch")

# ---- TAB 4: LUARAN ----
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        t = pasien.groupby(["final_diagnosis", "delivery_mode"]).size().reset_index(name="Jumlah")
        fig = px.bar(t, x="final_diagnosis", y="Jumlah", color="delivery_mode",
                     barmode="group", title="Cara persalinan",
                     labels={"final_diagnosis": "Diagnosis", "delivery_mode": "Cara persalinan"})
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.box(pasien, x="final_diagnosis", y="birth_weight_g", color="final_diagnosis",
                     color_discrete_map=WARNA, title="Berat badan lahir (gram)",
                     labels={"final_diagnosis": "Diagnosis", "birth_weight_g": "Berat lahir (g)"})
        st.plotly_chart(fig, width="stretch")

    luaran = {
        "prematurity_status": "Prematur",
        "bblr_status": "BBLR",
        "nicu_admission": "Masuk NICU",
        "neonatal_complication": "Komplikasi neonatal",
        "postpartum_hypertension_status": "Hipertensi pascasalin",
    }
    rows = []
    for col, nama in luaran.items():
        for grp, g in pasien.groupby("final_diagnosis"):
            rows.append({"Luaran": nama, "Diagnosis": grp,
                         "Persen": (g[col] == "Ya").mean() * 100})
    fig = px.bar(pd.DataFrame(rows), x="Luaran", y="Persen", color="Diagnosis",
                 barmode="group", color_discrete_map=WARNA,
                 title="Luaran ibu & bayi (% pasien)")
    st.plotly_chart(fig, width="stretch")

# ---- TAB 5: DATA ----
with tab5:
    st.write(f"Menampilkan {len(dff):,} baris sesuai filter.")
    st.dataframe(dff, width="stretch")
    st.download_button(
        "Unduh data terfilter (CSV)",
        dff.to_csv(index=False).encode("utf-8"),
        file_name="data_terfilter.csv",
        mime="text/csv",
    )