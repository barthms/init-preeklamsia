"""Dashboard Riwayat & Risiko Preeklamsia — data sintetis ANC Kabupaten Toba.

Jalankan dari root proyek:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =====================================================================
# 1. KONFIGURASI & SKEMA
# =====================================================================
# Dataset diasumsikan ada di: <root proyek>/dataset/dummy/dataset_sintetis_preeklamsia.csv
# app.py bisa diletakkan di root proyek ATAU di dalam folder app/.
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent if APP_DIR.name == "app" else APP_DIR

REL_CSV = Path("dataset") / "dummy" / "dataset_sintetis_preeklamsia.csv"
CSV_CANDIDATES = [
    PROJECT_DIR / REL_CSV,
    APP_DIR / REL_CSV,
    Path.cwd() / REL_CSV,
]
CSV_PATH = next((p for p in CSV_CANDIDATES if p.exists()), None)

if CSV_PATH is None:
    st.error(
        "Dataset tidak ditemukan. Dicari di:\n"
        + "\n".join(f"  - {p}" for p in CSV_CANDIDATES)
        + "\n\nPastikan file `dataset_sintetis_preeklamsia.csv` ada di "
          "`<root proyek>/dataset/dummy/`."
    )
    st.stop()
PLACEHOLDER = ["TIDAK_DIPERIKSA", "BELUM_DINILAI", "TIDAK_ADA",
               "TIDAK_BERLAKU", "NOT_APPLICABLE", "TIDAK_DIRUJUK"]

ID_COLS = ["patient_id", "pregnancy_id", "anc_visit_id", "facility_id", "baby_id"]
TEXT_COLS = [
    "facility_name", "facility_type", "facility_district",
    "maternal_domicile_district", "edema_location", "fetal_presentation",
    "referral_destination", "referral_reason_code", "preeclampsia_severity",
    "delivery_mode", "maternal_discharge_status", "baby_sex",
    "birth_status", "neonatal_outcome",
]
DATE_COLS = [
    "visit_date", "date_of_birth", "lmp_date", "estimated_due_date",
    "previous_delivery_date", "referral_date", "preeclampsia_diagnosis_date",
    "delivery_date", "postpartum_visit_date",
]
INT_COLS = [
    "visit_number", "maternal_age_years", "gravida", "parity", "abortion_count",
    "fetus_count", "systolic_bp_mmHg", "diastolic_bp_mmHg",
    "systolic_bp_change_mmHg", "diastolic_bp_change_mmHg", "prior_visit_count",
    "days_since_previous_visit", "fetal_heart_rate_bpm",
    "iron_tablets_given_count", "platelet_count_per_uL", "ast_u_l", "alt_u_l",
    "estimated_fetal_weight_g", "birth_order", "birth_weight_g",
    "apgar_1min", "apgar_5min", "postpartum_systolic_bp",
    "postpartum_diastolic_bp", "gestational_age_at_delivery_weeks",
]
FLOAT_COLS = [
    "height_cm", "prepregnancy_weight_kg", "prepregnancy_bmi",
    "interpregnancy_interval_years", "gestational_age_weeks", "weight_kg",
    "weight_change_from_baseline_kg", "lila_cm",
    "mean_arterial_pressure_mmHg", "tfu_cm", "hemoglobin_g_dl",
    "fasting_glucose_mg_dl", "glucose_2h_postprandial_mg_dl",
    "urine_protein_24h_mg", "urine_protein_creatinine_ratio",
    "serum_creatinine_mg_dl", "birth_length_cm",
]
BOOL_COLS = [
    "obesity_prepregnancy_flag", "nulliparity_flag", "new_partner_pregnancy_flag",
    "assisted_reproduction_flag", "multiple_pregnancy_flag",
    "previous_preeclampsia_flag", "family_history_preeclampsia_flag",
    "chronic_hypertension_flag", "preexisting_diabetes_flag",
    "gestational_diabetes_flag", "kidney_disease_flag",
    "systemic_lupus_erythematosus_flag", "antiphospholipid_syndrome_flag",
    "edema_present", "edema_persistent_flag", "headache_present",
    "visual_disturbance_present", "epigastric_or_ruq_pain_present",
    "dyspnea_present", "proteinuria_examined_flag",
    "fetal_growth_restriction_flag", "umbilical_artery_doppler_abnormal_flag",
    "suspected_preeclampsia_flag", "eclampsia_flag", "hellp_syndrome_flag",
    "pulmonary_edema_flag", "placental_abruption_flag",
    "maternal_icu_admission_flag", "preterm_delivery_flag",
    "low_birth_weight_flag", "nicu_admission_flag",
]
ORDERED = {
    "trimester": ["TRIMESTER_1", "TRIMESTER_2", "TRIMESTER_3"],
    "pe_screening_risk_category": ["RENDAH", "SEDANG", "TINGGI"],
    "proteinuria_dipstick_result": ["NEGATIF", "TRACE", "+1", "+2", "+3", "+4"],
    "fetal_presentation": ["KEPALA", "BOKONG", "LINTANG"],
}

WARNA_DX = {"NON_PREECLAMPSIA": "#4C78A8", "PREECLAMPSIA": "#E45756"}
WARNA_RISIKO = {"RENDAH": "#9ecae9", "SEDANG": "#f2a65a", "TINGGI": "#d62728"}
WARNA_STATUS = {"Non-PE": "#9ecae9", "Belum PE": "#f2a65a", "PE terdiagnosis": "#d62728"}

FAKTOR = {
    "previous_preeclampsia_flag": "Riwayat PE sebelumnya",
    "family_history_preeclampsia_flag": "Riwayat PE pada ibu/saudara",
    "chronic_hypertension_flag": "Hipertensi kronik",
    "preexisting_diabetes_flag": "Diabetes sebelum hamil",
    "gestational_diabetes_flag": "Diabetes gestasional",
    "kidney_disease_flag": "Penyakit ginjal",
    "systemic_lupus_erythematosus_flag": "Autoimun / SLE",
    "antiphospholipid_syndrome_flag": "Sindrom antifosfolipid",
    "nulliparity_flag": "Nulipara (hamil pertama)",
    "multiple_pregnancy_flag": "Kehamilan kembar",
    "assisted_reproduction_flag": "Teknologi reproduksi berbantu",
    "obesity_prepregnancy_flag": "Obesitas sebelum hamil (IMT > 30)",
}
LAB = {
    "hemoglobin_g_dl": "Hemoglobin (g/dL)",
    "platelet_count_per_uL": "Trombosit (/uL)",
    "serum_creatinine_mg_dl": "Kreatinin (mg/dL)",
    "ast_u_l": "AST (U/L)",
    "alt_u_l": "ALT (U/L)",
    "fasting_glucose_mg_dl": "Glukosa puasa (mg/dL)",
}
KOLOM_TABEL = {
    "patient_id": "ID pasien",
    "maternal_domicile_district": "Wilayah",
    "facility_name": "Fasilitas ANC",
    "maternal_age_years": "Usia",
    "n_kunjungan": "Jml kunjungan",
    "risiko_tertinggi": "Risiko tertinggi",
    "dirujuk": "Pernah dirujuk",
    "map_maks": "MAP maks",
    "final_diagnosis": "Diagnosis akhir",
}


# =====================================================================
# 2. UTIL
# =====================================================================
def ya(x) -> bool:
    return bool(x) if pd.notna(x) else False

def ya_tidak(x) -> str:
    return "—" if pd.isna(x) else ("Ya" if bool(x) else "Tidak")

def jenis_tipe(s: pd.Series) -> str:
    t = s.dtype
    if pd.api.types.is_datetime64_any_dtype(t): return "Tanggal"
    if pd.api.types.is_bool_dtype(t): return "Boolean (Ya/Tidak)"
    if isinstance(t, pd.CategoricalDtype):
        return "Kategori berurutan" if t.ordered else "Kategori"
    if pd.api.types.is_integer_dtype(t): return "Bilangan bulat"
    if pd.api.types.is_float_dtype(t): return "Desimal"
    return "Teks / ID"

def rentang(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty: return "—"
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return f"{s.min():%Y-%m-%d} s.d. {s.max():%Y-%m-%d}"
    if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
        return f"{s.min():g} s.d. {s.max():g}"
    if isinstance(s.dtype, pd.CategoricalDtype):
        return f"{s.cat.categories.size} kategori"
    return "—"

def _map_ya_tidak(s: pd.Series) -> pd.Series:
    return s.map({"YA": True, "TIDAK": False}).astype("boolean")


# =====================================================================
# 3. VALIDASI
# =====================================================================
def validasi(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["patient_id", "visit_number"])
    g = d.groupby("patient_id", observed=True)
    pe_ok = (df["final_diagnosis"] == "PREECLAMPSIA") == df["preeclampsia_diagnosis_date"].notna()
    umur_dari_dob = (df["visit_date"] - df["date_of_birth"]).dt.days / 365.25
    cek = [
        ("Diagnosis akhir konsisten per pasien",
         int((g["final_diagnosis"].nunique() != 1).sum())),
        ("Nomor kunjungan berurutan 1..n tanpa duplikat",
         int(((g["visit_number"].nunique() != g.size()) |
              (g["visit_number"].max() != g.size())).sum())),
        ("Tanggal kunjungan menaik per pasien",
         int((g["visit_date"].diff().dt.days < 0).sum())),
        ("Usia kehamilan menaik per pasien",
         int((g["gestational_age_weeks"].diff() < 0).sum())),
        ("Sistolik > diastolik",
         int((df["systolic_bp_mmHg"] <= df["diastolic_bp_mmHg"]).sum())),
        ("MAP = (S + 2D)/3 (toleransi 0,6)",
         int(((df["mean_arterial_pressure_mmHg"] -
               (df["systolic_bp_mmHg"] + 2*df["diastolic_bp_mmHg"]) / 3).abs() > 0.6).sum())),
        ("Kasus PE punya tgl diagnosis, non-PE tidak", int((~pe_ok).sum())),
        ("Tanggal kunjungan tidak mendahului HPHT",
         int((df["visit_date"] < df["lmp_date"]).sum())),
        ("Tanggal kunjungan tidak melewati persalinan",
         int((df["visit_date"] > df["delivery_date"]).sum())),
        ("Rujukan tidak ke fasilitas yang sama",
         int(((df["referral_status"] == "DIRUJUK") &
              (df["referral_destination"].astype(str) == df["facility_name"].astype(str))).sum())),
        ("Usia ibu sesuai tanggal lahir (selisih ≤ 1 th)",
         int(((umur_dari_dob - df["maternal_age_years"]).abs() > 1).sum())),
    ]
    out = pd.DataFrame(cek, columns=["Pemeriksaan", "Baris bermasalah"])
    out["Status"] = np.where(out["Baris bermasalah"] == 0, "Lolos", "Perlu dicek")
    return out


# =====================================================================
# 4. LOAD & PERSIAPAN (cached)
# =====================================================================
@st.cache_data(show_spinner="Memuat dan memvalidasi data...")
def siapkan(path: str):
    tipe_awal = pd.read_csv(path, encoding="utf-8-sig", nrows=0).dtypes.astype(str)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False,
                      encoding="utf-8-sig").replace({"": np.nan})
    df = raw.replace(PLACEHOLDER, np.nan)

    for c in ID_COLS + TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")
    for c in INT_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            df[c] = s.astype("int64") if s.notna().all() else s.astype("Int64")
    for c in FLOAT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = _map_ya_tidak(df[c])
    for c, cats in ORDERED.items():
        if c in df.columns:
            df[c] = pd.Categorical(df[c], categories=cats, ordered=True)

    # Kolom turunan
    df["fasilitas"] = (df["facility_name"].astype("string")
                       .str.replace(r"\s*\(Simulasi\)", "", regex=True)
                       .str.replace(" Sintetis", "", regex=False).astype("category"))
    df["fasilitas_tujuan"] = df["referral_destination"].astype("string").astype("category")

    # Status kunjungan (derivasi)
    pe = df["final_diagnosis"] == "PREECLAMPSIA"
    tgl = df["preeclampsia_diagnosis_date"]
    belum = pe & tgl.notna() & (df["visit_date"] < tgl)
    sudah = pe & tgl.notna() & (df["visit_date"] >= tgl)
    df["status_at_visit"] = np.select(
        [sudah, belum], ["PE terdiagnosis", "Belum PE"], default="Non-PE")

    df = df.sort_values(["patient_id", "visit_number"]).reset_index(drop=True)

    # ---- level pasien (1 baris = 1 pasien) ----
    g = df.groupby("patient_id", observed=True)
    first = g.first()
    agg = g.agg(
        n_kunjungan=("anc_visit_id", "count"),
        kunjungan_pertama=("visit_date", "min"),
        kunjungan_terakhir=("visit_date", "max"),
        sistolik_maks=("systolic_bp_mmHg", "max"),
        diastolik_maks=("diastolic_bp_mmHg", "max"),
        map_maks=("mean_arterial_pressure_mmHg", "max"),
        risiko_tertinggi=("pe_screening_risk_category", "max"),
        curiga=("suspected_preeclampsia_flag", lambda s: bool(s.fillna(False).any())),
        dirujuk=("referral_status", lambda s: bool((s == "DIRUJUK").any())),
    )
    kolom_first = [
        "facility_name", "maternal_domicile_district", "maternal_age_years",
        "final_diagnosis", "preeclampsia_diagnosis_date", "preeclampsia_severity",
        "lmp_date", "eclampsia_flag", "hellp_syndrome_flag", "prepregnancy_bmi",
        "interpregnancy_interval_years", "delivery_mode", "delivery_date",
        "birth_weight_g", "apgar_5min", "nicu_admission_flag",
        "preterm_delivery_flag", "low_birth_weight_flag",
        "maternal_icu_admission_flag", "maternal_discharge_status",
    ] + [c for c in FAKTOR if c in df.columns]
    kolom_first = [c for c in kolom_first if c in first.columns]
    pat = agg.join(first[kolom_first]).rename(columns={
        "facility_name": "fasilitas_awal",
        "maternal_domicile_district": "wilayah",
        "maternal_age_years": "usia",
        "final_diagnosis": "diagnosis",
        "preeclampsia_diagnosis_date": "tgl_diagnosis",
        "preeclampsia_severity": "severity",
    }).reset_index()

    for c in ["wilayah", "fasilitas_awal", "diagnosis", "risiko_tertinggi",
              "severity", "delivery_mode", "maternal_discharge_status"]:
        if c in pat.columns:
            pat[c] = pat[c].astype(str)
    for c in [c for c in FAKTOR if c in pat.columns] + [
        "eclampsia_flag", "hellp_syndrome_flag", "nicu_admission_flag",
        "preterm_delivery_flag", "low_birth_weight_flag",
        "maternal_icu_admission_flag"]:
        if c in pat.columns:
            pat[c] = pat[c].fillna(False).astype(bool)

    pat["obesitas"] = pat["prepregnancy_bmi"] > 30
    pat["usia35"] = pat["usia"] >= 35
    pat["jarak10"] = pat["interpregnancy_interval_years"] > 10
    pat["pe"] = pat["diagnosis"] == "PREECLAMPSIA"
    pat["ga_diagnosis"] = (pat["tgl_diagnosis"] - pat["lmp_date"]).dt.days / 7

    # ---- audit tipe data ----
    audit = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe awal": [tipe_awal.get(c, "(turunan)") for c in df.columns],
        "Tipe akhir": [str(df[c].dtype) for c in df.columns],
        "Jenis": [jenis_tipe(df[c]) for c in df.columns],
        "Penanda diganti kosong": [
            int(raw[c].isin(PLACEHOLDER).sum()) if c in raw.columns else 0
            for c in df.columns],
        "Kosong (akhir)": [int(df[c].isna().sum()) for c in df.columns],
        "Nilai unik": [int(df[c].nunique()) for c in df.columns],
        "Rentang / kategori": [rentang(df[c]) for c in df.columns],
        "Contoh": [", ".join(df[c].dropna().astype(str).unique()[:3]) for c in df.columns],
    })
    return df, pat, audit, validasi(df)


# =====================================================================
# 5. KOMPONEN BERSAMA
# =====================================================================
def tabel_pasien(tabel: pd.DataFrame, key: str):
    tabel = tabel.reset_index(drop=True)
    kol = [k for k in KOLOM_TABEL if k in tabel.columns]
    st.caption(f"{len(tabel)} pasien. **Klik satu baris** untuk melihat riwayat lengkapnya di bawah.")
    event = st.dataframe(
        tabel[kol].rename(columns=KOLOM_TABEL),
        hide_index=True, width="stretch", height=300,
        on_select="rerun", selection_mode="single-row", key=key,
    )
    rows = event.selection.rows
    return tabel.loc[rows[0], "patient_id"] if rows and rows[0] < len(tabel) else None


def tabel_proporsi(pat: pd.DataFrame, by: str) -> pd.DataFrame:
    t = pat.groupby(by, observed=True).agg(
        pasien=("patient_id", "count"), kasus_pe=("pe", "sum")).reset_index()
    t["persen_pe"] = t["kasus_pe"] / t["pasien"] * 100
    t["label"] = (t["kasus_pe"].astype(int).astype(str) + "/" +
                  t["pasien"].astype(str) + " (" +
                  t["persen_pe"].round(1).astype(str) + "%)")
    return t


def kurva_kelompok(df: pd.DataFrame, kolom: str, lebar: int = 4) -> pd.DataFrame:
    d = df[["final_diagnosis", "gestational_age_weeks", kolom]].dropna().copy()
    d[kolom] = d[kolom].astype(float)
    d["ga"] = (d["gestational_age_weeks"] // lebar) * lebar + lebar / 2
    return d.groupby(["final_diagnosis", "ga"], observed=True)[kolom].mean().reset_index()


def kv(judul, pasangan):
    st.markdown(f"**{judul}**")
    st.dataframe(pd.DataFrame(pasangan, columns=["Item", "Nilai"]),
                 hide_index=True, width="stretch")


# =====================================================================
# 6. GRAFIK & TABEL PASIEN
# =====================================================================
def grafik_tekanan_darah(v, df, p, f_last, bandingkan, key):
    fig = go.Figure()
    simbol = np.where(v["referral_status"] == "DIRUJUK", "star", "circle")
    ukuran = np.where(v["referral_status"] == "DIRUJUK", 14, 9)
    cd = np.stack([v["visit_number"], v["visit_date"].dt.strftime("%d/%m/%Y"),
                   v["status_at_visit"].astype(str), v["fasilitas"].astype(str)], axis=-1)
    tmpl = ("Kunjungan %{customdata[0]} (%{customdata[1]}) di %{customdata[3]}<br>"
            "Usia hamil %{x:.1f} mgg<br>%{fullData.name}: %{y}<br>"
            "Status: %{customdata[2]}<extra></extra>")
    for kol, nama, warna, dash in [
        ("systolic_bp_mmHg", "Sistolik", "#d62728", "solid"),
        ("diastolic_bp_mmHg", "Diastolik", "#1f77b4", "solid"),
        ("mean_arterial_pressure_mmHg", "MAP", "#2ca02c", "dash"),
    ]:
        fig.add_trace(go.Scatter(
            x=v["gestational_age_weeks"], y=v[kol].astype(float),
            mode="lines+markers", name=nama,
            line=dict(color=warna, dash=dash),
            marker=dict(size=ukuran, symbol=simbol),
            customdata=cd, hovertemplate=tmpl))
    if bandingkan != "Tidak":
        kol = {"Sistolik": "systolic_bp_mmHg", "Diastolik": "diastolic_bp_mmHg",
               "MAP": "mean_arterial_pressure_mmHg"}[bandingkan]
        kg = kurva_kelompok(df, kol)
        for dx, warna, lab in [("NON_PREECLAMPSIA", "#7f8c8d", "Non-PE"),
                               ("PREECLAMPSIA", "#f5a3a3", "PE")]:
            s = kg[kg["final_diagnosis"] == dx]
            fig.add_trace(go.Scatter(
                x=s["ga"], y=s[kol], mode="lines",
                name=f"Rata-rata {lab} ({bandingkan})",
                line=dict(color=warna, dash="dot", width=3), opacity=0.8))
    fig.add_hline(y=140, line_dash="dash", line_color="rgba(214,39,40,0.5)",
                  annotation_text="Sistolik 140", annotation_position="top left")
    fig.add_hline(y=90, line_dash="dash", line_color="rgba(31,119,180,0.5)",
                  annotation_text="Diastolik 90 / MAP 90", annotation_position="bottom left")
    if pd.notna(p["ga_diagnosis"]):
        fig.add_vline(x=float(p["ga_diagnosis"]), line_dash="dot", line_color="#d62728",
                      annotation_text="Diagnosis PE", annotation_position="top right")
    if pd.notna(f_last["gestational_age_at_delivery_weeks"]):
        fig.add_vline(x=float(f_last["gestational_age_at_delivery_weeks"]),
                      line_dash="dot", line_color="#888",
                      annotation_text="Persalinan", annotation_position="top left")
    fig.update_layout(height=430, margin=dict(t=30, b=10), hovermode="closest",
                      xaxis_title="Usia kehamilan (minggu)", yaxis_title="mmHg",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch", key=key)
    st.caption("Bintang = kunjungan yang berujung rujukan. Garis putus-putus = ambang klinis.")


def grafik_berat(v, key):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=v["gestational_age_weeks"], y=v["weight_kg"],
                             mode="lines+markers", name="Berat badan (kg)",
                             line=dict(color="#ff7f0e")), secondary_y=False)
    fig.add_trace(go.Scatter(x=v["gestational_age_weeks"], y=v["prepregnancy_bmi"],
                             mode="markers", name="IMT sebelum hamil (acuan)",
                             marker=dict(color="#9467bd", symbol="diamond",
                                         size=10)), secondary_y=True)
    fig.update_layout(height=360, margin=dict(t=20, b=10),
                      xaxis_title="Usia kehamilan (minggu)",
                      legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(title_text="kg", secondary_y=False)
    fig.update_yaxes(title_text="IMT (kg/m²)", secondary_y=True)
    st.plotly_chart(fig, width="stretch", key=key)


def grafik_lab(v, key):
    panjang = v.melt(id_vars=["visit_number", "gestational_age_weeks"],
                     value_vars=list(LAB), var_name="param", value_name="nilai"
                     ).dropna(subset=["nilai"])
    if panjang.empty:
        st.info("Tidak ada pemeriksaan laboratorium pada pasien ini.")
        return
    panjang["Parameter"] = panjang["param"].map(LAB)
    fig = px.line(panjang, x="gestational_age_weeks", y="nilai",
                  facet_col="Parameter", facet_col_wrap=3, markers=True,
                  labels={"gestational_age_weeks": "Usia hamil (mgg)", "nilai": ""})
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=420, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch", key=key)
    tdk = [LAB[c] for c in LAB if v[c].isna().all()]
    if tdk:
        st.caption("Tidak pernah diperiksa: " + ", ".join(tdk) + ".")


def tabel_kunjungan(v):
    gejala_map = {
        "headache_present": "sakit kepala",
        "visual_disturbance_present": "gangguan penglihatan",
        "epigastric_or_ruq_pain_present": "nyeri ulu hati",
        "dyspnea_present": "sesak",
    }
    gejala = v.apply(
        lambda r: ", ".join(n for c, n in gejala_map.items() if ya(r[c])) or "–", axis=1)
    t = pd.DataFrame({
        "Kunj.": v["visit_number"].astype(int),
        "Tanggal": v["visit_date"].dt.strftime("%d/%m/%Y"),
        "Usia hamil (mgg)": v["gestational_age_weeks"].round(1),
        "Fasilitas": v["fasilitas"].astype(str),
        "TD (mmHg)": (v["systolic_bp_mmHg"].astype(str) + "/" +
                       v["diastolic_bp_mmHg"].astype(str)),
        "MAP": v["mean_arterial_pressure_mmHg"].round(1),
        "Δ sistolik": v["systolic_bp_change_mmHg"],
        "Berat (kg)": v["weight_kg"],
        "Edema": np.where(v["edema_present"].fillna(False).astype(bool),
                           v["edema_location"].astype(str), "–"),
        "Proteinuria": v["proteinuria_dipstick_result"].astype(str),
        "Gejala": gejala,
        "Risiko skrining": v["pe_screening_risk_category"].astype(str),
        "Dugaan PE": np.where(v["suspected_preeclampsia_flag"].fillna(False).astype(bool),
                              "Ya", "Tidak"),
        "Rujukan": np.where(v["referral_status"] == "DIRUJUK",
                             "→ " + v["fasilitas_tujuan"].astype(str), "–"),
        "Status": v["status_at_visit"].astype(str),
    })
    tinggi = ((v["systolic_bp_mmHg"] >= 140) |
              (v["diastolic_bp_mmHg"] >= 90)).reset_index(drop=True)
    idx_td = t.columns.get_loc("TD (mmHg)")
    latar = {"PE terdiagnosis": "background-color: rgba(214,39,40,0.28)",
             "Belum PE": "background-color: rgba(242,166,90,0.22)"}

    def gaya(r):
        css = [latar.get(r["Status"], "")] * len(r)
        if tinggi.iloc[r.name]:
            css[idx_td] += ";color:#ff6b6b;font-weight:700"
        return css

    st.dataframe(t.reset_index(drop=True).style.apply(gaya, axis=1),
                 hide_index=True, width="stretch")
    st.caption("Latar merah = PE terdiagnosis, oranye = pasien akhirnya PE tapi belum "
               "terdiagnosis saat kunjungan. TD tebal merah = sistolik ≥ 140 atau diastolik ≥ 90.")


def tabel_gejala(v):
    def tanda(c):
        return np.where(v[c].fillna(False).astype(bool), "●", "·")
    t = pd.DataFrame({
        "Kunj.": v["visit_number"].astype(int),
        "Tanggal": v["visit_date"].dt.strftime("%d/%m/%Y"),
        "Edema": tanda("edema_present"),
        "Lokasi edema": v["edema_location"].astype(str),
        "Edema menetap": tanda("edema_persistent_flag"),
        "Sakit kepala": tanda("headache_present"),
        "Gangguan penglihatan": tanda("visual_disturbance_present"),
        "Nyeri ulu hati": tanda("epigastric_or_ruq_pain_present"),
        "Sesak": tanda("dyspnea_present"),
        "Proteinuria": v["proteinuria_dipstick_result"].astype(str),
    })
    st.dataframe(t, hide_index=True, width="stretch")
    st.caption("● = ada, · = tidak ada.")


def linimasa(v, f):
    ev = []
    for _, r in v.iterrows():
        ev.append((
            r["visit_date"],
            f"Kunjungan ANC ke-{int(r['visit_number'])}",
            f"{r['fasilitas']} · {r['gestational_age_weeks']:.1f} mgg · "
            f"TD {r['systolic_bp_mmHg']}/{r['diastolic_bp_mmHg']} · "
            f"{r['status_at_visit']}"
        ))
        if r.get("referral_status") == "DIRUJUK":
            tgl = r["referral_date"] if pd.notna(r.get("referral_date")) else r["visit_date"]
            alasan = r.get("referral_reason_code")
            alasan_txt = f" · alasan: {alasan}" if pd.notna(alasan) else ""
            ev.append((
                tgl,
                "Rujukan",
                f"ke {r.get('fasilitas_tujuan', '—')}{alasan_txt}"
            ))
    if pd.notna(f.get("preeclampsia_diagnosis_date")):
        ev.append((
            f["preeclampsia_diagnosis_date"],
            "Diagnosis preeklamsia",
            f"severity: {f.get('preeclampsia_severity', '—')}"
        ))
    if pd.notna(f.get("delivery_date")):
        ev.append((
            f["delivery_date"],
            "Persalinan",
            f"{f.get('delivery_mode', '—')} · usia kehamilan "
            f"{f.get('gestational_age_at_delivery_weeks', float('nan')):.1f} mgg"
            if pd.notna(f.get("gestational_age_at_delivery_weeks"))
            else f"{f.get('delivery_mode', '—')}"
        ))
    # Kolom postpartum mungkin tidak ada di dataset — gunakan .get()
    if pd.notna(f.get("postpartum_visit_date")):
        td = (f.get("postpartum_systolic_bp"), f.get("postpartum_diastolic_bp"))
        td_txt = (f"TD {td[0]}/{td[1]}"
                  if pd.notna(td[0]) and pd.notna(td[1]) else "")
        ev.append((
            f["postpartum_visit_date"],
            "Kontrol pascasalin",
            td_txt or "—"
        ))
    t = pd.DataFrame(ev, columns=["Tanggal", "Peristiwa", "Detail"]
                     ).sort_values("Tanggal", kind="stable")
    t["Tanggal"] = pd.to_datetime(t["Tanggal"]).dt.strftime("%d/%m/%Y")
    st.dataframe(t, hide_index=True, width="stretch")


def render_pasien(pid, df, pat, key):
    v = df[df["patient_id"] == pid].sort_values("visit_number").reset_index(drop=True)
    p = pat[pat["patient_id"] == pid].iloc[0]
    f0, f = v.iloc[0], v.iloc[-1]
    is_pe = bool(p["pe"])

    st.divider()
    warna = "red" if is_pe else "blue"
    st.subheader(f"Riwayat pasien {p['patient_id']}")
    st.markdown(f"Diagnosis akhir: :{warna}[**{p['diagnosis']}**] · "
                f"severity: {p['severity']}")

    c = st.columns(4)
    c[0].metric("Usia (kunjungan pertama)", f"{int(p['usia'])} th")
    c[1].metric("Wilayah (domisili)", p["wilayah"])
    c[2].metric("Kunjungan ANC", f"{int(p['n_kunjungan'])}×",
                f"{p['kunjungan_pertama']:%d/%m/%Y} – {p['kunjungan_terakhir']:%d/%m/%Y}",
                delta_color="off")
    c[3].metric("Risiko skrining tertinggi", p["risiko_tertinggi"])
    c = st.columns(4)
    rute = " → ".join(dict.fromkeys(v["fasilitas"].astype(str)))
    c[0].metric("Fasilitas ANC", rute)
    c[1].metric("Pernah dirujuk", "Ya" if p["dirujuk"] else "Tidak")
    c[2].metric("Tanggal diagnosis PE",
                f"{p['tgl_diagnosis']:%d/%m/%Y}" if pd.notna(p["tgl_diagnosis"]) else "—",
                f"usia hamil ≈ {p['ga_diagnosis']:.0f} mgg" if pd.notna(p["ga_diagnosis"]) else None,
                delta_color="off")
    c[3].metric("Persalinan",
                f"{p['delivery_date']:%d/%m/%Y}" if pd.notna(p["delivery_date"]) else "—",
                str(p["delivery_mode"]), delta_color="off")

    with st.expander("Profil & faktor risiko", expanded=True):
        a, b = st.columns(2)
        kiri = [(lab, ya_tidak(p[col])) for col, lab in FAKTOR.items() if col in p]
        with a:
            kv("Faktor risiko skrining (Buku KIA)", kiri)
        with b:
            kv("Riwayat obstetri & karakteristik", [
                ("Gravida / paritas / abortus",
                 f"{int(f0['gravida'])} / {int(f0['parity'])} / {int(f0['abortion_count'])}"),
                ("Jarak kehamilan sebelumnya (th)",
                 f"{f0['interpregnancy_interval_years']:.1f}"
                 if pd.notna(f0["interpregnancy_interval_years"]) else "—"),
                ("IMT sebelum hamil", f"{f0['prepregnancy_bmi']:.1f}"),
                ("Tinggi badan (cm)", f"{f0['height_cm']:.1f}"),
                ("HPHT / HPL",
                 f"{f0['lmp_date']:%d/%m/%Y} / {f0['estimated_due_date']:%d/%m/%Y}"),
            ])

    st.markdown("### Perkembangan selama kehamilan")
    t1, t2, t3, t4 = st.tabs(["Tekanan darah", "Berat badan & IMT",
                              "Laboratorium", "Gejala & proteinuria"])
    with t1:
        bandingkan = st.selectbox(
            "Bandingkan dengan rata-rata kelompok (per 4 minggu)",
            ["Tidak", "Sistolik", "Diastolik", "MAP"], key=f"{key}_cmp")
        grafik_tekanan_darah(v, df, p, f, bandingkan, f"{key}_td")
    with t2:
        grafik_berat(v, f"{key}_bb")
    with t3:
        grafik_lab(v, f"{key}_lab")
    with t4:
        tabel_gejala(v)

    st.markdown("### Tabel riwayat kunjungan")
    tabel_kunjungan(v)

    st.markdown("### Linimasa kejadian")
    linimasa(v, f)

    with st.expander("Luaran persalinan, pascasalin & bayi", expanded=False):
        a, b, c3 = st.columns(3)
        with a:
            kv("Ibu saat persalinan", [
                ("Cara persalinan", str(f.get("delivery_mode", "—"))),
                ("Eklampsia", ya_tidak(f.get("eclampsia_flag"))),
                ("HELLP", ya_tidak(f.get("hellp_syndrome_flag"))),
                ("Perdarahan pascapersalinan", ya_tidak(f.get("placental_abruption_flag"))),
                ("Perawatan intensif", ya_tidak(f.get("maternal_icu_admission_flag"))),
                ("Status saat pulang", str(f.get("maternal_discharge_status", "—"))),
            ])
        with b:
            kv("Pascasalin", [
                ("Tanggal kontrol",
                 f"{f['postpartum_visit_date']:%d/%m/%Y}"
                 if pd.notna(f.get("postpartum_visit_date")) else "—"),
                ("TD pascasalin",
                 f"{f.get('postpartum_systolic_bp', '—')}/{f.get('postpartum_diastolic_bp', '—')}"
                 if pd.notna(f.get("postpartum_systolic_bp"))
                 and pd.notna(f.get("postpartum_diastolic_bp")) else "—"),
            ])
        with c3:
            kv("Bayi", [
                ("Jenis kelamin", str(f.get("baby_sex", "—"))),
                ("Berat / panjang lahir",
                 f"{int(f['birth_weight_g']) if pd.notna(f.get('birth_weight_g')) else '—'} g / "
                 f"{f['birth_length_cm']:.1f} cm"
                 if pd.notna(f.get('birth_length_cm')) else "—"),
                ("Apgar 1' / 5'",
                 f"{f.get('apgar_1min', '—')} / {f.get('apgar_5min', '—')}"),
                ("Status lahir", str(f.get("birth_status", "—"))),
                ("BBLR / prematur",
                 f"{ya_tidak(f.get('low_birth_weight_flag'))} / "
                 f"{ya_tidak(f.get('preterm_delivery_flag'))}"),
                ("Masuk NICU", ya_tidak(f.get("nicu_admission_flag"))),
            ])
            
def halaman_ringkasan(df, pat):
    st.header("Ringkasan")
    n, n_pe = len(pat), int(pat["pe"].sum())
    c = st.columns(5)
    c[0].metric("Pasien", f"{n:,}")
    c[1].metric("Kunjungan ANC", f"{len(df):,}",
                f"{len(df)/n:.1f} per pasien", delta_color="off")
    c[2].metric("Pasien preeklamsia", f"{n_pe}")
    c[3].metric("Proporsi PE", f"{n_pe/n*100:.1f}%")
    c[4].metric("Pasien pernah dirujuk", f"{int(pat['dirujuk'].sum())}")
    st.caption("Semua angka pasien dihitung per pasien (bukan per baris kunjungan).")

    # Deteksi dini & tahapan alur
    a, b = st.columns(2)
    with a:
        pe = pat[pat["pe"]]
        st.markdown("**Deteksi dini di layanan ANC**")
        d = st.columns(3)
        d[0].metric("Kasus PE", f"{len(pe)}")
        d[1].metric("Pernah dicurigai PE", f"{int(pe['curiga'].sum())}")
        d[2].metric("Pernah dirujuk", f"{int(pe['dirujuk'].sum())}")
        tahap = pd.DataFrame({
            "Tahap": ["Semua pasien",
                      "Risiko skrining sedang/tinggi",
                      "Pernah dicurigai PE",
                      "Pernah dirujuk",
                      "PE terdiagnosis"],
            "Pasien": [n,
                       int(pat["risiko_tertinggi"].isin(["SEDANG", "TINGGI"]).sum()),
                       int(pat["curiga"].sum()),
                       int(pat["dirujuk"].sum()),
                       n_pe],
        })
        fig = px.bar(tahap, x="Pasien", y="Tahap", orientation="h", text="Pasien")
        fig.update_layout(height=300, margin=dict(t=10, b=10),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch", key="rg_tahap")
    with b:
        dist = pat["diagnosis"].value_counts().reset_index()
        dist.columns = ["Diagnosis", "Pasien"]
        dist["Label"] = dist["Diagnosis"].map(
            {"PREECLAMPSIA": "Preeklamsia", "NON_PREECLAMPSIA": "Non-Preeklamsia"})
        fig = px.pie(dist, names="Label", values="Pasien", hole=0.5,
                     color="Label",
                     color_discrete_map={"Preeklamsia": "#E45756",
                                          "Non-Preeklamsia": "#4C78A8"},
                     title="Diagnosis akhir (per pasien)")
        fig.update_layout(height=370, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch", key="rg_pie")

    # Proporsi per wilayah & fasilitas
    a, b = st.columns(2)
    for kol, judul, ui, kk in [
        ("wilayah", "Proporsi PE per wilayah (domisili)", a, "rg_wil"),
        ("fasilitas_awal", "Proporsi PE per fasilitas ANC", b, "rg_fas"),
    ]:
        t = tabel_proporsi(pat, kol).sort_values("persen_pe", ascending=False)
        fig = px.bar(t, x=kol, y="persen_pe", text="label", title=judul,
                     labels={kol: "", "persen_pe": "% pasien PE"})
        fig.update_layout(height=360, margin=dict(t=40, b=10))
        fig.update_traces(textposition="outside", cliponaxis=False)
        ui.plotly_chart(fig, width="stretch", key=kk)
    st.caption("Label = kasus PE / pasien (persen).")

    # Distribusi severity & komplikasi pada kasus PE
    st.subheader("Karakteristik kasus PE")
    pe = pat[pat["pe"]]
    a, b, c = st.columns(3)
    with a:
        sev = pe["severity"].value_counts().reset_index()
        sev.columns = ["Severity", "Pasien"]
        st.plotly_chart(
            px.bar(sev, x="Severity", y="Pasien", text="Pasien",
                   color="Severity",
                   color_discrete_map={"TANPA_SEVERE_FEATURES": "#f2a65a",
                                        "DENGAN_SEVERE_FEATURES": "#d62728"},
                   title="Severity preeklamsia"),
            width="stretch", key="rg_sev")
    with b:
        komp = pd.DataFrame({
            "Komplikasi": ["Eklampsia", "HELLP", "ICU maternal"],
            "Kasus": [int(pe["eclampsia_flag"].sum()),
                      int(pe["hellp_syndrome_flag"].sum()),
                      int(pe["maternal_icu_admission_flag"].sum())],
        })
        st.plotly_chart(px.bar(komp, x="Komplikasi", y="Kasus", text="Kasus",
                               color="Komplikasi"),
                        width="stretch", key="rg_komp")
    with c:
        luaran = pd.DataFrame({
            "Luaran": ["Prematur", "BBLR", "NICU"],
            "Kasus": [int(pe["preterm_delivery_flag"].sum()),
                      int(pe["low_birth_weight_flag"].sum()),
                      int(pe["nicu_admission_flag"].sum())],
        })
        st.plotly_chart(px.bar(luaran, x="Luaran", y="Kasus", text="Kasus",
                               color="Luaran"),
                        width="stretch", key="rg_luaran")


def halaman_pasien(df, pat):
    st.header("Riwayat per pasien")
    c = st.columns(5)
    wil = c[0].multiselect("Wilayah (domisili)", sorted(pat["wilayah"].unique()))
    fas = c[1].multiselect("Pernah ANC di fasilitas",
                            sorted(df["fasilitas"].astype(str).unique()))
    dx = c[2].selectbox("Diagnosis akhir",
                         ["Semua", "Preeklamsia", "Non-Preeklamsia"])
    rs = c[3].selectbox("Risiko skrining tertinggi",
                         ["Semua", "RENDAH", "SEDANG", "TINGGI"])
    cari = c[4].text_input("Cari ID pasien", placeholder="mis. PTN001")
    hanya = st.checkbox("Hanya pasien yang pernah dirujuk")

    t = pat
    if wil:
        t = t[t["wilayah"].isin(wil)]
    if fas:
        ids = df.loc[df["fasilitas"].astype(str).isin(fas), "patient_id"].unique()
        t = t[t["patient_id"].isin(ids)]
    if dx != "Semua":
        val = "PREECLAMPSIA" if dx == "Preeklamsia" else "NON_PREECLAMPSIA"
        t = t[t["diagnosis"] == val]
    if rs != "Semua":
        t = t[t["risiko_tertinggi"] == rs]
    if cari:
        t = t[t["patient_id"].str.contains(cari.strip(), case=False)]
    if hanya:
        t = t[t["dirujuk"]]
    if t.empty:
        st.warning("Tidak ada pasien yang cocok dengan filter.")
        return
    pid = tabel_pasien(t, "tbl_pasien")
    if pid is None:
        st.info("Pilih satu pasien untuk melihat riwayatnya.")
    else:
        render_pasien(pid, df, pat, "psn")
        
def halaman_wilayah(df, pat):
    st.header("Wilayah & fasilitas")
    mode = st.radio("Tingkat analisis",
                    ["Per wilayah (kecamatan domisili)",
                     "Per fasilitas (Puskesmas / Rumah Sakit)"], horizontal=True)
    per_wilayah = mode.startswith("Per wilayah")
    fas_all = df["fasilitas"].astype(str)

    if per_wilayah:
        opsi = sorted(pat["wilayah"].unique())
        pilih = st.selectbox("Pilih wilayah", opsi)
        ids = pat.loc[pat["wilayah"] == pilih, "patient_id"]
        banding = tabel_proporsi(pat, "wilayah").rename(columns={"wilayah": "nama"})
    else:
        opsi = sorted(fas_all.unique())
        pilih = st.selectbox("Pilih fasilitas", opsi)
        ids = pd.Series(df.loc[fas_all == pilih, "patient_id"].unique())
        baris = []
        for nama in opsi:
            s = pat[pat["patient_id"].isin(
                df.loc[fas_all == nama, "patient_id"])]
            baris.append({"nama": nama, "pasien": len(s),
                          "kasus_pe": int(s["pe"].sum())})
        banding = pd.DataFrame(baris)
        banding["persen_pe"] = banding["kasus_pe"] / banding["pasien"] * 100
        banding["label"] = (banding["kasus_pe"].astype(str) + "/" +
                            banding["pasien"].astype(str) + " (" +
                            banding["persen_pe"].round(1).astype(str) + "%)")

    sp = pat[pat["patient_id"].isin(ids)]
    sv = df[df["patient_id"].isin(ids)]
    if sp.empty:
        st.warning("Tidak ada data.")
        return

    c = st.columns(6)
    c[0].metric("Pasien", len(sp))
    c[1].metric("Kunjungan ANC", len(sv))
    c[2].metric("Kasus PE", int(sp["pe"].sum()))
    c[3].metric("Proporsi PE", f"{sp['pe'].mean()*100:.1f}%")
    c[4].metric("Pasien pernah dirujuk", int(sp["dirujuk"].sum()))
    c[5].metric("Rata-rata sistolik",
                f"{sv['systolic_bp_mmHg'].astype(float).mean():.0f} mmHg")

    if not per_wilayah:
        st.markdown(f"**Peran {pilih} dalam alur layanan**")
        tuj = df["fasilitas_tujuan"].astype(str)
        masuk = df.loc[(tuj == pilih) & (fas_all != pilih), "patient_id"].nunique()
        keluar = df.loc[(fas_all == pilih) &
                         (df["referral_status"] == "DIRUJUK") &
                         (tuj != pilih), "patient_id"].nunique()
        lahir = int((pat["fasilitas_awal"] == pilih).sum())
        d = st.columns(4)
        d[0].metric("Pasien dirujuk KELUAR", keluar)
        d[1].metric("Pasien dirujuk MASUK", masuk)
        d[2].metric("Pasien ANC di sini", lahir)
        d[3].metric("Kunjungan ANC di sini", int((fas_all == pilih).sum()))

    a, b = st.columns(2)
    with a:
        banding["Dipilih"] = banding["nama"] == pilih
        fig = px.bar(banding.sort_values("persen_pe", ascending=False),
                     x="nama", y="persen_pe", text="label", color="Dipilih",
                     color_discrete_map={True: "#E45756", False: "#9aa5b1"},
                     title="Proporsi PE: " +
                           ("semua wilayah" if per_wilayah else "semua fasilitas"),
                     labels={"nama": "", "persen_pe": "% pasien PE"})
        fig.update_layout(height=370, margin=dict(t=40, b=10), showlegend=False)
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, width="stretch", key="wf_banding")
    with b:
        if per_wilayah:
            k = sv["fasilitas"].astype(str).value_counts().reset_index()
            k.columns = ["Fasilitas", "Kunjungan"]
            fig = px.bar(k, x="Fasilitas", y="Kunjungan",
                         title=f"Ke mana pasien {pilih} berobat (ANC)")
        else:
            th = sv[fas_all.loc[sv.index] == pilih].copy()
            th["Tahun"] = th["visit_date"].dt.year
            k = th.groupby("Tahun").size().reset_index(name="Kunjungan")
            fig = px.bar(k, x="Tahun", y="Kunjungan",
                         title=f"Kunjungan ANC per tahun di {pilih}")
        fig.update_layout(height=370, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch", key="wf_kanan")

    st.markdown(f"#### Pasien — {pilih}")
    pid = tabel_pasien(sp, f"tbl_wf_{mode[:12]}_{pilih}")
    if pid is None:
        st.info("Klik satu pasien untuk membuka riwayat lengkapnya.")
    else:
        render_pasien(pid, df, pat, "wf")


def halaman_historis(df, pat):
    st.header("Data historis preeklamsia")

    # 1. Tren kasus
    st.subheader("1. Tren kasus dari waktu ke waktu")
    gran = st.radio("Periode", ["Tahun", "Kuartal", "Bulan"], horizontal=True)
    frek = {"Tahun": "Y", "Kuartal": "Q", "Bulan": "M"}[gran]
    pe = pat[pat["pe"]].dropna(subset=["tgl_diagnosis"])
    per = pe["tgl_diagnosis"].dt.to_period(frek)
    rentang_p = pd.period_range(per.min(), per.max(), freq=frek)
    cnt = per.value_counts().reindex(rentang_p, fill_value=0).sort_index()
    tren = pd.DataFrame({"Periode": cnt.index.astype(str), "Kasus baru": cnt.values})
    tren["Kumulatif"] = tren["Kasus baru"].cumsum()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=tren["Periode"], y=tren["Kasus baru"],
                         name="Kasus PE baru", marker_color="#E45756"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=tren["Periode"], y=tren["Kumulatif"],
                             name="Kumulatif", mode="lines+markers",
                             line=dict(color="#555")), secondary_y=True)
    fig.update_layout(height=340, margin=dict(t=20, b=10),
                      legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="Kasus baru", secondary_y=False)
    fig.update_yaxes(title_text="Kumulatif", secondary_y=True)
    st.plotly_chart(fig, width="stretch", key="hs_tren")

    pat2 = pat.assign(tahun=pat["kunjungan_pertama"].dt.year.astype(str))
    tp = tabel_proporsi(pat2, "tahun")
    fig = px.bar(tp, x="tahun", y="persen_pe", text="label",
                 title="Proporsi PE menurut tahun kunjungan ANC pertama",
                 labels={"tahun": "", "persen_pe": "% pasien PE"})
    fig.update_layout(height=330, margin=dict(t=40, b=10))
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, width="stretch", key="hs_tahun")

    # 2. Faktor risiko
    st.subheader("2. Riwayat PE sebelumnya & faktor risiko")
    baris = []
    for col, lab in FAKTOR.items():
        if col not in pat.columns:
            continue
        ada, tdk = pat[pat[col]], pat[~pat[col]]
        r_ada = ada["pe"].mean() * 100 if len(ada) else np.nan
        r_tdk = tdk["pe"].mean() * 100 if len(tdk) else np.nan
        baris.append({
            "Faktor": lab, "Pasien dengan faktor": len(ada),
            "PE (dengan)": int(ada["pe"].sum()), "% PE (dengan)": r_ada,
            "Pasien tanpa faktor": len(tdk), "PE (tanpa)": int(tdk["pe"].sum()),
            "% PE (tanpa)": r_tdk,
            "Rasio risiko": (r_ada / r_tdk) if (r_tdk and not np.isnan(r_ada)) else np.nan,
        })
    ft = pd.DataFrame(baris)
    lg = ft.melt(id_vars="Faktor",
                 value_vars=["% PE (dengan)", "% PE (tanpa)"],
                 var_name="Kelompok", value_name="% PE")
    lg["Kelompok"] = lg["Kelompok"].map({"% PE (dengan)": "Dengan faktor",
                                          "% PE (tanpa)": "Tanpa faktor"})
    fig = px.bar(lg, x="Faktor", y="% PE", color="Kelompok", barmode="group",
                 color_discrete_map={"Dengan faktor": "#E45756",
                                      "Tanpa faktor": "#9aa5b1"})
    fig.update_layout(height=380, margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch", key="hs_faktor")
    st.dataframe(ft.round(2), hide_index=True, width="stretch")
    st.caption("Rasio risiko = gambaran awal, bukan uji statistik. "
               "Rasio kosong = tidak ada kasus/pasien pada salah satu kelompok.")

    # 3. Lintasan tekanan darah
    st.subheader("3. Lintasan tekanan darah: pasien PE vs non-PE")
    pilih = st.radio("Parameter", ["Sistolik", "Diastolik", "MAP"],
                     horizontal=True, key="hs_param")
    kol = {"Sistolik": "systolic_bp_mmHg",
           "Diastolik": "diastolic_bp_mmHg",
           "MAP": "mean_arterial_pressure_mmHg"}[pilih]
    d = df[["patient_id", "final_diagnosis", "gestational_age_weeks", kol]].copy()
    d[kol] = d[kol].astype(float)
    d["ga"] = (d["gestational_age_weeks"] // 4) * 4 + 2
    non = d[d["final_diagnosis"] == "NON_PREECLAMPSIA"].groupby("ga")[kol].agg(
        rata="mean", q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75)).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(non["ga"]) + list(non["ga"][::-1]),
        y=list(non["q3"]) + list(non["q1"][::-1]),
        fill="toself", fillcolor="rgba(76,120,168,0.2)",
        line=dict(width=0), name="Non-PE: rentang 25–75%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=non["ga"], y=non["rata"], mode="lines",
                             name="Rata-rata non-PE",
                             line=dict(color="#4C78A8", width=3)))
    for kode, g in d[d["final_diagnosis"] == "PREECLAMPSIA"].groupby(
            "patient_id", observed=True):
        fig.add_trace(go.Scatter(
            x=g["gestational_age_weeks"], y=g[kol], mode="lines+markers",
            showlegend=False, line=dict(color="rgba(228,87,86,0.55)", width=1.5),
            marker=dict(size=5), name=str(kode),
            hovertemplate=f"{kode}<br>Usia hamil %{{x:.1f}} mgg<br>"
                          f"{pilih} %{{y}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             name="Pasien PE (1 garis = 1 pasien)",
                             line=dict(color="#E45756")))
    fig.update_layout(height=420, margin=dict(t=10, b=10),
                      xaxis_title="Usia kehamilan (minggu)",
                      yaxis_title=f"{pilih} (mmHg)",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch", key="hs_lintasan")

    # 4. Status per kunjungan
    st.subheader("4. Status pasien menurut nomor kunjungan")
    s = df.groupby(["visit_number", "status_at_visit"], observed=True
                   ).size().reset_index(name="Kunjungan")
    s["status_at_visit"] = s["status_at_visit"].astype(str)
    fig = px.bar(s, x="visit_number", y="Kunjungan", color="status_at_visit",
                 color_discrete_map=WARNA_STATUS,
                 labels={"visit_number": "Kunjungan ke-",
                         "status_at_visit": "Status"})
    fig.update_layout(barnorm="percent", height=330, margin=dict(t=10, b=10),
                      yaxis_title="% kunjungan",
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch", key="hs_status")
    st.caption("'Belum PE' = pasien yang kemudian terdiagnosis PE tapi belum "
               "terdiagnosis saat kunjungan tersebut.")

    # 5. Sebaran usia kehamilan saat diagnosis
    st.subheader("5. Sebaran usia kehamilan saat diagnosis PE")
    pe_ga = pe.dropna(subset=["ga_diagnosis"])
    if len(pe_ga):
        fig = px.histogram(pe_ga, x="ga_diagnosis", nbins=20,
                           labels={"ga_diagnosis": "Usia kehamilan saat diagnosis (minggu)"},
                           title="Distribusi usia kehamilan saat diagnosis PE")
        fig.add_vline(x=34, line_dash="dash", line_color="#555",
                      annotation_text="34 mgg", annotation_position="top")
        fig.add_vline(x=37, line_dash="dash", line_color="#555",
                      annotation_text="37 mgg (aterm)", annotation_position="top right")
        fig.update_layout(height=330, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch", key="hs_ga")
        a, b, c = st.columns(3)
        a.metric("Median usia diagnosis",
                 f"{pe_ga['ga_diagnosis'].median():.1f} mgg")
        b.metric("< 34 mgg (preterm berat)",
                 f"{int((pe_ga['ga_diagnosis'] < 34).sum())}")
        c.metric("< 37 mgg", f"{int((pe_ga['ga_diagnosis'] < 37).sum())}")

    # 6. Daftar kasus PE
    st.subheader("6. Daftar kasus PE (klik untuk membuka riwayat)")
    kasus = pat[pat["pe"]].sort_values("tgl_diagnosis")
    pid = tabel_pasien(kasus, "tbl_hist_pe")
    if pid is None:
        st.info("Klik satu kasus untuk melihat riwayat lengkap pasien.")
    else:
        render_pasien(pid, df, pat, "hs")


def halaman_kualitas(df, audit, cek):
    st.header("Kualitas data & tipe data")
    st.caption("Semua kolom dibaca sebagai teks dulu, lalu diubah ke tipe yang tepat. "
               "Penanda TIDAK_DIPERIKSA / BELUM_DINILAI / TIDAK_ADA / TIDAK_BERLAKU / "
               "NOT_APPLICABLE / TIDAK_DIRUJUK diganti menjadi nilai kosong.")
    jum = audit["Jenis"].value_counts().reset_index()
    jum.columns = ["Jenis", "Jumlah kolom"]
    a, b = st.columns([1, 2])
    a.dataframe(jum, hide_index=True, width="stretch")
    with b:
        st.markdown(
            f"**{len(df):,} baris × {df.shape[1]} kolom** · "
            f"{df['patient_id'].nunique()} pasien · "
            f"{audit['Penanda diganti kosong'].sum():,} penanda diganti kosong")
        st.markdown("**Pemeriksaan konsistensi**")
        st.dataframe(cek, hide_index=True, width="stretch")
    st.markdown("**Daftar tipe data per kolom**")
    jenis = st.multiselect("Saring jenis", sorted(audit["Jenis"].unique()))
    t = audit if not jenis else audit[audit["Jenis"].isin(jenis)]
    st.dataframe(t, hide_index=True, width="stretch", height=480)
    st.download_button("Unduh audit tipe data (CSV)",
                       audit.to_csv(index=False).encode("utf-8"),
                       file_name="audit_tipe_data.csv", mime="text/csv")


# =====================================================================
# 8. APLIKASI
# =====================================================================
def main():
    st.set_page_config(page_title="Dashboard Riwayat Preeklamsia",
                       page_icon="🩺", layout="wide")
    df, pat, audit, cek = siapkan(str(CSV_PATH))

    st.sidebar.title("🩺 Dashboard PE")
    halaman = st.sidebar.radio("Halaman", [
        "Ringkasan", "Riwayat per pasien", "Wilayah & fasilitas",
        "Data historis PE", "Kualitas data & tipe data",
    ])
    st.sidebar.caption(
        "Data sintetis ANC — Kabupaten Toba. Untuk pembelajaran, bukan data pasien nyata.")
    st.title("Dashboard Riwayat Preeklamsia")

    if halaman == "Ringkasan":
        halaman_ringkasan(df, pat)
    elif halaman == "Riwayat per pasien":
        halaman_pasien(df, pat)
    elif halaman == "Wilayah & fasilitas":
        halaman_wilayah(df, pat)
    elif halaman == "Data historis PE":
        halaman_historis(df, pat)
    else:
        halaman_kualitas(df, audit, cek)


if __name__ == "__main__":
    main()