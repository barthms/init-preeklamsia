"""Dashboard Riwayat & Risiko Preeklamsia (data sintetis ANC, Kabupaten Toba).

Jalankan dari root proyek:  streamlit run app/app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =====================================================================
# 1. KONFIGURASI & SKEMA TIPE DATA
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "dataset" / "dummy" / "dataset_sintetis_preeklamsia_1000_bersih_tanpa_null.csv"

PLACEHOLDER = ["TIDAK_DIPERIKSA", "BELUM_DINILAI", "TIDAK_ADA", "TIDAK_BERLAKU"]

ID_COLS = [
    "record_id", "facility_id", "patient_id", "patient_code", "pregnancy_id", "visit_id",
    "history_id", "screening_id", "referral_id", "destination_facility_id", "outcome_id",
    "delivery_id", "delivery_facility_id", "postpartum_visit_id", "baby_id",
]
TEXT_COLS = [
    "data_type", "screening_notes", "management_notes", "counseling_topic",
    "referral_reason", "baby_record_scope",
]
DATE_COLS = [
    "date_of_birth", "lmp_date", "estimated_due_date", "previous_delivery_date", "visit_date",
    "screening_date", "referral_date", "diagnosis_date", "delivery_date", "postpartum_visit_date",
]
INT_COLS = [
    "maternal_age", "gravida", "parity", "abortion_count", "visit_number", "anc_visit_count",
    "iron_tablet_count", "systolic_bp", "diastolic_bp", "systolic_change", "diastolic_change",
    "fetal_heart_rate_bpm", "postpartum_systolic_bp", "postpartum_diastolic_bp", "baby_count",
    "birth_weight_g", "apgar_1min", "apgar_5min", "birth_order",
]
FLOAT_COLS = [
    "height_cm", "prepregnancy_weight_kg", "prepregnancy_bmi", "interpregnancy_interval_years",
    "gestational_age_weeks", "weight_kg", "current_bmi", "weight_change_kg", "lila_cm", "map_mmhg",
    "hemoglobin_g_dl", "blood_glucose_mg_dl", "platelet_count", "creatinine_mg_dl", "ast_u_l",
    "alt_u_l", "tfu_cm", "gestational_age_delivery", "birth_length_cm",
]
ORDERED = {
    "trimester": ["I", "II", "III"],
    "screening_risk_category": ["Rendah", "Sedang", "Tinggi"],
    "socioeconomic_status": ["Rendah", "Sedang", "Tinggi"],
    "education_level": ["SD", "SMP", "SMA/SMK", "Diploma", "Sarjana"],
    "proteinuria_result": ["Tidak diperiksa", "Negatif", "Trace", "+1", "+2", "+3", "+4"],
    "status_at_visit": ["Non-PE", "Belum PE", "PE terdiagnosis"],
}

WARNA_DX = {"Non-preeklamsia": "#4C78A8", "Preeklamsia": "#E45756"}
WARNA_STATUS = {"Non-PE": "#9ecae9", "Belum PE": "#f2a65a", "PE terdiagnosis": "#d62728"}

# Faktor risiko (mengikuti skrining Buku KIA) -> label tampilan
FAKTOR = {
    "previous_preeclampsia": "Riwayat PE sebelumnya",
    "family_history_preeclampsia": "Riwayat PE pada ibu/saudara",
    "chronic_hypertension": "Hipertensi kronik",
    "preexisting_diabetes": "Diabetes sebelum hamil",
    "gestational_diabetes": "Diabetes gestasional",
    "kidney_disease": "Penyakit ginjal",
    "sle": "Autoimun / SLE",
    "nulliparous_status": "Nulipara (hamil pertama)",
    "kembar": "Kehamilan kembar",
    "ivf": "Teknologi reproduksi berbantu",
    "usia35": "Usia >= 35 tahun",
    "obesitas": "Obesitas (IMT > 30)",
    "jarak10": "Jarak kehamilan > 10 tahun",
}
LAB = {
    "hemoglobin_g_dl": "Hemoglobin (g/dL)",
    "platelet_count": "Trombosit (/uL)",
    "creatinine_mg_dl": "Kreatinin (mg/dL)",
    "ast_u_l": "AST (U/L)",
    "alt_u_l": "ALT (U/L)",
    "blood_glucose_mg_dl": "Glukosa (mg/dL)",
}


def ya(x) -> bool:
    return bool(x) if pd.notna(x) else False


def ya_tidak(x) -> str:
    return "—" if pd.isna(x) else ("Ya" if bool(x) else "Tidak")


# =====================================================================
# 2. LOAD, BERSIHKAN, VALIDASI (sekali, lalu di-cache)
# =====================================================================
def jenis_tipe(s: pd.Series) -> str:
    t = s.dtype
    if pd.api.types.is_datetime64_any_dtype(t):
        return "Tanggal"
    if pd.api.types.is_bool_dtype(t):
        return "Boolean (Ya/Tidak)"
    if isinstance(t, pd.CategoricalDtype):
        return "Kategori berurutan" if t.ordered else "Kategori"
    if pd.api.types.is_integer_dtype(t):
        return "Bilangan bulat"
    if pd.api.types.is_float_dtype(t):
        return "Desimal"
    return "Teks / ID"


def rentang(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "—"
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return f"{s.min():%Y-%m-%d} s.d. {s.max():%Y-%m-%d}"
    if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
        return f"{s.min():g} s.d. {s.max():g}"
    if isinstance(s.dtype, pd.CategoricalDtype):
        return f"{s.cat.categories.size} kategori"
    return "—"


def validasi(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["patient_id", "visit_number"])
    g = d.groupby("patient_id", observed=True)
    pe_ok = (df["final_diagnosis"] == "Preeklamsia") == df["diagnosis_date"].notna()
    umur_dari_dob = (df["visit_date"] - df["date_of_birth"]).dt.days / 365.25
    cek = [
        ("1 pasien = 1 kode pasien", int((g["patient_code"].nunique() != 1).sum()
                                          + (df.groupby("patient_code")["patient_id"].nunique() != 1).sum())),
        ("Diagnosis akhir konsisten di semua kunjungan satu pasien", int((g["final_diagnosis"].nunique() != 1).sum())),
        ("Nomor kunjungan berurutan 1..n tanpa duplikat",
         int(((g["visit_number"].nunique() != g.size()) | (g["visit_number"].max() != g.size())).sum())),
        ("Tanggal kunjungan menaik per pasien", int((g["visit_date"].diff().dt.days < 0).sum())),
        ("Usia kehamilan menaik per pasien", int((g["gestational_age_weeks"].diff() < 0).sum())),
        ("Sistolik > diastolik", int((df["systolic_bp"] <= df["diastolic_bp"]).sum())),
        ("MAP = (sistolik + 2 x diastolik) / 3 (toleransi 0,6)",
         int(((df["map_mmhg"] - (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3).abs() > 0.6).sum())),
        ("Kasus PE punya tanggal diagnosis, non-PE tidak", int((~pe_ok).sum())),
        ("Tanggal kunjungan tidak mendahului HPHT", int((df["visit_date"] < df["lmp_date"]).sum())),
        ("Tanggal kunjungan tidak melewati tanggal persalinan", int((df["visit_date"] > df["delivery_date"]).sum())),
        ("Rujukan tidak menuju fasilitas yang sama dengan tempat kunjungan",
         int(((df["referral_status"] == "Dirujuk") & (df["fasilitas_tujuan"].astype(str) == df["fasilitas"].astype(str))).sum())),
        ("Usia ibu sesuai tanggal lahir (selisih <= 1 tahun)", int(((umur_dari_dob - df["maternal_age"]).abs() > 1).sum())),
    ]
    out = pd.DataFrame(cek, columns=["Pemeriksaan", "Baris bermasalah"])
    out["Status"] = np.where(out["Baris bermasalah"] == 0, "Lolos", "Perlu dicek")
    return out


@st.cache_data(show_spinner="Memuat dan memvalidasi data...")
def siapkan(path: str):
    tipe_awal = pd.read_csv(path, encoding="utf-8-sig").dtypes.astype(str)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig").replace({"": np.nan})
    df = raw.replace(PLACEHOLDER, np.nan)

    khusus = set(ID_COLS + TEXT_COLS + DATE_COLS + INT_COLS + FLOAT_COLS + list(ORDERED))
    bool_cols = [c for c in df.columns if c not in khusus
                 and df[c].notna().any() and df[c].dropna().isin(["Ya", "Tidak"]).all()]

    for c in ID_COLS + TEXT_COLS:
        df[c] = df[c].astype("string")
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")
    for c in INT_COLS:
        s = pd.to_numeric(df[c], errors="coerce")
        df[c] = s.astype("int64") if s.notna().all() else s.astype("Int64")
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in bool_cols:
        df[c] = df[c].map({"Ya": True, "Tidak": False}).astype("boolean")
    for c, cats in ORDERED.items():
        df[c] = pd.Categorical(df[c], categories=cats, ordered=True)
    sudah = khusus | set(bool_cols)
    for c in df.columns:
        if c not in sudah:
            df[c] = df[c].astype("category")

    # Nama fasilitas versi singkat (tanpa kata "Sintetis")
    for src, dst in [("facility_name", "fasilitas"), ("destination_facility_name", "fasilitas_tujuan"),
                     ("delivery_facility_name", "fasilitas_persalinan")]:
        df[dst] = df[src].astype("string").str.replace(" Sintetis", "", regex=False).astype("category")

    df = df.sort_values(["patient_id", "visit_number"]).reset_index(drop=True)

    # ---- tabel level pasien (1 baris = 1 pasien) ----
    g = df.groupby("patient_id", observed=True)
    first = g.first()
    agg = g.agg(
        n_kunjungan=("visit_id", "count"),
        kunjungan_pertama=("visit_date", "min"),
        kunjungan_terakhir=("visit_date", "max"),
        sistolik_maks=("systolic_bp", "max"),
        diastolik_maks=("diastolic_bp", "max"),
        map_maks=("map_mmhg", "max"),
        risiko_tertinggi=("screening_risk_category", "max"),
        curiga=("suspected_preeclampsia", lambda s: bool(s.fillna(False).any())),
        dirujuk=("referral_status", lambda s: bool((s == "Dirujuk").any())),
    )
    kolom_first = ["patient_code", "domicile_district", "fasilitas", "maternal_age", "final_diagnosis",
                   "diagnosis_date", "diagnosis_source", "lmp_date", "eclampsia", "hellp",
                   "fasilitas_persalinan", "multiple_pregnancy_type", "conception_method",
                   "prepregnancy_bmi", "interpregnancy_interval_years"] + \
                  [c for c in FAKTOR if c in df.columns]
    pat = agg.join(first[kolom_first]).rename(columns={
        "patient_code": "kode", "domicile_district": "wilayah", "fasilitas": "fasilitas_awal",
        "maternal_age": "usia", "final_diagnosis": "diagnosis", "diagnosis_date": "tgl_diagnosis",
    }).reset_index()
    for c in ["kode", "wilayah", "fasilitas_awal", "diagnosis", "risiko_tertinggi", "fasilitas_persalinan",
              "diagnosis_source", "multiple_pregnancy_type", "conception_method"]:
        pat[c] = pat[c].astype(str)
    for c in [c for c in FAKTOR if c in df.columns] + ["eclampsia", "hellp"]:
        pat[c] = pat[c].fillna(False).astype(bool)
    pat["kembar"] = pat["multiple_pregnancy_type"] != "Tunggal"
    pat["ivf"] = pat["conception_method"] == "Teknologi reproduksi berbantu"
    pat["usia35"] = pat["usia"] >= 35
    pat["obesitas"] = pat["prepregnancy_bmi"] > 30
    pat["jarak10"] = pat["interpregnancy_interval_years"] > 10
    pat["pe"] = pat["diagnosis"] == "Preeklamsia"
    pat["ga_diagnosis"] = (pat["tgl_diagnosis"] - pat["lmp_date"]).dt.days / 7

    # ---- audit tipe data ----
    audit = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe awal (tebakan pandas)": [tipe_awal.get(c, "(kolom turunan)") for c in df.columns],
        "Tipe akhir": [str(df[c].dtype) for c in df.columns],
        "Jenis": [jenis_tipe(df[c]) for c in df.columns],
        "Penanda diganti kosong": [int(raw[c].isin(PLACEHOLDER).sum()) if c in raw.columns else 0 for c in df.columns],
        "Kosong (akhir)": [int(df[c].isna().sum()) for c in df.columns],
        "Nilai unik": [int(df[c].nunique()) for c in df.columns],
        "Rentang / jumlah kategori": [rentang(df[c]) for c in df.columns],
        "Contoh": [", ".join(df[c].dropna().astype(str).unique()[:3]) for c in df.columns],
    })
    return df, pat, audit, validasi(df)


# =====================================================================
# 3. KOMPONEN BERSAMA
# =====================================================================
KOLOM_TABEL = {
    "kode": "Kode pasien", "wilayah": "Wilayah", "fasilitas_awal": "Fasilitas ANC awal",
    "usia": "Usia", "n_kunjungan": "Jml kunjungan", "risiko_tertinggi": "Risiko skrining tertinggi",
    "dirujuk": "Pernah dirujuk", "map_maks": "MAP maks", "diagnosis": "Diagnosis akhir",
}


def tabel_pasien(tabel: pd.DataFrame, key: str):
    """Tabel pasien yang bisa diklik. Mengembalikan patient_id baris terpilih (atau None)."""
    tabel = tabel.reset_index(drop=True)
    st.caption(f"{len(tabel)} pasien. **Klik satu baris** untuk melihat riwayat lengkapnya di bawah.")
    event = st.dataframe(
        tabel[list(KOLOM_TABEL)].rename(columns=KOLOM_TABEL),
        hide_index=True, width="stretch", height=300,
        on_select="rerun", selection_mode="single-row", key=key,
    )
    baris = event.selection.rows
    if baris and baris[0] < len(tabel):
        return tabel.loc[baris[0], "patient_id"]
    return None


def tabel_proporsi(pat: pd.DataFrame, by: str) -> pd.DataFrame:
    t = pat.groupby(by, observed=True).agg(pasien=("patient_id", "count"), kasus_pe=("pe", "sum")).reset_index()
    t["persen_pe"] = t["kasus_pe"] / t["pasien"] * 100
    t["label"] = t["kasus_pe"].astype(int).astype(str) + "/" + t["pasien"].astype(str) + " (" + t["persen_pe"].round(1).astype(str) + "%)"
    return t


def kurva_kelompok(df: pd.DataFrame, kolom: str, lebar: int = 4) -> pd.DataFrame:
    d = df[["final_diagnosis", "gestational_age_weeks", kolom]].dropna().copy()
    d[kolom] = d[kolom].astype(float)
    d["ga"] = (d["gestational_age_weeks"] // lebar) * lebar + lebar / 2
    return d.groupby(["final_diagnosis", "ga"], observed=True)[kolom].mean().reset_index()


# =====================================================================
# 4. RIWAYAT SATU PASIEN
# =====================================================================
def grafik_tekanan_darah(v, df, p, f_last, bandingkan, key):
    fig = go.Figure()
    simbol = np.where(v["referral_status"] == "Dirujuk", "star", "circle")
    ukuran = np.where(v["referral_status"] == "Dirujuk", 14, 9)
    cd = np.stack([v["visit_number"], v["visit_date"].dt.strftime("%d/%m/%Y"),
                   v["status_at_visit"].astype(str), v["fasilitas"].astype(str)], axis=-1)
    tmpl = ("Kunjungan %{customdata[0]} (%{customdata[1]}) di %{customdata[3]}<br>"
            "Usia hamil %{x:.1f} mgg<br>%{fullData.name}: %{y}<br>Status: %{customdata[2]}<extra></extra>")
    for kolom, nama, warna, dash in [("systolic_bp", "Sistolik", "#d62728", "solid"),
                                     ("diastolic_bp", "Diastolik", "#1f77b4", "solid"),
                                     ("map_mmhg", "MAP", "#2ca02c", "dash")]:
        fig.add_trace(go.Scatter(x=v["gestational_age_weeks"], y=v[kolom].astype(float), mode="lines+markers",
                                 name=nama, line=dict(color=warna, dash=dash), marker=dict(size=ukuran, symbol=simbol),
                                 customdata=cd, hovertemplate=tmpl))
    if bandingkan != "Tidak":
        kol = {"Sistolik": "systolic_bp", "Diastolik": "diastolic_bp", "MAP": "map_mmhg"}[bandingkan]
        kg = kurva_kelompok(df, kol)
        for dx, warna in [("Non-preeklamsia", "#7f8c8d"), ("Preeklamsia", "#f5a3a3")]:
            s = kg[kg["final_diagnosis"] == dx]
            fig.add_trace(go.Scatter(x=s["ga"], y=s[kol], mode="lines", name=f"Rata-rata {dx} ({bandingkan})",
                                     line=dict(color=warna, dash="dot", width=3), opacity=0.8))
    fig.add_hline(y=140, line_dash="dash", line_color="rgba(214,39,40,0.5)",
                  annotation_text="Sistolik 140", annotation_position="top left")
    fig.add_hline(y=90, line_dash="dash", line_color="rgba(31,119,180,0.5)",
                  annotation_text="Diastolik 90 / MAP > 90 (Buku KIA)", annotation_position="bottom left")
    if pd.notna(p["ga_diagnosis"]):
        fig.add_vline(x=float(p["ga_diagnosis"]), line_dash="dot", line_color="#d62728",
                      annotation_text="Diagnosis PE", annotation_position="top right")
    if pd.notna(f_last["gestational_age_delivery"]):
        fig.add_vline(x=float(f_last["gestational_age_delivery"]), line_dash="dot", line_color="#888",
                      annotation_text="Persalinan", annotation_position="top left")
    fig.update_layout(height=430, margin=dict(t=30, b=10), hovermode="closest",
                      xaxis_title="Usia kehamilan (minggu)", yaxis_title="mmHg",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch", key=key)
    st.caption("Bintang = kunjungan yang berujung rujukan. Garis putus-putus = ambang klinis.")


def grafik_berat(v, key):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=v["gestational_age_weeks"], y=v["weight_kg"], mode="lines+markers",
                             name="Berat badan (kg)", line=dict(color="#ff7f0e")), secondary_y=False)
    fig.add_trace(go.Scatter(x=v["gestational_age_weeks"], y=v["current_bmi"], mode="lines+markers",
                             name="IMT saat ini", line=dict(color="#9467bd", dash="dash")), secondary_y=True)
    fig.update_layout(height=360, margin=dict(t=20, b=10), xaxis_title="Usia kehamilan (minggu)",
                      legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(title_text="kg", secondary_y=False)
    fig.update_yaxes(title_text="IMT (kg/m²)", secondary_y=True)
    st.plotly_chart(fig, width="stretch", key=key)


def grafik_lab(v, key):
    panjang = v.melt(id_vars=["visit_number", "gestational_age_weeks"], value_vars=list(LAB),
                     var_name="param", value_name="nilai").dropna(subset=["nilai"])
    if panjang.empty:
        st.info("Tidak ada pemeriksaan laboratorium yang tercatat pada pasien ini.")
        return
    panjang["Parameter"] = panjang["param"].map(LAB)
    fig = px.line(panjang, x="gestational_age_weeks", y="nilai", facet_col="Parameter", facet_col_wrap=3,
                  markers=True, labels={"gestational_age_weeks": "Usia hamil (mgg)", "nilai": ""})
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=420, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch", key=key)
    tdk = [LAB[c] for c in LAB if v[c].isna().all()]
    if tdk:
        st.caption("Tidak pernah diperiksa pada pasien ini: " + ", ".join(tdk) + ".")


def tabel_kunjungan(v):
    gejala_map = {"headache": "sakit kepala", "visual_disturbance": "gangguan penglihatan",
                  "epigastric_pain": "nyeri ulu hati", "dyspnea": "sesak"}
    gejala = v.apply(lambda r: ", ".join(n for c, n in gejala_map.items() if ya(r[c])) or "–", axis=1)
    t = pd.DataFrame({
        "Kunj.": v["visit_number"].astype(int),
        "Tanggal": v["visit_date"].dt.strftime("%d/%m/%Y"),
        "Usia hamil (mgg)": v["gestational_age_weeks"].round(1),
        "Fasilitas": v["fasilitas"].astype(str),
        "TD (mmHg)": v["systolic_bp"].astype(str) + "/" + v["diastolic_bp"].astype(str),
        "MAP": v["map_mmhg"].round(1),
        "Δ sistolik": v["systolic_change"],
        "Berat (kg)": v["weight_kg"],
        "Edema": np.where(v["edema_present"].fillna(False).astype(bool), v["edema_location"].astype(str), "–"),
        "Proteinuria": v["proteinuria_result"].astype(str),
        "Gejala": gejala,
        "Risiko skrining": v["screening_risk_category"].astype(str),
        "Dugaan PE": np.where(v["suspected_preeclampsia"].fillna(False).astype(bool), "Ya", "Tidak"),
        "Rujukan": np.where(v["referral_status"] == "Dirujuk", "→ " + v["fasilitas_tujuan"].astype(str), "–"),
        "Status": v["status_at_visit"].astype(str),
        "Tata laksana": v["management_notes"].astype(str),
    })
    tinggi = ((v["systolic_bp"] >= 140) | (v["diastolic_bp"] >= 90)).reset_index(drop=True)
    idx_td = t.columns.get_loc("TD (mmHg)")
    latar = {"PE terdiagnosis": "background-color: rgba(214,39,40,0.28)",
             "Belum PE": "background-color: rgba(242,166,90,0.22)"}

    def gaya(r):
        css = [latar.get(r["Status"], "")] * len(r)
        if tinggi.iloc[r.name]:
            css[idx_td] += ";color:#ff6b6b;font-weight:700"
        return css

    st.dataframe(t.reset_index(drop=True).style.apply(gaya, axis=1), hide_index=True, width="stretch")
    st.caption("Latar merah = PE terdiagnosis, oranye = belum terdiagnosis tetapi pasien akhirnya PE. "
               "TD merah tebal = sistolik >= 140 atau diastolik >= 90.")


def tabel_gejala(v):
    def tanda(c):
        return np.where(v[c].fillna(False).astype(bool), "●", "·")
    t = pd.DataFrame({
        "Kunj.": v["visit_number"].astype(int),
        "Tanggal": v["visit_date"].dt.strftime("%d/%m/%Y"),
        "Edema": tanda("edema_present"), "Lokasi edema": v["edema_location"].astype(str),
        "Edema menetap": tanda("edema_persistent"), "Sakit kepala": tanda("headache"),
        "Gangguan penglihatan": tanda("visual_disturbance"), "Nyeri ulu hati": tanda("epigastric_pain"),
        "Sesak": tanda("dyspnea"), "Proteinuria": v["proteinuria_result"].astype(str),
    })
    st.dataframe(t, hide_index=True, width="stretch")
    st.caption("● = ada, · = tidak ada. Proteinuria 'Tidak diperiksa' = tidak ada hasil dipstick pada kunjungan itu.")


def linimasa(v, f):
    ev = []
    for _, r in v.iterrows():
        ev.append((r["visit_date"], f"Kunjungan ANC ke-{int(r['visit_number'])}",
                   f"{r['fasilitas']} · usia hamil {r['gestational_age_weeks']:.1f} mgg · TD {r['systolic_bp']}/{r['diastolic_bp']} · {r['status_at_visit']}"))
        if r["referral_status"] == "Dirujuk":
            tgl = r["referral_date"] if pd.notna(r["referral_date"]) else r["visit_date"]
            ev.append((tgl, "Rujukan", f"ke {r['fasilitas_tujuan']} · alasan: {r['referral_reason']}"))
    if pd.notna(f["diagnosis_date"]):
        ev.append((f["diagnosis_date"], "Diagnosis preeklamsia", f"sumber: {f['diagnosis_source']}"))
    if pd.notna(f["delivery_date"]):
        ev.append((f["delivery_date"], "Persalinan",
                   f"{f['fasilitas_persalinan']} · {f['delivery_mode']} · usia kehamilan {f['gestational_age_delivery']:.1f} mgg"))
    if pd.notna(f["postpartum_visit_date"]):
        ev.append((f["postpartum_visit_date"], "Kontrol pascasalin",
                   f"{f['postpartum_symptoms']} · TD {f['postpartum_systolic_bp']}/{f['postpartum_diastolic_bp']}"))
    t = pd.DataFrame(ev, columns=["Tanggal", "Peristiwa", "Detail"]).sort_values("Tanggal", kind="stable")
    t["Tanggal"] = t["Tanggal"].dt.strftime("%d/%m/%Y")
    st.dataframe(t, hide_index=True, width="stretch")


def kv(judul, pasangan):
    st.markdown(f"**{judul}**")
    st.dataframe(pd.DataFrame(pasangan, columns=["Item", "Nilai"]), hide_index=True, width="stretch")


def render_pasien(pid, df, pat, key):
    v = df[df["patient_id"] == pid].sort_values("visit_number").reset_index(drop=True)
    p = pat[pat["patient_id"] == pid].iloc[0]
    f0, f = v.iloc[0], v.iloc[-1]
    is_pe = bool(p["pe"])

    st.divider()
    warna = "red" if is_pe else "blue"
    st.subheader(f"Riwayat pasien {p['kode']}")
    st.markdown(f"Diagnosis akhir: :{warna}[**{p['diagnosis']}**] · sumber diagnosis: {p['diagnosis_source']}")

    c = st.columns(4)
    c[0].metric("Usia (kunjungan pertama)", f"{int(p['usia'])} th")
    c[1].metric("Wilayah (domisili)", p["wilayah"])
    c[2].metric("Kunjungan ANC", f"{int(p['n_kunjungan'])}×",
                f"{p['kunjungan_pertama']:%d/%m/%Y} – {p['kunjungan_terakhir']:%d/%m/%Y}", delta_color="off")
    c[3].metric("Risiko skrining tertinggi", p["risiko_tertinggi"])
    c = st.columns(4)
    rute = " → ".join(dict.fromkeys(v["fasilitas"].astype(str)))
    c[0].metric("Fasilitas ANC", rute)
    c[1].metric("Pernah dirujuk", "Ya" if p["dirujuk"] else "Tidak")
    c[2].metric("Tanggal diagnosis PE", f"{p['tgl_diagnosis']:%d/%m/%Y}" if pd.notna(p["tgl_diagnosis"]) else "—",
                f"usia hamil ≈ {p['ga_diagnosis']:.0f} mgg" if pd.notna(p["ga_diagnosis"]) else None, delta_color="off")
    c[3].metric("Persalinan", f"{f['delivery_date']:%d/%m/%Y}" if pd.notna(f["delivery_date"]) else "—",
                str(f["delivery_mode"]), delta_color="off")

    with st.expander("Profil & faktor risiko (riwayat sebelum kehamilan ini)", expanded=True):
        a, b = st.columns(2)
        kiri = [(lab, ya_tidak(p[col])) for col, lab in FAKTOR.items()]
        with a:
            kv("Faktor risiko skrining (Buku KIA)", kiri)
        with b:
            kv("Riwayat obstetri & karakteristik", [
                ("Gravida / paritas / abortus", f"{int(f0['gravida'])} / {int(f0['parity'])} / {int(f0['abortion_count'])}"),
                ("Jarak kehamilan sebelumnya (th)", f"{f0['interpregnancy_interval_years']:.1f}" if pd.notna(f0["interpregnancy_interval_years"]) else "—"),
                ("IMT sebelum hamil", f"{f0['prepregnancy_bmi']:.1f}"),
                ("Tinggi badan (cm)", f"{f0['height_cm']:.1f}"),
                ("Pendidikan", str(f0["education_level"])),
                ("Pekerjaan", str(f0["occupation"])),
                ("Status sosial ekonomi", str(f0["socioeconomic_status"])),
                ("HPHT / HPL", f"{f0['lmp_date']:%d/%m/%Y} / {f0['estimated_due_date']:%d/%m/%Y}"),
            ])

    st.markdown("### Perkembangan selama kehamilan")
    t1, t2, t3, t4 = st.tabs(["Tekanan darah", "Berat badan & IMT", "Laboratorium", "Gejala & proteinuria"])
    with t1:
        bandingkan = st.selectbox("Bandingkan dengan rata-rata kelompok (per 4 minggu)",
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
                ("Fasilitas persalinan", str(f["fasilitas_persalinan"])),
                ("Cara persalinan", str(f["delivery_mode"])),
                ("Komplikasi", str(f["delivery_complication_type"])),
                ("Eklampsia", ya_tidak(f["eclampsia"])), ("HELLP", ya_tidak(f["hellp"])),
                ("Perdarahan pascapersalinan", ya_tidak(f["postpartum_hemorrhage"])),
                ("Perawatan intensif", ya_tidak(f["maternal_intensive_care"])),
                ("Status saat pulang", str(f["maternal_status_discharge"])),
            ])
        with b:
            kv("Pascasalin", [
                ("Tanggal kontrol", f"{f['postpartum_visit_date']:%d/%m/%Y}" if pd.notna(f["postpartum_visit_date"]) else "—"),
                ("TD pascasalin", f"{f['postpartum_systolic_bp']}/{f['postpartum_diastolic_bp']}"),
                ("Hipertensi pascasalin", ya_tidak(f["postpartum_hypertension_status"])),
                ("Gejala", str(f["postpartum_symptoms"])),
                ("Dirujuk pascasalin", ya_tidak(f["postpartum_referral"])),
            ])
        with c3:
            kv("Bayi", [
                ("Jumlah bayi / jenis kelamin", f"{int(f['baby_count'])} / {f['baby_sex']}"),
                ("Berat / panjang lahir", f"{int(f['birth_weight_g'])} g / {f['birth_length_cm']:.1f} cm"),
                ("Apgar 1' / 5'", f"{int(f['apgar_1min'])} / {int(f['apgar_5min'])}"),
                ("Status lahir", str(f["birth_status"])),
                ("BBLR / prematur", f"{ya_tidak(f['bblr_status'])} / {ya_tidak(f['prematurity_status'])}"),
                ("Masuk NICU", ya_tidak(f["nicu_admission"])),
                ("Komplikasi neonatal", ya_tidak(f["neonatal_complication"])),
            ])


# =====================================================================
# 5. HALAMAN
# =====================================================================
def halaman_ringkasan(df, pat):
    st.header("Ringkasan")
    n, n_pe = len(pat), int(pat["pe"].sum())
    c = st.columns(5)
    c[0].metric("Pasien", f"{n:,}")
    c[1].metric("Kunjungan ANC", f"{len(df):,}", f"{len(df) / n:.1f} per pasien", delta_color="off")
    c[2].metric("Pasien preeklamsia", f"{n_pe}")
    c[3].metric("Proporsi PE", f"{n_pe / n * 100:.1f}%")
    c[4].metric("Pasien pernah dirujuk", f"{int(pat['dirujuk'].sum())}")
    st.caption("Semua angka pasien dihitung per pasien (bukan per baris kunjungan), karena satu pasien punya 3–6 kunjungan.")

    a, b = st.columns(2)
    with a:
        pe = pat[pat["pe"]]
        st.markdown("**Seberapa awal kasus PE terdeteksi di layanan ANC?**")
        d = st.columns(3)
        d[0].metric("Kasus PE", f"{len(pe)}")
        d[1].metric("Pernah dicurigai PE saat ANC", f"{int(pe['curiga'].sum())}")
        d[2].metric("Pernah dirujuk", f"{int(pe['dirujuk'].sum())}")
        tahap = pd.DataFrame({
            "Tahap": ["Semua pasien", "Risiko skrining sedang/tinggi", "Pernah dicurigai PE", "Pernah dirujuk", "PE terdiagnosis"],
            "Pasien": [n, int(pat["risiko_tertinggi"].isin(["Sedang", "Tinggi"]).sum()), int(pat["curiga"].sum()),
                       int(pat["dirujuk"].sum()), n_pe],
        })
        fig = px.bar(tahap, x="Pasien", y="Tahap", orientation="h", text="Pasien")
        fig.update_layout(height=300, margin=dict(t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch", key="rg_tahap")
        st.caption("Tahap-tahap ini tidak saling bersarang (pasien PE tidak harus pernah dirujuk).")
    with b:
        dist = pat["diagnosis"].value_counts().reset_index()
        dist.columns = ["Diagnosis", "Pasien"]
        fig = px.pie(dist, names="Diagnosis", values="Pasien", hole=0.5, color="Diagnosis",
                     color_discrete_map=WARNA_DX, title="Diagnosis akhir (per pasien)")
        fig.update_layout(height=370, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch", key="rg_pie")

    a, b = st.columns(2)
    for kol, judul, kolom_ui, kk in [("wilayah", "Proporsi PE per wilayah (domisili)", a, "rg_wil"),
                                     ("fasilitas_awal", "Proporsi PE per fasilitas ANC awal", b, "rg_fas")]:
        t = tabel_proporsi(pat, kol).sort_values("persen_pe", ascending=False)
        fig = px.bar(t, x=kol, y="persen_pe", text="label", title=judul,
                     labels={kol: "", "persen_pe": "% pasien PE"})
        fig.update_layout(height=360, margin=dict(t=40, b=10))
        fig.update_traces(textposition="outside", cliponaxis=False)
        kolom_ui.plotly_chart(fig, width="stretch", key=kk)
    st.caption("Label = kasus PE / pasien (persen). Dengan kasus PE yang sedikit, persentase per kelompok mudah naik-turun.")


def halaman_pasien(df, pat):
    st.header("Riwayat per pasien")
    c = st.columns(5)
    wil = c[0].multiselect("Wilayah (domisili)", sorted(pat["wilayah"].unique()))
    fas = c[1].multiselect("Pernah ANC di fasilitas", sorted(df["fasilitas"].astype(str).unique()))
    dx = c[2].selectbox("Diagnosis akhir", ["Semua", "Preeklamsia", "Non-preeklamsia"])
    rs = c[3].selectbox("Risiko skrining tertinggi", ["Semua", "Rendah", "Sedang", "Tinggi"])
    cari = c[4].text_input("Cari kode pasien", placeholder="mis. P0005")
    hanya = st.checkbox("Hanya pasien yang pernah dirujuk")

    t = pat
    if wil:
        t = t[t["wilayah"].isin(wil)]
    if fas:
        ids = df.loc[df["fasilitas"].astype(str).isin(fas), "patient_id"].unique()
        t = t[t["patient_id"].isin(ids)]
    if dx != "Semua":
        t = t[t["diagnosis"] == dx]
    if rs != "Semua":
        t = t[t["risiko_tertinggi"] == rs]
    if cari:
        t = t[t["kode"].str.contains(cari.strip(), case=False)]
    if hanya:
        t = t[t["dirujuk"]]
    if t.empty:
        st.warning("Tidak ada pasien yang cocok dengan filter.")
        return
    pid = tabel_pasien(t, "tbl_pasien")
    if pid is None:
        st.info("Pilih satu pasien dari tabel untuk melihat riwayatnya.")
    else:
        render_pasien(pid, df, pat, "psn")


def halaman_wilayah(df, pat):
    st.header("Wilayah & fasilitas")
    mode = st.radio("Tingkat analisis", ["Per wilayah (kecamatan domisili)", "Per fasilitas (Puskesmas / Rumah Sakit)"],
                    horizontal=True)
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
            s = pat[pat["patient_id"].isin(df.loc[fas_all == nama, "patient_id"])]
            baris.append({"nama": nama, "pasien": len(s), "kasus_pe": int(s["pe"].sum())})
        banding = pd.DataFrame(baris)
        banding["persen_pe"] = banding["kasus_pe"] / banding["pasien"] * 100
        banding["label"] = banding["kasus_pe"].astype(str) + "/" + banding["pasien"].astype(str) + " (" + banding["persen_pe"].round(1).astype(str) + "%)"

    sp = pat[pat["patient_id"].isin(ids)]
    sv = df[df["patient_id"].isin(ids)]
    if sp.empty:
        st.warning("Tidak ada data.")
        return

    c = st.columns(6)
    c[0].metric("Pasien", len(sp))
    c[1].metric("Kunjungan ANC", len(sv))
    c[2].metric("Kasus PE", int(sp["pe"].sum()))
    c[3].metric("Proporsi PE", f"{sp['pe'].mean() * 100:.1f}%")
    c[4].metric("Pasien pernah dirujuk", int(sp["dirujuk"].sum()))
    c[5].metric("Rata-rata sistolik", f"{sv['systolic_bp'].astype(float).mean():.0f} mmHg")

    if not per_wilayah:
        st.markdown(f"**Peran {pilih} dalam alur layanan**")
        tuj = df["fasilitas_tujuan"].astype(str)
        masuk = df.loc[(tuj == pilih) & (fas_all != pilih), "patient_id"].nunique()
        keluar = df.loc[(fas_all == pilih) & (df["referral_status"] == "Dirujuk") & (tuj != pilih), "patient_id"].nunique()
        lahir = int((pat["fasilitas_persalinan"] == pilih).sum())
        d = st.columns(4)
        d[0].metric("Pasien dirujuk KELUAR", keluar)
        d[1].metric("Pasien dirujuk MASUK", masuk)
        d[2].metric("Persalinan di sini", lahir)
        d[3].metric("Kunjungan ANC di sini", int((fas_all == pilih).sum()))

    a, b = st.columns(2)
    with a:
        banding["Dipilih"] = banding["nama"] == pilih
        fig = px.bar(banding.sort_values("persen_pe", ascending=False), x="nama", y="persen_pe", text="label",
                     color="Dipilih", color_discrete_map={True: "#E45756", False: "#9aa5b1"},
                     title="Proporsi PE: " + ("semua wilayah" if per_wilayah else "semua fasilitas"),
                     labels={"nama": "", "persen_pe": "% pasien PE"})
        fig.update_layout(height=370, margin=dict(t=40, b=10), showlegend=False)
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, width="stretch", key="wf_banding")
    with b:
        if per_wilayah:
            k = sv["fasilitas"].astype(str).value_counts().reset_index()
            k.columns = ["Fasilitas", "Kunjungan"]
            fig = px.bar(k, x="Fasilitas", y="Kunjungan", title=f"Ke mana pasien {pilih} berobat (ANC)")
        else:
            th = sv[fas_all.loc[sv.index] == pilih].copy()
            th["Tahun"] = th["visit_date"].dt.year
            k = th.groupby("Tahun").size().reset_index(name="Kunjungan")
            fig = px.bar(k, x="Tahun", y="Kunjungan", title=f"Kunjungan ANC per tahun di {pilih}")
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

    # ---- a. tren kasus ----
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
    fig.add_trace(go.Bar(x=tren["Periode"], y=tren["Kasus baru"], name="Kasus PE baru", marker_color="#E45756"), secondary_y=False)
    fig.add_trace(go.Scatter(x=tren["Periode"], y=tren["Kumulatif"], name="Kumulatif", mode="lines+markers",
                             line=dict(color="#555")), secondary_y=True)
    fig.update_layout(height=340, margin=dict(t=20, b=10), legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="Kasus baru", secondary_y=False)
    fig.update_yaxes(title_text="Kumulatif", secondary_y=True)
    st.plotly_chart(fig, width="stretch", key="hs_tren")
    st.caption("Dihitung dari tanggal diagnosis PE (per pasien).")

    pat2 = pat.assign(tahun=pat["kunjungan_pertama"].dt.year.astype(str))
    tp = tabel_proporsi(pat2, "tahun")
    fig = px.bar(tp, x="tahun", y="persen_pe", text="label",
                 title="Proporsi PE menurut tahun kunjungan ANC pertama", labels={"tahun": "", "persen_pe": "% pasien PE"})
    fig.update_layout(height=330, margin=dict(t=40, b=10))
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, width="stretch", key="hs_tahun")

    # ---- b. faktor risiko & riwayat PE sebelumnya ----
    st.subheader("2. Riwayat PE sebelumnya & faktor risiko")
    baris = []
    for col, lab in FAKTOR.items():
        ada, tdk = pat[pat[col]], pat[~pat[col]]
        r_ada = ada["pe"].mean() * 100 if len(ada) else np.nan
        r_tdk = tdk["pe"].mean() * 100 if len(tdk) else np.nan
        baris.append({"Faktor": lab, "Pasien dengan faktor": len(ada), "PE (dengan)": int(ada["pe"].sum()),
                      "% PE (dengan)": r_ada, "Pasien tanpa faktor": len(tdk), "PE (tanpa)": int(tdk["pe"].sum()),
                      "% PE (tanpa)": r_tdk,
                      "Rasio risiko": (r_ada / r_tdk) if (r_tdk and not np.isnan(r_ada)) else np.nan})
    ft = pd.DataFrame(baris)
    lg = ft.melt(id_vars="Faktor", value_vars=["% PE (dengan)", "% PE (tanpa)"], var_name="Kelompok", value_name="% PE")
    lg["Kelompok"] = lg["Kelompok"].map({"% PE (dengan)": "Dengan faktor", "% PE (tanpa)": "Tanpa faktor"})
    fig = px.bar(lg, x="Faktor", y="% PE", color="Kelompok", barmode="group",
                 color_discrete_map={"Dengan faktor": "#E45756", "Tanpa faktor": "#9aa5b1"})
    fig.update_layout(height=380, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch", key="hs_faktor")
    st.dataframe(ft.round(2), hide_index=True, width="stretch")
    st.caption("Kasus PE hanya sedikit, jadi rasio risiko di sini bersifat gambaran awal, bukan bukti statistik. "
               "Rasio kosong = tidak ada kasus/pasien pada salah satu kelompok.")

    # ---- c. lintasan tekanan darah ----
    st.subheader("3. Lintasan tekanan darah: pasien PE vs non-PE")
    pilih = st.radio("Parameter", ["Sistolik", "Diastolik", "MAP"], horizontal=True, key="hs_param")
    kol = {"Sistolik": "systolic_bp", "Diastolik": "diastolic_bp", "MAP": "map_mmhg"}[pilih]
    d = df[["patient_code", "final_diagnosis", "gestational_age_weeks", kol]].copy()
    d[kol] = d[kol].astype(float)
    d["ga"] = (d["gestational_age_weeks"] // 4) * 4 + 2
    non = d[d["final_diagnosis"] == "Non-preeklamsia"].groupby("ga")[kol].agg(
        rata="mean", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75)).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(non["ga"]) + list(non["ga"][::-1]), y=list(non["q3"]) + list(non["q1"][::-1]),
                             fill="toself", fillcolor="rgba(76,120,168,0.2)", line=dict(width=0),
                             name="Non-PE: rentang 25–75%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=non["ga"], y=non["rata"], mode="lines", name="Rata-rata non-PE",
                             line=dict(color="#4C78A8", width=3)))
    for kode, g in d[d["final_diagnosis"] == "Preeklamsia"].groupby("patient_code", observed=True):
        fig.add_trace(go.Scatter(x=g["gestational_age_weeks"], y=g[kol], mode="lines+markers", showlegend=False,
                                 line=dict(color="rgba(228,87,86,0.55)", width=1.5), marker=dict(size=5),
                                 name=str(kode), hovertemplate=f"{kode}<br>Usia hamil %{{x:.1f}} mgg<br>{pilih} %{{y}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Pasien PE (tiap garis = 1 pasien)",
                             line=dict(color="#E45756")))
    fig.update_layout(height=420, margin=dict(t=10, b=10), xaxis_title="Usia kehamilan (minggu)",
                      yaxis_title=f"{pilih} (mmHg)", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch", key="hs_lintasan")
    st.caption("Kelompok non-PE juga mencakup ibu bertekanan darah tinggi yang setelah pemeriksaan lanjutan tidak didiagnosis PE.")

    # ---- d. status per kunjungan ----
    st.subheader("4. Status pasien menurut nomor kunjungan")
    s = df.groupby(["visit_number", "status_at_visit"], observed=True).size().reset_index(name="Kunjungan")
    s["status_at_visit"] = s["status_at_visit"].astype(str)
    fig = px.bar(s, x="visit_number", y="Kunjungan", color="status_at_visit", color_discrete_map=WARNA_STATUS,
                 labels={"visit_number": "Kunjungan ke-", "status_at_visit": "Status"})
    fig.update_layout(barnorm="percent", height=330, margin=dict(t=10, b=10), yaxis_title="% kunjungan",
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch", key="hs_status")
    st.caption("'Belum PE' = kunjungan dari pasien yang kemudian terdiagnosis PE, tetapi diagnosisnya belum ditegakkan saat itu.")

    # ---- e. daftar kasus ----
    st.subheader("5. Daftar kasus PE (klik untuk membuka riwayat)")
    kasus = pat[pat["pe"]].sort_values("tgl_diagnosis")
    pid = tabel_pasien(kasus, "tbl_hist_pe")
    if pid is None:
        st.info("Klik satu kasus untuk melihat riwayat lengkap pasien.")
    else:
        render_pasien(pid, df, pat, "hs")


def halaman_kualitas(df, audit, cek):
    st.header("Kualitas data & tipe data")
    st.caption("Semua kolom dibaca sebagai teks lebih dulu, lalu diubah ke tipe yang tepat. Penanda seperti "
               "TIDAK_DIPERIKSA / BELUM_DINILAI / TIDAK_ADA / TIDAK_BERLAKU diganti menjadi nilai kosong.")
    jum = audit["Jenis"].value_counts().reset_index()
    jum.columns = ["Jenis", "Jumlah kolom"]
    a, b = st.columns([1, 2])
    a.dataframe(jum, hide_index=True, width="stretch")
    with b:
        st.markdown(f"**{len(df):,} baris × {df.shape[1]} kolom** · {df['patient_id'].nunique()} pasien · "
                    f"{audit['Penanda diganti kosong'].sum():,} penanda diganti kosong")
        st.markdown("**Pemeriksaan konsistensi**")
        st.dataframe(cek, hide_index=True, width="stretch")
    st.markdown("**Daftar tipe data per kolom**")
    jenis = st.multiselect("Saring jenis", sorted(audit["Jenis"].unique()))
    t = audit if not jenis else audit[audit["Jenis"].isin(jenis)]
    st.dataframe(t, hide_index=True, width="stretch", height=480)
    st.download_button("Unduh audit tipe data (CSV)", audit.to_csv(index=False).encode("utf-8"),
                       file_name="audit_tipe_data.csv", mime="text/csv")


# =====================================================================
# 6. APLIKASI
# =====================================================================
def main():
    st.set_page_config(page_title="Dashboard Riwayat Preeklamsia", page_icon="🩺", layout="wide")
    df, pat, audit, cek = siapkan(str(CSV_PATH))

    st.sidebar.title("🩺 Dashboard PE")
    halaman = st.sidebar.radio("Halaman", ["Ringkasan", "Riwayat per pasien", "Wilayah & fasilitas",
                                           "Data historis PE", "Kualitas data & tipe data"])
    st.sidebar.caption("Data sintetis ANC — Kabupaten Toba. Untuk pembelajaran, bukan data pasien nyata.")
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