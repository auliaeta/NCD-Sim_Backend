"""
NCD-Sim - Backend API dengan Model XGBoost Asli
================================================================================
Model dimuat dari: Models_XG_Boost_Penyakit.pkl
Isi file pkl (dict):
    - 'Heart'    → XGBClassifier  | fitur: _AGE80, WTKG3, _BMI5, HEIGHT3, _SMOKER3, INCOME3, _TOTINDA
    - 'Stroke'   → XGBClassifier  | fitur: _AGE80, WTKG3, _BMI5, HEIGHT3, INCOME3, _SMOKER3, _TOTINDA
    - 'Ginjal'   → XGBClassifier  | fitur: _AGE80, WTKG3, _BMI5, HEIGHT3, INCOME3, PHYSHLTH, DRNKANY6, _SMOKER3
    - 'Diabetes' → CalibratedClassifierCV (XGBoost) | fitur: _AGE80, WTKG3, _BMI5, HEIGHT3, _SMOKER3, _TOTINDA

Encoding BRFSS yang dipakai model:
    _AGE80    : usia (tahun, max 80)
    WTKG3     : berat badan dalam kg × 100  → 70 kg = 7000
    _BMI5     : BMI × 100                   → BMI 25.0 = 2500
    HEIGHT3   : tinggi badan dalam cm (langsung, misal 165)
    _SMOKER3  : 1=perokok aktif/hari, 2=kadang, 3=bekas, 4=tidak pernah
    INCOME3   : 1-8 (makin besar makin tinggi pendapatan)
    _TOTINDA  : 1=aktif fisik, 2=tidak aktif
    PHYSHLTH  : jumlah hari fisik buruk dalam 30 hari terakhir (0-30)
    DRNKANY6  : konsumsi alkohol → 1=ya, 2=tidak

Cara menjalankan:
    uvicorn app:app --reload

Cek endpoint di: http://localhost:8000/docs
"""

from fastapi import FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
import sqlite3
from datetime import datetime
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# SETUP APLIKASI

app = FastAPI(title="NCD-Sim API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "lifepath.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            email TEXT,
            usia INTEGER,
            jenis_kelamin INTEGER,
            bmi REAL,
            aktivitas_fisik INTEGER,
            merokok INTEGER,
            alkohol INTEGER,
            hari_kesehatan_fisik_buruk INTEGER,
            risiko_diabetes REAL,
            risiko_jantung REAL,
            risiko_stroke REAL,
            risiko_ginjal REAL
        )
    """)
    conn.commit()
    conn.close()


init_db()

# LOAD MODEL XGBoost dari file .pkl
# File harus ada di folder yang SAMA dengan app.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Models_XG_Boost_Penyakit.pkl"

print("BASE_DIR =", BASE_DIR)
print("MODEL_PATH =", MODEL_PATH)
print("EXISTS =", MODEL_PATH.exists())

try:
    MODELS = joblib.load(MODEL_PATH)

    # Support dua format:
    # 1. {'Heart': model, ...}
    # 2. {'models': {'Heart': model, ...}, ...}
    if isinstance(MODELS, dict) and "models" in MODELS:
        MODELS = MODELS["models"]

    print("Model XGBoost berhasil dimuat!")
    print(f"   Penyakit tersedia: {list(MODELS.keys())}"
    )
except FileNotFoundError:
    raise RuntimeError(
        "File 'Models_XG_Boost_Penyakit.pkl' tidak ditemukan!\n"
        "Pastikan file .pkl ada di folder yang sama dengan app.py."
    )

# FEATURE IMPORTANCE (diekstrak dari model saat training)
# Nama kolom BRFSS jd label ramah pengguna

LABEL_FITUR = {
    "_AGE80":   "Usia",
    "WTKG3":    "Berat badan",
    "_BMI5":    "BMI",
    "HEIGHT3":  "Tinggi badan",
    "_SMOKER3": "Kebiasaan merokok",
    "INCOME3":  "Pendapatan",
    "_TOTINDA": "Aktivitas fisik",
    "PHYSHLTH": "Hari fisik buruk",
    "DRNKANY6": "Konsumsi alkohol",
}

# Feature importance per penyakit (dari model.feature_importances_ / calibrated base)
FEATURE_IMPORTANCE = {
    "Heart": [
        {"fitur": "_AGE80",   "label": "Usia",               "importance": 0.5438},
        {"fitur": "_SMOKER3", "label": "Kebiasaan merokok",  "importance": 0.1383},
        {"fitur": "_TOTINDA", "label": "Aktivitas fisik",    "importance": 0.0913},
        {"fitur": "INCOME3",  "label": "Pendapatan",         "importance": 0.0778},
        {"fitur": "WTKG3",    "label": "Berat badan",        "importance": 0.0711},
        {"fitur": "HEIGHT3",  "label": "Tinggi badan",       "importance": 0.0450},
        {"fitur": "_BMI5",    "label": "BMI",                "importance": 0.0326},
    ],
    "Stroke": [
        {"fitur": "_AGE80",   "label": "Usia",               "importance": 0.4398},
        {"fitur": "INCOME3",  "label": "Pendapatan",         "importance": 0.2458},
        {"fitur": "_TOTINDA", "label": "Aktivitas fisik",    "importance": 0.1437},
        {"fitur": "_SMOKER3", "label": "Kebiasaan merokok",  "importance": 0.0910},
        {"fitur": "_BMI5",    "label": "BMI",                "importance": 0.0355},
        {"fitur": "WTKG3",    "label": "Berat badan",        "importance": 0.0252},
        {"fitur": "HEIGHT3",  "label": "Tinggi badan",       "importance": 0.0189},
    ],
    "Ginjal": [
        {"fitur": "_AGE80",   "label": "Usia",               "importance": 0.3653},
        {"fitur": "DRNKANY6", "label": "Konsumsi alkohol",   "importance": 0.2062},
        {"fitur": "PHYSHLTH", "label": "Hari fisik buruk",   "importance": 0.1924},
        {"fitur": "INCOME3",  "label": "Pendapatan",         "importance": 0.1024},
        {"fitur": "_BMI5",    "label": "BMI",                "importance": 0.0437},
        {"fitur": "WTKG3",    "label": "Berat badan",        "importance": 0.0360},
        {"fitur": "HEIGHT3",  "label": "Tinggi badan",       "importance": 0.0287},
        {"fitur": "_SMOKER3", "label": "Kebiasaan merokok",  "importance": 0.0253},
    ],
    "Diabetes": [
        {"fitur": "_AGE80",   "label": "Usia",               "importance": 0.4482},
        {"fitur": "_TOTINDA", "label": "Aktivitas fisik",    "importance": 0.2416},
        {"fitur": "_BMI5",    "label": "BMI",                "importance": 0.2245},
        {"fitur": "WTKG3",    "label": "Berat badan",        "importance": 0.0433},
        {"fitur": "_SMOKER3", "label": "Kebiasaan merokok",  "importance": 0.0266},
        {"fitur": "HEIGHT3",  "label": "Tinggi badan",       "importance": 0.0160},
    ],
}


# BENTUK DATA INPUT

class ProfilInput(BaseModel):
    """
    Input dari frontend. Encoding sesuai BRFSS:
        jenis_kelamin   : 1 = Laki-laki, 2 = Perempuan (info tambahan, tidak dipakai model saat ini)
        aktivitas_fisik : 1 = Aktif, 2 = Tidak aktif  (_TOTINDA)
        merokok         : 1 = Aktif/hari, 2 = Kadang, 3 = Bekas, 4 = Tidak pernah (_SMOKER3)
        alkohol         : 1 = Ya, 2 = Tidak  (DRNKANY6)
        pendidikan      : 1-6 (opsional, tidak dipakai model saat ini)
        pendapatan      : 1-8 (INCOME3, opsional — default 5 kalau tidak diisi)
    """
    usia: int = Field(gt=0, le=120)
    jenis_kelamin: Literal[1, 2]
    tinggi_cm: float = Field(gt=0, le=250)
    berat_kg: float = Field(gt=0, le=300)
    aktivitas_fisik: Literal[1, 2]
    merokok: Literal[1, 2, 3, 4]
    alkohol: Literal[1, 2]
    hari_kesehatan_fisik_buruk: int = Field(default=0, ge=0, le=30)
    pendidikan: Optional[Literal[1, 2, 3, 4, 5, 6]] = None
    pendapatan: Optional[Literal[1, 2, 3, 4, 5, 6, 7, 8]] = None


class SimulasiInput(BaseModel):
    usia: int = Field(gt=0, le=120)
    jenis_kelamin: Literal[1, 2]
    tinggi_cm: float = Field(gt=0, le=250)
    berat_kg: float = Field(gt=0, le=300)
    aktivitas_fisik: Literal[1, 2]
    merokok: Literal[1, 2, 3, 4]
    alkohol: Literal[1, 2]
    hari_kesehatan_fisik_buruk: int = Field(default=0, ge=0, le=30)
    pendidikan: Optional[Literal[1, 2, 3, 4, 5, 6]] = None
    pendapatan: Optional[Literal[1, 2, 3, 4, 5, 6, 7, 8]] = None
    # Skenario what-if
    skenario_berat_kg: Optional[float] = None
    skenario_aktivitas_fisik: Optional[Literal[1, 2]] = None
    skenario_merokok: Optional[Literal[1, 2, 3, 4]] = None
    skenario_alkohol: Optional[Literal[1, 2]] = None


class SimpanRiwayatInput(BaseModel):
    email: str
    usia: int = Field(gt=0, le=120)
    jenis_kelamin: Literal[1, 2]
    bmi: float = Field(gt=0, le=100)
    aktivitas_fisik: Literal[1, 2]
    merokok: Literal[1, 2, 3, 4]
    alkohol: Literal[1, 2]
    hari_kesehatan_fisik_buruk: int = 0
    risiko_diabetes: float
    risiko_jantung: float
    risiko_stroke: float
    risiko_ginjal: float


# FUNGSI KONVERSI INPUT dgn FORMAT BRFSS (sesuai encoding saat model dilatih)

def hitung_bmi(tinggi_cm: float, berat_kg: float) -> float:
    tinggi_m = tinggi_cm / 100
    return round(berat_kg / (tinggi_m ** 2), 1)


def buat_fitur_brfss(usia, tinggi_cm, berat_kg, merokok, alkohol,
                     aktivitas_fisik, hari_fisik_buruk, pendapatan):
    """
    Konversi nilai dari frontend ke encoding BRFSS yang dipakai model:
        _AGE80   : usia (max 80)
        WTKG3    : berat_kg × 100
        _BMI5    : BMI × 100
        HEIGHT3  : tinggi_cm (langsung)
        _SMOKER3 : merokok (sudah sama: 1/2/3/4)
        INCOME3  : pendapatan (1-8, default 5 kalau tidak diisi)
        _TOTINDA : aktivitas_fisik (sudah sama: 1=aktif, 2=tidak)
        PHYSHLTH : hari_fisik_buruk (0-30)
        DRNKANY6 : alkohol (sudah sama: 1=ya, 2=tidak)
    """
    bmi = hitung_bmi(tinggi_cm, berat_kg)
    return {
        "_AGE80":   min(usia, 80),
        "WTKG3":    round(berat_kg * 100),
        "_BMI5":    round(bmi * 100),
        "HEIGHT3":  tinggi_cm,
        "_SMOKER3": merokok,
        "INCOME3":  pendapatan if pendapatan is not None else 5,
        "_TOTINDA": aktivitas_fisik,
        "PHYSHLTH": hari_fisik_buruk,
        "DRNKANY6": alkohol,
    }


def prediksi_satu_model(nama_model: str, fitur_dict: dict) -> float:
    """
    Jalankan prediksi satu penyakit. Ambil hanya kolom yang dibutuhkan
    model tersebut, sesuai urutan feature_names_in_.
    Return: probabilitas positif (0-100).
    """
    model = MODELS[nama_model]
    kolom = list(model.feature_names_in_)
    X = np.array([[fitur_dict[k] for k in kolom]])
    prob = model.predict_proba(X)[0][1]
    return round(float(prob) * 100, 2)


# FUNGSI UTAMA: hitung risiko + breakdown kontribusi per faktor
# Breakdown dihitung dengan cara mengubah satu faktor ke kondisi "netral"
# dan melihat selisih probabilitasnya (poor-man's SHAP)

# Nilai "netral" / kondisi terbaik untuk tiap faktor modifiable
KONDISI_NETRAL = {
    "_SMOKER3": 4,   # tidak pernah merokok
    "_TOTINDA": 1,   # aktif fisik
    "DRNKANY6": 2,   # tidak minum alkohol
    # BMI netral = 22 (normal), direfleksikan ke WTKG3 & _BMI5
}

FAKTOR_MODIFIABLE = ["bmi", "aktivitas_fisik", "merokok", "alkohol"]

LABEL_FAKTOR = {
    "bmi": "BMI (berat & tinggi)",
    "aktivitas_fisik": "Aktivitas fisik",
    "merokok": "Kebiasaan merokok",
    "alkohol": "Konsumsi alkohol",
    "usia": "Usia",
    "base": "Risiko dasar",
}


def hitung_risiko_dengan_breakdown(usia, tinggi_cm, berat_kg, aktivitas_fisik,
                                    merokok, alkohol, hari_fisik=0, pendapatan=None):
    """
    Hitung probabilitas risiko 4 penyakit menggunakan model XGBoost asli,
    beserta breakdown kontribusi tiap faktor modifiable.
    """
    fitur = buat_fitur_brfss(usia, tinggi_cm, berat_kg, merokok, alkohol,
                              aktivitas_fisik, hari_fisik, pendapatan)
    bmi = hitung_bmi(tinggi_cm, berat_kg)

    # Probabilitas dengan kondisi saat ini
    prob_diabetes = prediksi_satu_model("Diabetes", fitur)
    prob_jantung  = prediksi_satu_model("Heart",    fitur)
    prob_stroke   = prediksi_satu_model("Stroke",   fitur)
    prob_ginjal   = prediksi_satu_model("Ginjal",   fitur)

    # Hitung kontribusi tiap faktor dengan cara "ubah ke kondisi netral"
    def kontribusi_faktor(nama_penyakit, ubahan: dict):
        """Selisih prob asli vs prob kalau faktor diubah ke kondisi netral."""
        fitur_baru = fitur.copy()
        fitur_baru.update(ubahan)
        prob_baru = prediksi_satu_model(nama_penyakit, fitur_baru)
        # Kontribusi positif = faktor ini menambah risiko
        prob_asli_map = {
            "Diabetes": prob_diabetes,
            "Heart":    prob_jantung,
            "Stroke":   prob_stroke,
            "Ginjal":   prob_ginjal,
        }
        return round(prob_asli_map[nama_penyakit] - prob_baru, 2)

    def get_kontribusi_bmi(nama_penyakit):

        berat_netral = round(22 * (tinggi_cm / 100) ** 2, 2) 
        
        fitur_baru = {
            **fitur,
            "WTKG3": round(berat_netral * 100),
            "_BMI5": 2200,
        }
        nilai = kontribusi_faktor(
            nama_penyakit,
            {
                "WTKG3": round(berat_netral * 100),
                "_BMI5": 2200,
            }
        )
        return abs(nilai)  
    
    def get_kontribusi_merokok(nama_penyakit):
        return kontribusi_faktor(nama_penyakit, {"_SMOKER3": 4})  # tidak pernah

    def get_kontribusi_aktivitas(nama_penyakit):
        return kontribusi_faktor(nama_penyakit, {"_TOTINDA": 1})  # aktif

    def get_kontribusi_alkohol(nama_penyakit):
        return kontribusi_faktor(nama_penyakit, {"DRNKANY6": 2})  
        return abs(nilai) # tidak minum

    # Susun breakdown per penyakit
    def build_breakdown(nama_penyakit, prob_asli):
        bd_raw = {}
        bd_raw["bmi"]             = get_kontribusi_bmi(nama_penyakit)
        bd_raw["merokok"]         = get_kontribusi_merokok(nama_penyakit)
        bd_raw["aktivitas_fisik"] = get_kontribusi_aktivitas(nama_penyakit)
        bd_raw["alkohol"]         = get_kontribusi_alkohol(nama_penyakit)

        # Clamp nilai negatif ke 0 untuk display breakdown.
        # Nilai negatif terjadi karena anomali korelasi di data BRFSS
        # (contoh: peminum alkohol dalam data cenderung punya income lebih tinggi,
        # sehingga model menangkap korelasi terbalik pada Ginjal).
        # Prediksi probabilitas utama tetap menggunakan model XGBoost asli (tidak diclamp).
        bd = {k: round(abs(v), 2) for k, v in bd_raw.items()}

        # "base" = sisa probabilitas setelah dikurangi semua kontribusi positif
        total_positif = sum(bd.values())
        bd["base"] = round(max(prob_asli - total_positif, 0), 2)
        return bd

    return {
        "bmi": bmi,
        "diabetes": {
            "probabilitas": prob_diabetes,
            "breakdown": build_breakdown("Diabetes", prob_diabetes),
        },
        "jantung": {
            "probabilitas": prob_jantung,
            "breakdown": build_breakdown("Heart", prob_jantung),
        },
        "stroke": {
            "probabilitas": prob_stroke,
            "breakdown": build_breakdown("Stroke", prob_stroke),
        },
        "ginjal": {
            "probabilitas": prob_ginjal,
            "breakdown": build_breakdown("Ginjal", prob_ginjal),
        },
    }


# ENDPOINT API

@app.get("/")
def root():
    return {"message": "NCD-Sim API aktif", "model": "XGBoost (Models_XG_Boost_Penyakit.pkl)"}


@app.post("/predict")
def predict(data: ProfilInput):
    """
    Prediksi risiko 4 penyakit + breakdown kontribusi faktor.
    Menggunakan model XGBoost asli dari file .pkl.
    """
    hasil_lengkap = hitung_risiko_dengan_breakdown(
        usia=data.usia,
        tinggi_cm=data.tinggi_cm,
        berat_kg=data.berat_kg,
        aktivitas_fisik=data.aktivitas_fisik,
        merokok=data.merokok,
        alkohol=data.alkohol,
        hari_fisik=data.hari_kesehatan_fisik_buruk,
        pendapatan=data.pendapatan,
    )

    bmi = hasil_lengkap.pop("bmi")

    # Hitung rata-rata kontribusi tiap faktor modifiable lintas 4 penyakit
    rata_kontribusi = {}
    for faktor in FAKTOR_MODIFIABLE:
        nilai = [
            hasil_lengkap[p]["breakdown"].get(faktor, 0)
            for p in hasil_lengkap
        ]
        rata_kontribusi[faktor] = round(sum(nilai) / len(nilai), 2)

    faktor_teratas = sorted(rata_kontribusi.items(), key=lambda x: x[1], reverse=True)

    return {
        "bmi": bmi,
        "risiko": hasil_lengkap,
        "faktor_modifiable_teratas": [
            {"faktor": f, "rata_kontribusi": v} for f, v in faktor_teratas
        ],
    }


@app.post("/simulate")
def simulate(data: SimulasiInput):
    """
    What-If Simulation: bandingkan risiko kondisi asli vs skenario yang diubah.
    """
    # Kondisi asli
    hasil_asli = hitung_risiko_dengan_breakdown(
        usia=data.usia,
        tinggi_cm=data.tinggi_cm,
        berat_kg=data.berat_kg,
        aktivitas_fisik=data.aktivitas_fisik,
        merokok=data.merokok,
        alkohol=data.alkohol,
        hari_fisik=data.hari_kesehatan_fisik_buruk,
        pendapatan=data.pendapatan,
    )
    bmi_asli = hasil_asli.pop("bmi")
    risiko_asli = {p: hasil_asli[p]["probabilitas"] for p in hasil_asli}

    # Kondisi skenario
    berat_baru    = data.skenario_berat_kg       if data.skenario_berat_kg       is not None else data.berat_kg
    aktivitas_baru = data.skenario_aktivitas_fisik if data.skenario_aktivitas_fisik is not None else data.aktivitas_fisik
    merokok_baru  = data.skenario_merokok        if data.skenario_merokok        is not None else data.merokok
    alkohol_baru  = data.skenario_alkohol        if data.skenario_alkohol        is not None else data.alkohol

    fitur_test = buat_fitur_brfss(
        usia=data.usia,
        tinggi_cm=data.tinggi_cm,
        berat_kg=berat_baru,
        merokok=merokok_baru,
        alkohol=alkohol_baru,
        aktivitas_fisik=aktivitas_baru,
        hari_fisik_buruk=data.hari_kesehatan_fisik_buruk,
        pendapatan=data.pendapatan,
    )
    hasil_baru = hitung_risiko_dengan_breakdown(
        usia=data.usia,
        tinggi_cm=data.tinggi_cm,
        berat_kg=berat_baru,
        aktivitas_fisik=aktivitas_baru,
        merokok=merokok_baru,
        alkohol=alkohol_baru,
        hari_fisik=data.hari_kesehatan_fisik_buruk,
        pendapatan=data.pendapatan,
    )
    bmi_baru = hasil_baru.pop("bmi")
    risiko_baru = {p: hasil_baru[p]["probabilitas"] for p in hasil_baru}

    selisih = {p: round(risiko_asli[p] - risiko_baru[p], 2) for p in risiko_asli}

    return {
        "sebelum": {"bmi": bmi_asli, "risiko": risiko_asli},
        "sesudah": {"bmi": bmi_baru, "risiko": risiko_baru},
        "selisih": selisih,
    }


@app.post("/ranking")
def ranking_intervensi(data: ProfilInput):
    """
    Ranking Intervensi Otomatis: coba beberapa skenario standar,
    urutkan dari yang paling berdampak terhadap total risiko 4 penyakit.
    """
    hasil_asli = hitung_risiko_dengan_breakdown(
        usia=data.usia,
        tinggi_cm=data.tinggi_cm,
        berat_kg=data.berat_kg,
        aktivitas_fisik=data.aktivitas_fisik,
        merokok=data.merokok,
        alkohol=data.alkohol,
        hari_fisik=data.hari_kesehatan_fisik_buruk,
        pendapatan=data.pendapatan,
    )
    hasil_asli.pop("bmi")
    risiko_awal = {p: hasil_asli[p]["probabilitas"] for p in hasil_asli}

    bmi_asli = hitung_bmi(data.tinggi_cm, data.berat_kg)

    def hitung_skenario(berat_kg=None, aktivitas_fisik=None, merokok=None, alkohol=None):
        berat     = berat_kg        if berat_kg        is not None else data.berat_kg
        aktivitas = aktivitas_fisik if aktivitas_fisik is not None else data.aktivitas_fisik
        rokok     = merokok         if merokok         is not None else data.merokok
        alk       = alkohol         if alkohol         is not None else data.alkohol
        h = hitung_risiko_dengan_breakdown(
            usia=data.usia,
            tinggi_cm=data.tinggi_cm,
            berat_kg=berat,
            aktivitas_fisik=aktivitas,
            merokok=rokok,
            alkohol=alk,
            hari_fisik=data.hari_kesehatan_fisik_buruk,
            pendapatan=data.pendapatan,
        )
        h.pop("bmi")
        return {p: h[p]["probabilitas"] for p in h}

    daftar_skenario = []

    if bmi_asli > 23:
        berat_target = round(23 * (data.tinggi_cm / 100) ** 2, 2)
        daftar_skenario.append(("Turunkan BMI ke rentang normal", hitung_skenario(berat_kg=berat_target)))

    if data.aktivitas_fisik == 2:
        daftar_skenario.append(("Mulai aktivitas fisik rutin", hitung_skenario(aktivitas_fisik=1)))

    if data.merokok in (1, 2):
        daftar_skenario.append(("Berhenti merokok", hitung_skenario(merokok=4)))

    if data.alkohol == 1:
        daftar_skenario.append(("Berhenti konsumsi alkohol", hitung_skenario(alkohol=2)))

    hasil_ranking = []
    for nama, risiko_baru in daftar_skenario:
        dampak_per_penyakit = {p: round(max(risiko_awal[p] - risiko_baru[p], 0), 2) for p in risiko_awal}
        dampak_total = round(sum(dampak_per_penyakit.values()), 2)
        hasil_ranking.append({
            "intervensi": nama,
            "risiko_baru": risiko_baru,
            "dampak_per_penyakit": dampak_per_penyakit,
            "dampak_total": dampak_total,
        })

    hasil_ranking.sort(key=lambda x: x["dampak_total"], reverse=True)

    return {
        "risiko_awal": risiko_awal,
        "ranking_intervensi": hasil_ranking,
    }


@app.post("/history")
def simpan_riwayat(data: SimpanRiwayatInput):
    """Simpan satu entri riwayat ke SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO history
        (timestamp, email, usia, jenis_kelamin, bmi, aktivitas_fisik, merokok, alkohol,
         hari_kesehatan_fisik_buruk,
         risiko_diabetes, risiko_jantung, risiko_stroke, risiko_ginjal)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        data.email,
        data.usia, data.jenis_kelamin, data.bmi, data.aktivitas_fisik,
        data.merokok, data.alkohol,
        data.hari_kesehatan_fisik_buruk,
        data.risiko_diabetes, data.risiko_jantung, data.risiko_stroke, data.risiko_ginjal,
    ))
    conn.commit()
    conn.close()
    return {"status": "tersimpan"}


@app.get("/history")
def ambil_riwayat(email: Optional[str] = None):
    """Ambil riwayat per email (atau semua kalau email kosong)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if email:
        cursor.execute("SELECT * FROM history WHERE email = ? ORDER BY timestamp ASC", (email,))
    else:
        cursor.execute("SELECT * FROM history ORDER BY timestamp ASC")
    rows = cursor.fetchall()
    print(rows)
    conn.close()

    return [
        {
            "id": row[0], "timestamp": row[1], "email": row[2],
            "usia": row[3], "jenis_kelamin": row[4], "bmi": row[5],
            "aktivitas_fisik": row[6], "merokok": row[7], "alkohol": row[8],
            "hari_kesehatan_fisik_buruk": row[9], "risiko_diabetes": row[10], "risiko_jantung": row[11],
            "risiko_stroke": row[12], "risiko_ginjal": row[13],
        }
        for row in rows
    ]

@app.get("/feature-importance")
def get_feature_importance():
    """
    Kembalikan feature importance untuk tiap penyakit.
    Diekstrak dari model saat training.
    """
    return {
        "diabetes": FEATURE_IMPORTANCE["Diabetes"],
        "jantung":  FEATURE_IMPORTANCE["Heart"],
        "stroke":   FEATURE_IMPORTANCE["Stroke"],
        "ginjal":   FEATURE_IMPORTANCE["Ginjal"],
    }


@app.get("/stats")
def get_stats():
    """
    Statistik penggunaan platform: jumlah pengguna & prediksi tersimpan.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM history")
    total_prediksi = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT email) FROM history")
    total_pengguna = cursor.fetchone()[0]
    conn.close()
    return {
        "total_prediksi": total_prediksi,
        "total_pengguna": total_pengguna,
    }

