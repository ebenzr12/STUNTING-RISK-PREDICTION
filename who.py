"""Fungsi bantu standar pertumbuhan WHO (tinggi/panjang badan menurut umur, TB/U).

Z-score = ((tinggi / M) ** L - 1) / (S * L), dengan L = 1 untuk indikator TB/U.
Ambang status mengikuti Permenkes No. 2 Tahun 2020:
    < -3 SD         : sangat pendek (severely stunted)
    -3 s.d. < -2 SD : pendek (stunted)
    -2 s.d. +3 SD   : normal
    > +3 SD         : tinggi
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# Label bawaan dataset -> tampilan di aplikasi
NAMA = {
    "severely stunted": "Sangat pendek (stunting berat)",
    "stunted": "Pendek (stunting)",
    "normal": "Normal",
    "tinggi": "Tinggi",
}
WARNA = {
    "severely stunted": "#C53030",
    "stunted": "#C77700",
    "normal": "#2F855A",
    "tinggi": "#2B6CB0",
}
JK_KODE = {"laki-laki": 0, "perempuan": 1}
BATAS_Z_WAJAR = 6  # WHO menandai |Z| > 6 pada TB/U sebagai nilai yang tidak wajar


def load_who() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "who_lhfa.csv")


def hitung_z(df: pd.DataFrame, who: pd.DataFrame) -> pd.Series:
    """df punya kolom: umur (bulan), jk ('laki-laki'/'perempuan'), tinggi (cm)."""
    m = df.reset_index(drop=True)[["umur", "jk", "tinggi"]].merge(who, on=["jk", "umur"], how="left")
    return ((m["tinggi"] / m["M"]) ** m["L"] - 1) / (m["S"] * m["L"])


def status_dari_z(z: pd.Series) -> pd.Series:
    z = pd.Series(z).reset_index(drop=True)
    s = np.select(
        [z < -3, z < -2, z <= 3],
        ["severely stunted", "stunted", "normal"],
        default="tinggi",
    )
    return pd.Series(np.where(z.isna(), None, s), dtype=object)


def kurva(who: pd.DataFrame, jk: str) -> pd.DataFrame:
    """Garis median, -3 SD, -2 SD, dan +3 SD per bulan untuk satu jenis kelamin."""
    t = who[who["jk"] == jk].sort_values("umur")
    out = pd.DataFrame({"umur": t["umur"].values})
    for nama, z in (("sd3n", -3), ("sd2n", -2), ("median", 0), ("sd3p", 3)):
        out[nama] = (t["M"] * (1 + t["S"] * z)).values  # L = 1
    return out


# ------------------------------------------------------------------ tabel & hasil
PETA_JK = {
    "l": "laki-laki", "lk": "laki-laki", "laki-laki": "laki-laki", "laki laki": "laki-laki",
    "laki": "laki-laki", "male": "laki-laki", "m": "laki-laki",
    "p": "perempuan", "pr": "perempuan", "perempuan": "perempuan", "wanita": "perempuan",
    "female": "perempuan", "f": "perempuan",
}


def _cari_kolom(kolom, kata_kunci):
    for c in kolom:
        if any(k in str(c).lower() for k in kata_kunci):
            return c
    return None


def siapkan_tabel(raw: pd.DataFrame) -> pd.DataFrame:
    """Rapikan tabel unggahan: kolom umur, jk, tinggi + kolom 'masalah' (kosong jika valid)."""
    umur_c = _cari_kolom(raw.columns, ["umur", "usia", "age"])
    jk_c = _cari_kolom(raw.columns, ["kelamin", "jenis", "jk", "sex", "gender"])
    tb_c = _cari_kolom(raw.columns, ["tinggi", "panjang", "height", "tb"])
    hilang = [n for n, c in (("umur", umur_c), ("jenis kelamin", jk_c), ("tinggi badan", tb_c)) if c is None]
    if hilang:
        raise ValueError("Kolom tidak ditemukan: " + ", ".join(hilang))

    angka = lambda s: pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
    std = pd.DataFrame(
        {
            "umur": np.floor(angka(raw[umur_c])),
            "jk": raw[jk_c].astype(str).str.strip().str.lower().map(PETA_JK),
            "tinggi": angka(raw[tb_c]),
        }
    ).reset_index(drop=True)

    masalah = np.select(
        [
            std["umur"].isna() | (std["umur"] < 0) | (std["umur"] > 60),
            std["jk"].isna(),
            std["tinggi"].isna() | (std["tinggi"] <= 0),
        ],
        ["Umur harus 0-60 bulan", "Jenis kelamin tidak dikenali", "Tinggi badan tidak valid"],
        default="",
    )
    std["masalah"] = masalah
    return std


def hitung_hasil(std: pd.DataFrame, who: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Tambahkan Z-score, status WHO, dan prediksi model. Baris bermasalah tidak diklasifikasi."""
    out = std.copy().reset_index(drop=True)
    n = len(out)
    out["z"] = np.nan
    out["status"] = pd.Series([None] * n, dtype=object)
    out["prediksi"] = pd.Series([None] * n, dtype=object)
    out["keyakinan"] = np.nan

    valid = out["masalah"] == ""
    if valid.any():
        sub = out.loc[valid, ["umur", "jk", "tinggi"]].copy()
        sub["umur"] = sub["umur"].astype(int)
        z = hitung_z(sub, who)
        z.index = sub.index
        out.loc[valid, "z"] = z

        tak_wajar = out["z"].abs() > BATAS_Z_WAJAR
        out.loc[tak_wajar, "masalah"] = "Nilai tidak wajar (Z di luar -6 s.d. +6), ukur ulang"

        ok = out["masalah"] == ""
        if ok.any():
            out.loc[ok, "status"] = status_dari_z(out.loc[ok, "z"]).values
            X = pd.DataFrame(
                {
                    "umur": out.loc[ok, "umur"].astype(int),
                    "jk_kode": out.loc[ok, "jk"].map(JK_KODE),
                    "tinggi": out.loc[ok, "tinggi"],
                }
            )[bundle["fitur"]]
            proba = bundle["model"].predict_proba(X)
            out.loc[ok, "prediksi"] = bundle["model"].classes_[proba.argmax(axis=1)]
            out.loc[ok, "keyakinan"] = proba.max(axis=1)
    return out


def latih_cepat() -> dict:
    """Cadangan bila model.pkl tidak bisa dimuat (misalnya beda versi scikit-learn):
    latih KNN dari data/data_balita.csv. Cukup beberapa detik."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(DATA_DIR / "data_balita.csv")
    df.columns = ["umur", "jk", "tinggi", "label"]
    df["jk"] = df["jk"].str.strip().str.lower()
    df["label"] = df["label"].str.strip().str.lower()
    df = df.dropna().drop_duplicates()
    df["jk_kode"] = df["jk"].map(JK_KODE)
    fitur = ["umur", "jk_kode", "tinggi"]
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance"))
    model.fit(df[fitur], df["label"])
    return {"model": model, "fitur": fitur, "tau": 0.8, "cadangan": True}
