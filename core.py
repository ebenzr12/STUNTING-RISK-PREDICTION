"""Fungsi data, model, dan penjelasan. Semua angka dihitung dari dataset/model asli."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import who as W

BASE = Path(__file__).parent
ART = BASE / "artifacts"
FITUR = ["umur", "jk_kode", "tinggi"]
URUT = ["severely stunted", "stunted", "normal", "tinggi"]


@st.cache_data
def load_df():
    raw = pd.read_csv(BASE / "data" / "data_balita.csv")
    raw.columns = ["umur", "jk", "tinggi", "label"]
    info = {"baris_awal": len(raw), "missing": int(raw.isna().sum().sum())}
    df = raw.copy()
    df["jk"] = df["jk"].str.strip().str.lower()
    df["label"] = df["label"].str.strip().str.lower()
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["jk_kode"] = df["jk"].map(W.JK_KODE)
    return df, info


@st.cache_data
def load_metrics():
    return json.loads((ART / "metrics.json").read_text(encoding="utf-8"))


@st.cache_resource
def load_bundle():
    try:
        return joblib.load(ART / "model.pkl")
    except Exception:  # beda versi scikit-learn: latih ulang cepat
        return W.latih_cepat()


@st.cache_resource
def load_who():
    return W.load_who()


@st.cache_data
def eval_holdout():
    """Latih salinan model pada 80% data lalu uji pada 20% (bukan model final yang sudah melihat semua data)."""
    df, _ = load_df()
    Xtr, Xte, ytr, yte = train_test_split(df[FITUR], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    m = clone(load_bundle()["model"]).fit(Xtr, ytr)
    pred = m.predict(Xte)
    labs = [c for c in URUT if c in set(df["label"])]
    rep = classification_report(yte, pred, labels=labs, output_dict=True, zero_division=0)
    return labs, confusion_matrix(yte, pred, labels=labs), rep


@st.cache_data
def importance():
    """Permutation importance: penurunan akurasi saat satu fitur diacak."""
    df, _ = load_df()
    s = df.sample(3000, random_state=0)
    r = permutation_importance(load_bundle()["model"], s[FITUR], s["label"], n_repeats=3, random_state=0, scoring="accuracy")
    nama = {"umur": "Umur", "jk_kode": "Jenis kelamin", "tinggi": "Tinggi badan"}
    return pd.Series(r.importances_mean, index=[nama[f] for f in FITUR]).sort_values()


def predict_one(jk, umur, tinggi):
    std = pd.DataFrame({"umur": [umur], "jk": [jk], "tinggi": [tinggi], "masalah": [""]})
    b = load_bundle()
    return W.hitung_hasil(std, load_who(), b).iloc[0], b


def proba_kelas(jk, umur, tinggi):
    """Peluang tiap kelas dari model (predict_proba)."""
    b = load_bundle()
    X = pd.DataFrame({"umur": [umur], "jk_kode": [W.JK_KODE[jk]], "tinggi": [tinggi]})[b["fitur"]]
    return pd.Series(b["model"].predict_proba(X)[0], index=b["model"].classes_)
