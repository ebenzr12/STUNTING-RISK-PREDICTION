"""Latih & bandingkan Random Forest vs KNN untuk status stunting, simpan model terbaik.

Jalankan dari folder proyek:  python train_model.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import who as W

BASE = Path(__file__).parent
DATA_CSV = BASE / "data" / "data_balita.csv"
ARTIFACTS = BASE / "artifacts"
FITUR = ["umur", "jk_kode", "tinggi"]
TAU = 0.8  # ambang keyakinan zona abu-abu


def muat_data() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA_CSV)
    df.columns = ["umur", "jk", "tinggi", "label"]
    df["jk"] = df["jk"].str.strip().str.lower()
    df["label"] = df["label"].str.strip().str.lower()
    n_awal = len(df)
    df = df.dropna().drop_duplicates().reset_index(drop=True)  # duplikat dibuang SEBELUM split
    df["jk_kode"] = df["jk"].map(W.JK_KODE)
    info = {"baris_awal": n_awal, "baris_unik": len(df), "duplikat_dibuang": n_awal - len(df)}
    return df, info


def main() -> None:
    df, info = muat_data()
    print(f"Baris awal {info['baris_awal']:,} -> unik {info['baris_unik']:,} (duplikat dibuang {info['duplikat_dibuang']:,})")

    # Validasi label dataset terhadap Z-score WHO
    who = W.load_who()
    z = W.hitung_z(df, who)
    cocok = float((W.status_dari_z(z) == df["label"]).mean())
    tak_wajar = int((z.abs() > W.BATAS_Z_WAJAR).sum())
    print(f"Label dataset cocok dengan Z-score WHO: {cocok:.2%} | baris |Z| > 6: {tak_wajar:,}")

    X, y = df[FITUR], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # KNN: skala fitur lalu cari k terbaik
    knn = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(weights="distance"))]),
        {"knn__n_neighbors": [3, 5, 7, 11]},
        cv=3, scoring="f1_macro", n_jobs=-1,
    ).fit(X_tr, y_tr)
    kandidat = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "KNN": knn.best_estimator_,
    }
    print(f"KNN: k terbaik = {knn.best_params_['knn__n_neighbors']}")

    hasil = {}
    for nama, model in kandidat.items():
        sk = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        hasil[nama] = {
            "cv_akurasi": float(sk["test_accuracy"].mean()),
            "cv_akurasi_std": float(sk["test_accuracy"].std()),
            "cv_f1_macro": float(sk["test_f1_macro"].mean()),
            "uji_akurasi": float(accuracy_score(y_te, pred)),
            "uji_f1_macro": float(f1_score(y_te, pred, average="macro")),
        }
        print(f"\n=== {nama} ===")
        print(f"CV 5-fold  akurasi {hasil[nama]['cv_akurasi']:.2%} (+/- {hasil[nama]['cv_akurasi_std']:.2%}) | F1-macro {hasil[nama]['cv_f1_macro']:.2%}")
        print(f"Data uji   akurasi {hasil[nama]['uji_akurasi']:.2%} | F1-macro {hasil[nama]['uji_f1_macro']:.2%}")
        print(classification_report(y_te, pred, zero_division=0))

    terbaik = max(hasil, key=lambda n: hasil[n]["cv_f1_macro"])

    # Zona abu-abu: kasus dengan keyakinan model < TAU ditandai "ukur ulang" (model masih dilatih hanya pada data latih)
    best = kandidat[terbaik]
    proba = best.predict_proba(X_te)
    pred = best.classes_[proba.argmax(axis=1)]
    conf, benar = proba.max(axis=1), pred == y_te.values
    jawab = conf >= TAU
    zona = {
        "tau": TAU,
        "ditandai": float(1 - jawab.mean()),
        "akurasi_dijawab": float(benar[jawab].mean()),
        "kesalahan_tertangkap": float(((~benar) & (~jawab)).sum() / max((~benar).sum(), 1)),
    }
    print(f"Zona abu-abu (keyakinan < {TAU}): {zona['ditandai']:.1%} kasus ditandai | akurasi kasus dijawab {zona['akurasi_dijawab']:.2%} | "
          f"{zona['kesalahan_tertangkap']:.1%} kesalahan tertangkap")
    print(f"Model terbaik (F1-macro CV): {terbaik}")

    final = kandidat[terbaik].fit(X, y)  # latih ulang pada seluruh data unik
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"model": final, "fitur": FITUR, "tau": TAU}, ARTIFACTS / "model.pkl", compress=3)
    metrik = {
        "model_terbaik": terbaik,
        "hasil": hasil,
        "data": info,
        "cocok_dengan_who": cocok,
        "zona_abu": zona,
        "baris_z_tak_wajar": tak_wajar,
        "versi_sklearn": sklearn.__version__,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrik, indent=2), encoding="utf-8")
    ukuran = (ARTIFACTS / "model.pkl").stat().st_size / 1e6
    print(f"\nTersimpan di artifacts/ (model.pkl {ukuran:.1f} MB)")
    print(f"Versi scikit-learn: {sklearn.__version__}  <- samakan di requirements.txt")


if __name__ == "__main__":
    main()
