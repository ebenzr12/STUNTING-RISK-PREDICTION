from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core as K
import who as W

st.set_page_config(page_title="Stunting Risk Prediction", page_icon=":material/monitor_heart:", layout="wide")
TEAL = "#14838D"
st.markdown("""<style>
:root{--dark:#123F46;--teal:#14838D;--teal-h:#106F78;--ink:#17343A;--mute:#60777B;--line:rgba(18,63,70,.10)}
.stApp{background:radial-gradient(1000px 420px at 92% -8%,rgba(20,131,141,.12),transparent 70%),radial-gradient(700px 360px at -5% 105%,rgba(18,63,70,.06),transparent 70%),#EAF5F5}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1080px;padding-top:2.2rem;padding-bottom:3rem}
h1,h2,h3,h4{color:var(--ink);letter-spacing:-.01em}
h2{font-size:1.7rem}
/* sidebar */
section[data-testid="stSidebar"]{background:#DDEDEE;border-right:1px solid var(--line)}
.brand{padding:6px 4px 16px}
.brand-kicker{font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;color:var(--teal);font-weight:700}
.brand-title{font-size:1.3rem;line-height:1.15;font-weight:800;color:var(--dark);letter-spacing:.04em;margin-top:4px}
section[data-testid="stSidebar"] div[role="radiogroup"]{gap:4px}
section[data-testid="stSidebar"] label[data-baseweb="radio"]{padding:9px 14px;border-radius:10px;width:100%;margin:0;cursor:pointer}
section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child:not(:has(p)){display:none}
section[data-testid="stSidebar"] label[data-baseweb="radio"] p{color:#2C5057;font-weight:500;margin:0}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover{background:rgba(20,131,141,.12)}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked){background:#14838D}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) p{color:#fff;font-weight:700}
/* hero */
.hero{background:radial-gradient(520px 260px at 92% 15%,rgba(20,131,141,.45),transparent 70%),#123F46;border-radius:18px;padding:30px 34px;margin-bottom:16px}
.hero h1{color:#fff;font-size:2.2rem;margin:0 0 6px}
.hero p{color:#CFE5E7;font-size:1rem;max-width:620px;margin:0}
/* kartu */
.card,.m,.res,div[data-testid="stPlotlyChart"],div[data-testid="stExpander"]{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:0 1px 2px rgba(18,63,70,.04)}
.card{padding:16px 18px;height:100%}
.card .ct{margin:0 0 6px;color:var(--teal);font-weight:700;font-size:1.1rem}
.num{color:var(--teal);font-weight:700;font-size:.78rem;letter-spacing:.12em}
div[data-testid="stPlotlyChart"]{padding:6px;overflow:hidden}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin:6px 0 14px}
.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin:6px 0 14px}
.m{padding:14px 18px}
.ml{color:var(--mute);font-size:.85rem}.mv{color:var(--ink);font-size:1.6rem;font-weight:700;line-height:1.3}.ms{color:var(--mute);font-size:.8rem;min-height:1.2em}
.cap{color:var(--mute);font-size:.85rem;min-height:3.4em;margin-top:-4px}
.note{color:var(--mute);font-size:.85rem}
.res{border-width:2px;padding:16px 22px;margin-bottom:10px}
.flow{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
.step{background:#DDEDEE;color:var(--ink);border-radius:999px;padding:6px 14px;font-weight:600;font-size:.88rem}
/* tombol */
button[data-testid="stBaseButton-primary"],button[data-testid="stBaseButton-primaryFormSubmit"]{background:#14838D;border:1px solid #14838D;color:#fff;border-radius:10px;font-weight:600}
button[data-testid="stBaseButton-primary"]:hover,button[data-testid="stBaseButton-primaryFormSubmit"]:hover{background:#106F78;border-color:#106F78;color:#fff}
button[data-testid="stBaseButton-secondary"],button[data-testid="stBaseButton-secondaryFormSubmit"]{background:#fff;border:1px solid #B8D0D3;color:var(--ink);border-radius:10px;font-weight:600}
button[data-testid="stBaseButton-secondary"]:hover{border-color:#14838D;color:#14838D}
div[data-testid="stForm"]{background:#fff;border:1px solid var(--line);border-radius:16px}
</style>""", unsafe_allow_html=True)

pct = lambda x: f"{x * 100:.2f}%".replace(".", ",")


def card(title, body, num=None):
    n = f'<div class="num">{num}</div>' if num else ""
    return f'<div class="card">{n}<div class="ct">{title}</div><div>{body}</div></div>'


def cards(items):
    st.markdown('<div class="grid">' + "".join(card(*it) for it in items) + "</div>", unsafe_allow_html=True)


def mgrid(items):
    """Baris kartu angka dengan tinggi dan lebar sama."""
    html = ""
    for it in items:
        l, v, sub = (list(it) + [""])[:3]
        html += f'<div class="m"><div class="ml">{l}</div><div class="mv">{v}</div><div class="ms">{sub}</div></div>'
    st.markdown(f'<div class="mgrid">{html}</div>', unsafe_allow_html=True)


def sym(f, h=320):
    """Ukuran, margin, dan legenda seragam untuk grafik yang berdampingan."""
    f.update_layout(height=h, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return f


def panel(f, caption):
    st.plotly_chart(sym(f), width="stretch")
    st.markdown(f'<div class="cap">{caption}</div>', unsafe_allow_html=True)


def flow(steps):
    st.markdown('<div class="flow">' + " → ".join(f'<span class="step">{s}</span>' for s in steps) + "</div>", unsafe_allow_html=True)


def header(t, sub):
    st.markdown(f"## {t}")
    st.caption(sub)


def goto(p):
    st.session_state.page = p


def set_contoh(jk, umur, tb):
    st.session_state.update(p_jk=jk, p_umur=int(umur), p_tb=float(tb))


def load_row():
    df, _ = K.load_df()
    r = df.sample(1).iloc[0]
    set_contoh(r["jk"], r["umur"], round(r["tinggi"], 1))


# ---------------------------------------------------------------- Home
def home():
    st.markdown('<div class="hero"><h1>Stunting Risk Prediction</h1><p>Machine Learning-Based Early Risk Detection for Child Growth. '
                'Memanfaatkan data pertumbuhan anak dan standar WHO untuk membantu skrining awal risiko stunting.</p></div>', unsafe_allow_html=True)
    a, b, _ = st.columns([1.3, 1.3, 5])
    a.button("Mulai Prediksi", type="primary", on_click=goto, args=("Prediction",), width="stretch")
    b.button("Eksplorasi Data", on_click=goto, args=("Data Explorer",), width="stretch")
    cards([("Machine Learning", "Model KNN/Random Forest dilatih dari data umur, jenis kelamin, dan tinggi badan."),
           ("Data-Driven Prediction", "Status utama dihitung dari Z-score WHO, model menjadi pembanding."),
           ("Early Risk Detection", "Kasus dengan keyakinan rendah ditandai agar diukur ulang.")])
    st.markdown("#### Alur sistem")
    flow(["Child Data", "Machine Learning", "Risk Prediction"])
    st.info("Alat bantu skrining berbasis data, bukan alat diagnosis medis.")


# ---------------------------------------------------------------- Understanding
def paham():
    header("Understanding Stunting", "Apa itu stunting, kenapa penting, dan apa tujuan sistem ini")
    t1, t2, t3 = st.tabs(["Apa itu stunting?", "Mengapa prediksi?", "Tujuan sistem"])
    with t1:
        cards([("Pertumbuhan", "Stunting adalah kondisi tinggi badan anak yang terlalu rendah untuk usianya, terjadi akibat hambatan pertumbuhan jangka panjang."),
               ("Cara menilai", "Tinggi badan dibandingkan dengan standar WHO. Z-score di bawah -2 SD disebut pendek, di bawah -3 SD sangat pendek (Permenkes No. 2 Tahun 2020)."),
               ("Dampak", "Stunting dapat berkaitan dengan perkembangan fisik dan kognitif, sehingga deteksi dini penting.")])
        st.warning("Sistem ini alat bantu prediksi berbasis data dan tidak menggantikan pemeriksaan tenaga kesehatan.")
    with t2:
        st.markdown("1. Data pertumbuhan anak memuat pola tertentu.\n2. Machine Learning mempelajari pola itu dari data historis.\n"
                    "3. Model memperkirakan status untuk data baru.\n4. Hasilnya menjadi informasi awal untuk skrining.")
        flow(["Data Anak", "Preprocessing", "Model ML", "Risk Prediction", "Interpretation"])
    with t3:
        cards([("Early Detection", "Membantu menandai anak yang tinggi badannya di bawah standar.", "01"),
               ("Data Analysis", "Menelusuri hubungan umur, jenis kelamin, dan tinggi badan.", "02"),
               ("Machine Learning", "Menerapkan klasifikasi dan membandingkan dengan Z-score WHO.", "03"),
               ("Explainable Result", "Menunjukkan posisi anak di kurva dan fitur yang paling berpengaruh.", "04")])


# ---------------------------------------------------------------- Data Explorer
def eksplorasi():
    header("Data Explorer", "Dataset, contoh data, EDA, dan korelasi. Semua angka dihitung dari data asli.")
    df, info = K.load_df()
    m = K.load_metrics()
    t1, t2, t3, t4 = st.tabs(["Dataset", "Contoh data", "EDA", "Korelasi"])
    with t1:
        mgrid([("Baris awal", f"{info['baris_awal']:,}"), ("Baris unik (dipakai)", f"{len(df):,}"), ("Jumlah fitur", "3"), ("Missing values", info["missing"])])
        st.table(pd.DataFrame({"Fitur": ["Umur", "Jenis kelamin", "Tinggi badan", "Status gizi (target)"],
                               "Deskripsi": ["Umur dalam bulan (0-60)", "Laki-laki / perempuan", "Panjang/tinggi badan (cm)", "Severely stunted, stunted, normal, tinggi"],
                               "Tipe": ["Numerik", "Kategorikal", "Numerik", "Kategorikal"]}))
        st.caption(f"{m['data']['duplikat_dibuang']:,} baris duplikat dibuang sebelum pembagian data latih dan uji.")
    with t2:
        st.dataframe(df[["umur", "jk", "tinggi", "label"]].head(10).rename(columns={"umur": "Umur (bln)", "jk": "Jenis kelamin", "tinggi": "Tinggi (cm)", "label": "Status"}), width="stretch")
        st.caption("Contoh data membantu memahami format yang dipakai model.")
        st.button("Load Example Data (baris acak)", on_click=load_row)
        if "p_tb" in st.session_state:
            st.success(f"Dimuat ke form prediksi: {st.session_state.p_jk}, {st.session_state.p_umur} bulan, {st.session_state.p_tb} cm. Buka halaman Prediction.")
    with t3:
        vc = df["label"].value_counts()
        top = vc.index[0]
        med = df.groupby("label")["tinggi"].median()
        g = df["jk"].value_counts()
        d2 = df.assign(Status=df["label"].map(W.NAMA))
        a, b = st.columns(2)
        with a:
            panel(px.bar(vc.rename(index=W.NAMA).reset_index(), x="label", y="count", labels={"label": "", "count": "Jumlah"}, color_discrete_sequence=[TEAL]),
                  f"Kelas terbanyak adalah {W.NAMA[top]} ({vc.iloc[0] / len(df):.1%} dari data); kelas paling sedikit {W.NAMA[vc.index[-1]]} ({vc.iloc[-1] / len(df):.1%}).")
        with b:
            panel(px.histogram(df, x="umur", nbins=30, labels={"umur": "Umur (bulan)"}, color_discrete_sequence=[TEAL]),
                  f"Umur rata-rata {df['umur'].mean():.1f} bulan (median {df['umur'].median():.0f}).")
        a, b = st.columns(2)
        with a:
            panel(px.box(d2, x="Status", y="tinggi", color="Status", labels={"tinggi": "Tinggi (cm)"}, color_discrete_sequence=[W.WARNA[k] for k in W.NAMA]),
                  "Median tinggi per kelas: " + ", ".join(f"{W.NAMA[k]} {v:.1f} cm" for k, v in med.items()) + ".")
        with b:
            panel(px.pie(g.reset_index(), names="jk", values="count", hole=0.5, color_discrete_sequence=[TEAL, "#E8A33D"]).update_traces(textinfo="label+percent"),
                  "Proporsi: " + ", ".join(f"{k} {v / len(df):.1%}" for k, v in g.items()) + ".")
        st.markdown("**Mengapa EDA?** Untuk melihat distribusi, data hilang, outlier, ketidakseimbangan kelas, dan menentukan preprocessing sebelum modeling.")
        flow(["Raw Data", "EDA", "Data Preparation", "Modeling"])
    with t4:
        cm = df[["umur", "jk_kode", "tinggi"]].rename(columns={"jk_kode": "jenis kelamin (0=L, 1=P)"}).corr().round(2)
        st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Teal", zmin=-1, zmax=1), width="stretch")
        st.caption(f"Korelasi umur dan tinggi badan: {cm.iloc[0, 2]:.2f}. Hanya ada 3 fitur, sehingga heatmap ini kecil.")


# ---------------------------------------------------------------- How it works
def cara_kerja():
    header("How It Works", "Preprocessing, model, evaluasi, dan pipeline")
    m = K.load_metrics()
    t1, t2, t3, t4 = st.tabs(["Preprocessing", "Machine Learning", "Evaluation", "Pipeline"])
    with t1:
        st.markdown(f"**1. Cleaning:** buang data kosong dan {m['data']['duplikat_dibuang']:,} baris duplikat (sisa {m['data']['baris_unik']:,}).  \n"
                    "**2. Encoding:** jenis kelamin menjadi angka (laki-laki = 0, perempuan = 1).  \n"
                    "**3. Seleksi fitur:** umur, jenis kelamin, tinggi badan.  \n"
                    "**4. Scaling:** StandardScaler untuk KNN (tidak diperlukan Random Forest).  \n"
                    "**5. Train-test split:** 80% latih, 20% uji, stratified, random_state 42.  \n"
                    f"**6. Validasi label:** label dataset dicek terhadap Z-score WHO, cocok {pct(m['cocok_dengan_who'])}.")
    with t2:
        rows = [{"Model": n, "Akurasi CV": pct(v["cv_akurasi"]), "F1-macro CV": pct(v["cv_f1_macro"]), "Akurasi uji": pct(v["uji_akurasi"]), "F1-macro uji": pct(v["uji_f1_macro"])} for n, v in m["hasil"].items()]
        st.table(pd.DataFrame(rows))
        st.markdown(f"**Why {m['model_terbaik']}?** Dipilih karena F1-macro cross-validation tertinggi, tetapi selisihnya dengan model lain sangat tipis. "
                    "KNN mencari anak-anak dengan umur, jenis kelamin, dan tinggi paling mirip, lalu memilih status yang paling banyak di antara mereka. "
                    "Kekurangannya: lambat pada data sangat besar dan sensitif pada skala fitur (karena itu fitur distandarkan).")
        st.caption("Random Forest: kelebihan ada pada feature importance bawaan dan ketahanan terhadap outlier.")
    with t3:
        labs, cmx, rep = K.eval_holdout()
        st.plotly_chart(px.imshow(cmx, x=[W.NAMA[l] for l in labs], y=[W.NAMA[l] for l in labs], text_auto=True, color_continuous_scale="Teal", labels={"x": "Prediksi", "y": "Sebenarnya"}), width="stretch")
        st.caption("Confusion matrix dari data uji 20% memakai salinan model yang dilatih hanya pada 80% data latih.")
        st.dataframe(pd.DataFrame(rep).T.loc[labs + ["macro avg"], ["precision", "recall", "f1-score", "support"]].round(3), width="stretch")
        st.markdown("**Accuracy:** proporsi prediksi yang benar. **Precision:** seberapa tepat prediksi per kelas. **Recall:** seberapa banyak kasus kelas itu yang berhasil ditemukan. "
                    "**F1:** keseimbangan precision dan recall. Untuk skrining, recall kelas stunting penting agar kasus berisiko tidak terlewat.")
    with t4:
        for t, d in [("Dataset", "Data balita dari Kaggle (umur, jenis kelamin, tinggi, status)."), ("Cleaning", "Buang kosong dan duplikat sebelum split."),
                     ("EDA", "Periksa distribusi, kelas, outlier."), ("Preprocessing", "Encoding jenis kelamin, scaling untuk KNN."),
                     ("Train/Test Split", "80/20 stratified."), ("Training", "Bandingkan Random Forest dan KNN dengan CV 5-fold."),
                     ("Evaluation", "Akurasi, F1-macro, confusion matrix."), ("Prediction", "Z-score WHO sebagai rujukan, model sebagai pembanding."),
                     ("Interpretation", "Kurva WHO, feature importance, what-if.")]:
            with st.expander(t):
                st.write(d)


# ---------------------------------------------------------------- Prediction
def prediksi():
    header("Prediction", "Masukkan data anak, lihat hasil, pahami alasannya, lalu coba what-if")
    st.session_state.setdefault("hist", [])
    st.session_state.setdefault("p_jk", "laki-laki")
    st.session_state.setdefault("p_umur", 18)
    st.session_state.setdefault("p_tb", 78.0)
    t1, t2, t3 = st.tabs(["Prediksi", "What-if", "Riwayat"])
    with t1:
        st.markdown("**Try an Example** (ilustrasi, bukan klaim medis)")
        a, b, c = st.columns(3)
        a.button("Contoh 1: risiko rendah", on_click=set_contoh, args=("laki-laki", 24, 88.0), width="stretch")
        b.button("Contoh 2: dekat batas", on_click=set_contoh, args=("laki-laki", 24, 81.5), width="stretch")
        c.button("Contoh 3: risiko lebih tinggi", on_click=set_contoh, args=("perempuan", 30, 80.0), width="stretch")
        with st.form("f"):
            x, y, z = st.columns(3)
            jk = x.selectbox("Jenis kelamin", ["laki-laki", "perempuan"], key="p_jk", help="Standar WHO berbeda untuk laki-laki dan perempuan.")
            umur = y.number_input("Umur (bulan)", 0, 60, key="p_umur", help="Usia anak dalam bulan, 0-60. Menentukan tinggi yang diharapkan.")
            tb = z.number_input("Tinggi/panjang badan (cm)", 40.0, 130.0, step=0.1, format="%.1f", key="p_tb", help="Panjang (berbaring, di bawah 24 bulan) atau tinggi (berdiri).")
            go_ = st.form_submit_button("Predict Risk", type="primary")
        with st.expander("Why is each input needed?"):
            st.markdown("- **Umur:** tinggi normal berubah seiring umur.\n- **Jenis kelamin:** kurva pertumbuhan WHO terpisah untuk laki-laki dan perempuan.\n- **Tinggi:** ukuran utama pertumbuhan linear anak.")
        if go_:
            r, _ = K.predict_one(jk, umur, tb)
            if not r["masalah"]:
                st.session_state.hist.append({"Waktu": datetime.now().strftime("%H:%M:%S"), "Input": f"{jk}, {umur} bln, {tb} cm", "Status WHO": W.NAMA[r["status"]], "Prediksi model": W.NAMA[r["prediksi"]], "Keyakinan": f"{r['keyakinan']:.0%}"})
            st.session_state.last = (jk, umur, tb)
        if "last" in st.session_state:
            hasil(*st.session_state.last)
    with t2:
        whatif()
    with t3:
        if st.session_state.hist:
            st.dataframe(pd.DataFrame(st.session_state.hist), width="stretch")
        else:
            st.info("Belum ada prediksi pada sesi ini.")
        st.button("Clear History", on_click=lambda: st.session_state.update(hist=[]))
        st.caption("Riwayat hanya disimpan selama sesi dan tidak dikirim ke database.")


KELAS = ["severely stunted", "stunted", "normal", "tinggi"]
SINGKAT = ["Sangat pendek", "Pendek", "Normal", "Tinggi"]
TAU_TINGGI = 0.95  # batas tampilan "High confidence" (batas zona abu-abu diambil dari model)


def konteks_who(jk, umur, tb, z):
    k = W.kurva(K.load_who(), jk)
    med = float(k.loc[k["umur"] == umur, "median"].iloc[0])
    pos, col = ("Di bawah -3 SD", "#C53030") if z < -3 else ("Di bawah -2 SD", "#C77700") if z < -2 else \
        ("Dalam rentang rujukan (-2 s.d. +3 SD)", "#2F855A") if z <= 3 else ("Di atas +3 SD", "#2B6CB0")
    mgrid([("Umur", f"{umur} bulan"), ("Tinggi", f"{tb} cm"), ("Median WHO", f"{med:.1f} cm", f"selisih {tb - med:+.1f} cm"), ("Z-score", f"{z:.2f}")])
    st.markdown(f"<span style='color:{col};font-weight:700'>● {pos}</span>", unsafe_allow_html=True)
    st.caption("Sistem membandingkan karakteristik pertumbuhan anak dengan standar WHO yang dipakai saat model dikembangkan. Ini bukan diagnosis.")
    f = go.Figure()
    for n, nm, c_, dsh in (("sd3p", "+3 SD", "#2B6CB0", "dot"), ("median", "Median", "#2F855A", "solid"), ("sd2n", "-2 SD", "#C77700", "dash"), ("sd3n", "-3 SD", "#C53030", "dot")):
        f.add_scatter(x=k["umur"], y=k[n], name=nm, line=dict(color=c_, dash=dsh))
    f.add_scatter(x=[umur], y=[tb], mode="markers", name="Anak", marker=dict(size=14, color=col, line=dict(width=2, color="#17343A")))
    f.update_layout(xaxis_title="Umur (bulan)", yaxis_title="Tinggi (cm)", height=340, margin=dict(t=10))
    with st.expander("Lihat posisi di kurva WHO"):
        st.plotly_chart(f, width="stretch")


def skala(pred, status):
    f = go.Figure()
    f.add_scatter(x=[0, 1, 2, 3], y=[0] * 4, mode="lines+markers+text", text=SINGKAT, textposition="bottom center", showlegend=False,
                  line=dict(color="#B8CBD1", width=5), marker=dict(size=10, color=[W.WARNA[k] for k in KELAS]))
    f.add_scatter(x=[KELAS.index(status)], y=[0], mode="markers", name="Rujukan WHO", marker=dict(size=28, symbol="circle-open", line=dict(width=3, color="#17343A")))
    f.add_scatter(x=[KELAS.index(pred)], y=[0], mode="markers+text", name="Prediksi model", text=["Prediksi"], textposition="top center",
                  marker=dict(size=18, color=W.WARNA[pred]))
    f.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10), yaxis=dict(visible=False, range=[-1, 1]), xaxis=dict(visible=False, range=[-0.4, 3.4]),
                    legend=dict(orientation="h", y=1.15))
    return f


def tingkat(c, tau):
    if c < tau:
        return "Uncertain / Borderline", "#C77700"
    return ("Moderate confidence", "#14838D") if c < TAU_TINGGI else ("High confidence", "#2F855A")


def hasil(jk, umur, tb):
    r, b = K.predict_one(jk, umur, tb)
    if r["masalah"]:
        st.error(r["masalah"])
        return
    tau = b.get("tau", 0.8)
    flow(["User Input", "Growth Context", "Model Analysis", "Prediction", "Confidence", "Explanation", "Sensitivity"])
    st.markdown("### 1. Growth Reference")
    konteks_who(jk, umur, tb, r["z"])

    st.markdown("### 2. Predicted Growth Status")
    unc = r["keyakinan"] < tau
    lvl, lcol = tingkat(r["keyakinan"], tau)
    col = W.WARNA[r["prediksi"]]
    if unc:
        st.markdown(f'<div class="res" style="border-color:#C77700"><div class="note">Prediksi</div><h2 style="color:#C77700;margin:2px 0">UNCERTAIN</h2>'
                    'Input berada di wilayah prediksi yang borderline. Pertimbangkan meninjau ulang data input atau melakukan penilaian tambahan (ukur ulang).</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="res" style="border-color:{col}"><div class="note">Predicted growth status (model)</div><h2 style="color:{col};margin:2px 0">{W.NAMA[r["prediksi"]]}</h2></div>', unsafe_allow_html=True)
    st.plotly_chart(skala(r["prediksi"], r["status"]), width="stretch")
    st.caption("Status pertumbuhan berurutan, dari sangat pendek sampai tinggi. Lingkaran bertepi = status menurut Z-score WHO, titik terisi = prediksi model.")
    if r["prediksi"] != r["status"]:
        st.info(f"Model ({W.NAMA[r['prediksi']]}) berbeda dari Z-score WHO ({W.NAMA[r['status']]}). Rujukan utama tetap Z-score WHO.")

    st.markdown("### 3. Confidence")
    pk = K.proba_kelas(jk, umur, tb).sort_values(ascending=False)
    st.markdown(f'<b style="color:{lcol}">{lvl}</b> · keyakinan model <b>{r["keyakinan"]:.0%}</b>'
                f'<div style="background:#E6EEF0;border-radius:8px;height:12px;margin:8px 0"><div style="width:{r["keyakinan"] * 100:.0f}%;background:{lcol};height:12px;border-radius:8px"></div></div>',
                unsafe_allow_html=True)
    st.caption("Peluang per kelas: " + ", ".join(f"{W.NAMA[k]} {v:.0%}" for k, v in pk.items() if v > 0) + f". Zona abu-abu: keyakinan di bawah {tau}. Batas 95% untuk label High hanya pengelompokan tampilan.")

    st.markdown("### 4. Explanation")
    imp = K.importance()
    st.plotly_chart(px.bar(imp, orientation="h", labels={"value": "Penurunan akurasi saat fitur diacak", "index": ""}, color_discrete_sequence=[TEAL]).update_layout(showlegend=False, height=240), width="stretch")
    st.caption("Permutation importance menunjukkan fitur yang paling memengaruhi keputusan model secara keseluruhan, bukan penyebab langsung stunting.")

    st.markdown("### 5. Measurement Sensitivity")
    sensitivitas(jk, umur, tb, r["prediksi"])


def sensitivitas(jk, umur, tb, pred0):
    st.caption("Jelajahi bagaimana perubahan kecil pada pengukuran dapat memengaruhi prediksi model.")
    step = st.radio("Variasi pengukuran (cm)", [0.5, 1.0, 2.0], index=1, horizontal=True, key="sens_step")
    rows, preds = [], []
    for lab, t in (("Skenario -", tb - step), ("Asli", tb), ("Skenario +", tb + step)):
        t = round(t, 1)
        r, _ = K.predict_one(jk, umur, t)
        if r["masalah"]:
            continue
        preds.append(r["prediksi"])
        rows.append({"Skenario": lab, "Tinggi (cm)": t, "Prediksi model": W.NAMA[r["prediksi"]], "Keyakinan": f"{r['keyakinan']:.0%}", "Status WHO": W.NAMA[r["status"]]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if len(set(preds)) <= 1:
        st.success(f"Prediction Stability: **High**. Prediksi tetap {W.NAMA[pred0]} pada variasi ±{step} cm.")
    else:
        st.warning("Prediction changed. Perubahan kecil pada tinggi badan menghasilkan prediksi berbeda, artinya model sensitif terhadap pengukuran ini.")
    st.caption("Ini analisis sensitivitas, bukan penilaian akurasi. Pada uji di notebook, galat ukur 1 cm menurunkan akurasi dari 0,990 menjadi sekitar 0,921 untuk semua pendekatan, termasuk aturan WHO murni.")


def whatif():
    if "last" not in st.session_state:
        st.info("Lakukan satu prediksi dulu di tab Prediksi.")
        return
    jk, umur, tb = st.session_state.last
    st.markdown(f"Dasar: {jk}, {umur} bulan, {tb} cm. Ubah tinggi badan untuk melihat perubahan status.")
    d = st.slider("Perubahan tinggi (cm)", -10.0, 10.0, 0.0, 0.5)
    t = np.round(tb + np.arange(-10, 10.1, 1.0), 1)
    sw = pd.DataFrame({"umur": umur, "jk": jk, "tinggi": t})
    z = W.hitung_z(sw, K.load_who())
    f = go.Figure(go.Scatter(x=t, y=z, mode="lines+markers", line=dict(color=TEAL)))
    for lv, c_ in ((-3, "#C53030"), (-2, "#C77700"), (3, "#2B6CB0")):
        f.add_hline(y=lv, line_dash="dash", line_color=c_, annotation_text=f"{lv} SD")
    f.add_vline(x=tb + d, line_color="#17343A")
    f.update_layout(xaxis_title="Tinggi (cm)", yaxis_title="Z-score", height=340, margin=dict(t=20))
    st.plotly_chart(f, width="stretch")
    r, _ = K.predict_one(jk, umur, round(tb + d, 1))
    if r["masalah"]:
        st.error(r["masalah"])
    else:
        mgrid([(f"Tinggi {tb + d:.1f} cm", W.NAMA[r["status"]], f"Z = {r['z']:.2f}")])
        st.caption(f"Model: {W.NAMA[r['prediksi']]} (keyakinan {r['keyakinan']:.0%}).")


# ---------------------------------------------------------------- Model
def model_page():
    header("Model", "Informasi, metrik, dan keterbatasan")
    m, (df, _), b = K.load_metrics(), K.load_df(), K.load_bundle()
    t1, t2 = st.tabs(["Model information", "Limitations"])
    with t1:
        k = None
        try:
            k = b["model"].named_steps["knn"].n_neighbors
        except Exception:
            pass
        st.table(pd.DataFrame({"Item": ["Algoritma", "Jumlah sampel (unik)", "Fitur", "Train/test split", "Zona abu-abu (tau)", "Versi scikit-learn (saat training)", "Tetangga (k)"],
                               "Nilai": [str(v) for v in (m["model_terbaik"], f"{len(df):,}", "umur, jenis kelamin, tinggi", "80% / 20%, stratified", b.get("tau", 0.8), m["versi_sklearn"], k if k else "-")]}))
        z = m["zona_abu"]
        st.caption(f"Pada data uji, {z['ditandai']:.1%} kasus ditandai zona abu-abu; kasus yang dijawab model akurasinya {pct(z['akurasi_dijawab'])}.")
        if b.get("cadangan"):
            st.warning("model.pkl tidak cocok dengan versi scikit-learn di server, model dilatih ulang otomatis.")
    with t2:
        st.markdown("- Dataset tampak hasil simulasi dari standar WHO, bukan pengukuran lapangan, dan "
                    f"{m['baris_z_tak_wajar']:,} baris punya |Z| > 6.\n- Akurasi tinggi berarti model mempelajari ulang aturan Z-score, jadi Z-score WHO dijadikan rujukan dan model hanya pembanding.\n"
                    "- Selisih ukur berbaring vs berdiri (sekitar 0,7 cm di usia 24 bulan) tidak dikoreksi.\n- Dataset belum tentu mewakili seluruh populasi.\n- Hasil bukan diagnosis medis.")


def about():
    header("About", "Proyek dan disclaimer")
    st.markdown("Proyek Machine Learning untuk edukasi dan portofolio. Sumber: WHO Child Growth Standards (TB/U 0-5 tahun), Permenkes No. 2 Tahun 2020, dan dataset *Stunting Toddler (Balita) Detection (121K rows)* (Pradana, Kaggle).")
    st.error("**Important Notice:** hasil prediksi adalah estimasi berdasarkan pola pada dataset dan bukan diagnosis medis. Untuk penilaian kondisi kesehatan anak, konsultasikan dengan tenaga kesehatan yang kompeten.")


PAGES = {"Home": home, "Understanding Stunting": paham, "Data Explorer": eksplorasi, "How It Works": cara_kerja, "Prediction": prediksi, "Model": model_page, "About": about}
st.session_state.setdefault("page", "Home")
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-kicker">Child growth screening</div><div class="brand-title">STUNTING<br>RISK PREDICTION</div></div>', unsafe_allow_html=True)
    st.radio("Navigasi", list(PAGES), key="page", label_visibility="collapsed")
    st.caption("Alat bantu skrining, bukan diagnosis.")
PAGES[st.session_state.page]()
