import streamlit as st
import pandas as pd
import numpy as np
import os

# -------- Helpers --------
def try_kaggle_download(url: str, data_dir: str) -> str | None:
    """
    Tries to download a Kaggle dataset using opendatasets.
    Returns a path to a CSV file if found, else None.
    """
    try:
        import opendatasets as od  # pip install opendatasets
        if not os.path.exists(data_dir):
            st.info("⏬ Kaggle'dan veri indiriliyor... (ilk çalıştırmada biraz sürebilir)")
            od.download(url, data_dir=data_dir)
        # Walk the directory tree to find the first CSV
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(".csv"):
                    return os.path.join(root, f)
        return None
    except Exception as e:
        st.warning(f"Kaggle indirme denemesi başarısız: {e}")
        return None

def preprocess_kepler(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kepler verisi için temel ön işleme:
      - koi_disposition -> label (CONFIRMED/CANDIDATE=1, FALSE POSITIVE=0)
      - sayısal kolonların dtype dönüşümü ve eksik doldurma (median)
      - tekil değeri tek olan sabit sütunları kaldırma
    """
    df = df.copy()
    # Label
    if "koi_disposition" in df.columns:
        label_map = {"CONFIRMED": 1, "CANDIDATE": 1, "FALSE POSITIVE": 0}
        df["label"] = (
            df["koi_disposition"].astype(str).str.upper().str.strip().map(label_map)
        )
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(int)
    else:
        st.warning("`koi_disposition` kolonu bulunamadı; label üretilemedi.")
    # Tip dönüşümleri
    for c in df.columns:
        if c == "label":
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")
    # Bazı yaygın Kepler kolonları yoksa sorun değil; mevcutlarla devam edilir
    # Eksikleri doldur (sadece sayısallar)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "label"]
    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
    # Sabit sütunları ele
    nunique = df.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    df = df[keep].copy()
    return df

# -------- UI --------
st.set_page_config(page_title="Kepler Exoplanet Dataset", page_icon="🪐", layout="wide")
st.title("🪐 Kepler Exoplanet Dataset Görüntüleyici")

with st.sidebar:
    st.header("Ayarlar")
    default_url = "https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results"
    KAGGLE_URL = st.text_input("Kaggle Dataset URL", value=default_url)
    use_kaggle = st.checkbox("Açılışta Kaggle'dan indir", value=True)
    st.caption("Not: Kaggle indirmesi için ortamda kimlik doğrulama gerekir (kaggle.json / opendatasets).")

DATA_DIR = "./data_kepler"

# ---- Load data ----
csv_path = None
df = None

if use_kaggle and KAGGLE_URL.strip():
    csv_path = try_kaggle_download(KAGGLE_URL.strip(), DATA_DIR)

uploaded = st.file_uploader("Ya da CSV dosyasını buradan yükle (alternatif yöntem):", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif csv_path is not None and os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

if df is None:
    st.info("Henüz veri yüklenmedi. Soldan Kaggle indir veya yukarıdan bir CSV yükle.")
    st.stop()

st.success("Veri yüklendi!")

# ---- Preprocess ----
with st.expander("⚙️ Ön işleme (Kepler) — detayları göster/gizle", expanded=False):
    st.write("• `koi_disposition` → `label` (CONFIRMED/CANDIDATE=1, FALSE POSITIVE=0)")
    st.write("• Sayısal kolonlarda median ile eksik doldurma, tekil değerli sabit sütunları kaldırma")
df_pp = preprocess_kepler(df)

# ---- Overview ----
c0, c1, c2 = st.columns(3)
with c0:
    st.metric("Satır Sayısı", f"{df_pp.shape[0]:,}")
with c1:
    st.metric("Sütun Sayısı", f"{df_pp.shape[1]:,}")
with c2:
    st.write("Örnek sütunlar:", ", ".join(df_pp.columns.astype(str).tolist()[:8]) + ("..." if df_pp.shape[1] > 8 else ""))

st.subheader("📊 İlk Satırlar")
st.dataframe(df_pp.head(20))

# Label dağılımı
if "label" in df_pp.columns:
    st.subheader("🏷️ Label Dağılımı")
    vc = df_pp["label"].value_counts().sort_index()
    lbl_df = pd.DataFrame({"label": ["0=FP", "1=Planet"], "count": [int(vc.get(0, 0)), int(vc.get(1, 0))]})
    st.table(lbl_df)
else:
    st.info("Label üretilemedi (koi_disposition bulunamadı).")

# ---- Column selector ----
st.subheader("🔎 Sütun İncele")
sel_cols = st.multiselect("Gösterilecek sütunları seçin:", df_pp.columns.tolist())
if sel_cols:
    st.dataframe(df_pp[sel_cols].head(100))

# ---- Quick Filters ----
with st.expander("🔍 Hızlı Filtreleme"):
    col_to_filter = st.selectbox("Filtrelenecek sütun seçin:", ["— Seç —"] + df_pp.columns.astype(str).tolist(), index=0)
    if col_to_filter != "— Seç —":
        series = df_pp[col_to_filter]
        if pd.api.types.is_numeric_dtype(series):
            min_v, max_v = float(series.min()), float(series.max())
            fmin, fmax = st.slider("Aralık seçin:", min_value=min_v, max_value=max_v, value=(min_v, max_v))
            st.dataframe(df_pp[(series >= fmin) & (series <= fmax)].head(200))
        else:
            text = st.text_input("Metin araması (contains)", "")
            if text:
                st.dataframe(df_pp[series.astype(str).str.contains(text, case=False, na=False)].head(200))

# ---- Describe ----
st.subheader("📈 İstatistikler")
with st.expander("Özet İstatistikler (describe)"):
    st.write(df_pp.describe(include="all"))

# ---- Download ----
st.subheader("💾 Düzenlenmiş Veriyi İndir")
to_save = df_pp if not sel_cols else df_pp[sel_cols]
st.download_button(
    label="CSV'yi indir",
    data=to_save.to_csv(index=False).encode("utf-8"),
    file_name="kepler_preprocessed.csv",
    mime="text/csv"
)

st.caption("Not: Bu sayfa yalnızca keşif/görüntüleme ve basit ön işleme içindir. İstersen modelleme (LightGBM/RF) sekmesi ekleyebilirim.")
