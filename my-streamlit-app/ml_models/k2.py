import streamlit as st
import pandas as pd
import os

# Optional: Use opendatasets to pull from Kaggle (fallback to uploader if token missing)
def try_kaggle_download(url: str, data_dir: str) -> str | None:
    """
    Tries to download a Kaggle dataset using opendatasets.
    Returns a path to a CSV file if found, else None.
    """
    try:
        import opendatasets as od  # requires: pip install opendatasets
        if not os.path.exists(data_dir):
            st.info("⏬ Kaggle'dan veri indiriliyor... (ilk çalıştırmada biraz sürebilir)")
            od.download(url, data_dir=data_dir)
        # Walk the directory to find the first CSV
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(".csv"):
                    return os.path.join(root, f)
        return None
    except Exception as e:
        st.warning(f"Kaggle indirme denemesi başarısız: {e}")
        return None

st.set_page_config(page_title="NASA K2 Dataset", page_icon="🚀", layout="wide")
st.title("🚀 NASA K2 Dataset Görüntüleyici")

# ==== Settings ====
KAGGLE_URL = "https://www.kaggle.com/datasets/dc07759f3611e93d7d10c09094fc05e27ddc787024e140043545ad9e0f4298c6"
DATA_DIR = "./data_k2"

with st.sidebar:
    st.header("Ayarlar")
    use_kaggle = st.checkbox("Açılışta Kaggle'dan indir", value=True)
    st.caption("Kaggle indirmesi için ortamda 'kaggle.json' veya opendatasets kimlik doğrulaması gerekir.")

# ==== Load data ====
csv_path = None

if use_kaggle:
    csv_path = try_kaggle_download(KAGGLE_URL, DATA_DIR)

uploaded = st.file_uploader("Ya da CSV dosyasını buradan yükle (alternatif yöntem):", type=["csv"])
if uploaded is not None:
    csv_path = None  # prioritize uploaded file
    df = pd.read_csv(uploaded)
else:
    # If nothing uploaded, try reading from csv_path (if any)
    if csv_path is not None and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = None

if df is None:
    st.info(
        "Henüz veri yüklenmedi. Soldaki ayardan Kaggle indirmesini açabilir veya yukarıdan bir CSV yükleyebilirsiniz."
    )
    st.stop()

st.success("Veri yüklendi!")

# ==== Basic overview ====
st.subheader("📊 İlk Satırlar")
st.dataframe(df.head(20))

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Satır Sayısı", f"{df.shape[0]:,}")
with c2:
    st.metric("Sütun Sayısı", f"{df.shape[1]:,}")
with c3:
    st.write("Sütunlar:", ", ".join(df.columns.astype(str).tolist()[:8]) + ("..." if df.shape[1] > 8 else ""))

# ==== Column selector ====
st.subheader("🔎 Sütun İncele")
sel_cols = st.multiselect("Gösterilecek sütunları seçin:", df.columns.tolist())
if sel_cols:
    st.dataframe(df[sel_cols].head(100))

# ==== Filter tools ====
with st.expander("🔍 Hızlı Filtreleme"):
    col_to_filter = st.selectbox("Filtrelenecek sütun seçin:", ["— Seç —"] + df.columns.astype(str).tolist(), index=0)
    if col_to_filter != "— Seç —":
        series = df[col_to_filter]
        if pd.api.types.is_numeric_dtype(series):
            min_v, max_v = float(series.min()), float(series.max())
            fmin, fmax = st.slider("Aralık seçin:", min_value=min_v, max_value=max_v, value=(min_v, max_v))
            st.dataframe(df[(series >= fmin) & (series <= fmax)].head(200))
        else:
            text = st.text_input("Metin araması (contains)", "")
            if text:
                st.dataframe(df[series.astype(str).str.contains(text, case=False, na=False)].head(200))

# ==== Describe ====
st.subheader("📈 İstatistikler")
with st.expander("Özet İstatistikler (describe)"):
    st.write(df.describe(include="all"))

# ==== Download subset ====
st.subheader("💾 Alt Küme İndir")
if sel_cols:
    csv_bytes = df[sel_cols].to_csv(index=False).encode("utf-8")
else:
    csv_bytes = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="CSV'yi indir",
    data=csv_bytes,
    file_name="k2_subset.csv",
    mime="text/csv",
)

st.caption("Not: Kaggle indirme başarısız olursa dosyayı yükleyici ile ekleyebilirsiniz. Büyük veri setlerinde ilk yükleme zaman alabilir.")
