import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple

# -------------------- UI Setup --------------------
st.set_page_config(page_title="TESS / TOI Exoplanet Classifier", page_icon="🛰️", layout="wide")
st.title("🛰️ TESS / TOI Exoplanet Classifier (SMOTE + RandomForest)")

with st.sidebar:
    st.header("Ayarlar")
    default_url = "https://www.kaggle.com/datasets/nasa/toi-catalog"  # kullanıcı kendi URL'sini girebilir
    kaggle_url = st.text_input("Kaggle Dataset URL", value=default_url)
    use_kaggle = st.checkbox("Açılışta Kaggle'dan indir", value=True)
    st.caption("Not: Kaggle indirmesi için kimlik doğrulama gerekir (kaggle.json / opendatasets).")
    test_size = st.slider("Test oranı", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.number_input("Rastgele tohum (random_state)", min_value=0, value=42, step=1)

DATA_DIR = "./data_toi"

# -------------------- Helpers --------------------
def try_kaggle_download(url: str, data_dir: str) -> Optional[str]:
    """
    Tries to download a Kaggle dataset using opendatasets.
    Returns a path to a CSV file if found, else None.
    """
    try:
        import opendatasets as od  # pip install opendatasets
        if not os.path.exists(data_dir):
            st.info("⏬ Kaggle'dan veri indiriliyor... (ilk çalıştırmada biraz sürebilir)")
            od.download(url, data_dir=data_dir)
        # Walk the directory to find a CSV
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(".csv"):
                    return os.path.join(root, f)
        return None
    except Exception as e:
        st.warning(f"Kaggle indirme denemesi başarısız: {e}")
        return None

def load_dataframe(kaggle_first: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    csv_path = None
    df = None
    if kaggle_first and use_kaggle and kaggle_url.strip():
        csv_path = try_kaggle_download(kaggle_url.strip(), DATA_DIR)

    uploaded = st.file_uploader("Ya da CSV dosyasını buradan yükle (alternatif yöntem):", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        return df, "uploaded"
    if csv_path is not None and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df, csv_path
    return None, None

# -------------------- Load Data --------------------
df, src = load_dataframe(kaggle_first=True)
if df is None:
    st.info("Henüz veri yüklenmedi. Soldan Kaggle indir veya yukarıdan bir CSV yükle.")
    st.stop()

st.success("Veri yüklendi!")
st.caption(f"Kaynak: {src}")

# -------------------- Preprocess --------------------
st.subheader("⚙️ Ön İşleme")
with st.expander("Ön işleme adımları (TOI verisine göre):", expanded=False):
    st.markdown("""
- Hedef **`tfopwg_disp`** sütunundan üretilir: `'FP' → 0`, diğerleri → `1` (PLANET).
- `rowid, toi, toipfx, tid, ctoi_alias` (id) ve `rastr, decstr` (metin) gibi kolonlar modelden düşürülür.
- Kalan sayısal sütunlar seçilir.
- Ek özellikler (varsa): `log_trandep`, `dur_per_ratio`, `insol_rad2`, `log_teff`.
- Eksikler **median** ile doldurulur, ardından **StandardScaler** uygulanır.
- Sınıf dengesizliği için **SMOTE** kullanılır.
""")

# Target mapping
target_col = 'tfopwg_disp'
assert target_col in df.columns, "tfopwg_disp kolonu bulunamadı!"
labels = df[target_col].astype(str).str.upper().str.strip()
y = np.where(labels == 'FP', 0, 1)  # 0=FP, 1=PLANET

id_cols = ['rowid','toi','toipfx','tid','ctoi_alias']
drop_text_cols = ['rastr','decstr']

# Feature engineering (conditional)
df_fe = df.copy()
if 'pl_trandep' in df_fe.columns:
    df_fe['log_trandep'] = np.log1p(df_fe['pl_trandep'])
if 'pl_trandurh' in df_fe.columns and 'pl_orbper' in df_fe.columns:
    df_fe['dur_per_ratio'] = df_fe['pl_trandurh'] / (df_fe['pl_orbper'] + 1e-6)
if 'pl_insol' in df_fe.columns and 'st_rad' in df_fe.columns:
    df_fe['insol_rad2'] = df_fe['pl_insol'] / (df_fe['st_rad']**2 + 1e-6)
if 'st_teff' in df_fe.columns:
    df_fe['log_teff'] = np.log1p(df_fe['st_teff'])

X = df_fe.drop(columns=[c for c in (id_cols + drop_text_cols + [target_col]) if c in df_fe.columns], errors='ignore')
num_cols = X.select_dtypes(include=['number']).columns.tolist()
X = X[num_cols].copy()

st.write("Seçilen sayısal özellik sayısı:", len(num_cols))
st.dataframe(X.head(10))

# -------------------- Modeling --------------------
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay,
                             average_precision_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocess = ColumnTransformer(
    transformers=[('num', numeric_transformer, num_cols)],
    remainder='drop'
)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    random_state=random_state,
    n_jobs=-1
)

imb_pipe = ImbPipeline(steps=[
    ('prep', preprocess),
    ('smote', SMOTE(random_state=random_state, sampling_strategy='auto')),
    ('model', rf)
])

tabs = st.tabs(["📊 Keşif", "🤖 Model Eğitimi", "📈 Değerlendirme", "⭐ Önem / Permütasyon", "🧪 Çapraz Doğrulama", "💾 İndir / Kaydet"])

with tabs[0]:
    st.subheader("Veri Önizleme")
    st.write("İlk 20 satır:")
    st.dataframe(df.head(20))

with tabs[1]:
    st.subheader("Modeli Eğit")
    if st.button("Eğitimi Başlat"):
        imb_pipe.fit(X_train, y_train)
        st.success("Model eğitildi!")
        st.session_state['model'] = imb_pipe
    else:
        st.info("Eğitmek için düğmeye basın.")

with tabs[2]:
    st.subheader("Test Seti Değerlendirme")
    if 'model' not in st.session_state:
        st.warning("Önce modeli eğitmelisiniz (Model Eğitimi sekmesi).")
    else:
        model = st.session_state['model']
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        c1, c2, c3 = st.columns(3)
        with c1:
            acc = (y_pred == y_test).mean()
            st.metric("Doğruluk", f"{acc:.3f}")
        with c2:
            auc = roc_auc_score(y_test, y_proba)
            st.metric("ROC-AUC", f"{auc:.3f}")
        with c3:
            ap = average_precision_score(y_test, y_proba)
            st.metric("Average Precision", f"{ap:.3f}")

        st.code(classification_report(y_test, y_pred, target_names=['FP','PLANET'], digits=3), language="text")

        # Plots (matplotlib only, no seaborn)
        import matplotlib.pyplot as plt
        fig_cm, ax_cm = plt.subplots(figsize=(5,5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['FP','PLANET'], ax=ax_cm, colorbar=False)
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

        fig_roc, ax_roc = plt.subplots(figsize=(5,4))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax_roc, name="RandomForest")
        ax_roc.set_title("ROC Curve (PLANET = positive)")
        st.pyplot(fig_roc)

        fig_pr, ax_pr = plt.subplots(figsize=(5,4))
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax_pr, name="RandomForest")
        ax_pr.set_title("Precision–Recall (PLANET = positive)")
        st.pyplot(fig_pr)

        # Threshold search for FP emphasis
        ths = np.linspace(0.05, 0.95, 181)
        from sklearn.metrics import f1_score
        fp_proba = 1 - y_proba

        best_t_fp, best_f1_fp = 0.5, -1
        for t in ths:
            y_pred_fp = (fp_proba >= t).astype(int)
            f1t = f1_score((y_test==0).astype(int), y_pred_fp)
            if f1t > best_f1_fp:
                best_f1_fp, best_t_fp = f1t, t

        st.info(f"[FP=1] En iyi threshold: **{best_t_fp:.2f}** | F1(FP): **{best_f1_fp:.3f}**")

        y_pred_best = (fp_proba >= best_t_fp).astype(int)
        fig_cm2, ax_cm2 = plt.subplots(figsize=(5,5))
        ConfusionMatrixDisplay.from_predictions((y_test==0).astype(int), y_pred_best,
                                                display_labels=['Not-FP','FP'], ax=ax_cm2, colorbar=False)
        ax_cm2.set_title(f"Confusion Matrix @ threshold={best_t_fp:.2f} (FP=positive)")
        st.pyplot(fig_cm2)

with tabs[3]:
    st.subheader("Özellik Önemleri")
    if 'model' not in st.session_state:
        st.warning("Önce modeli eğitmelisiniz (Model Eğitimi sekmesi).")
    else:
        model = st.session_state['model']
        import matplotlib.pyplot as plt
        # Gini importances
        rf_fitted = model.named_steps['model']
        try:
            gini_imp = rf_fitted.feature_importances_
            feat_names = model.named_steps['prep'].get_feature_names_out()
            imp_df = (pd.DataFrame({'feature': feat_names, 'importance': gini_imp})
                        .sort_values('importance', ascending=False).head(20))
            fig_gini, ax_gini = plt.subplots(figsize=(7,6))
            ax_gini.barh(imp_df['feature'][::-1], imp_df['importance'][::-1])
            ax_gini.set_title("RandomForest Gini Importances (Top 20)")
            st.pyplot(fig_gini)
        except Exception as e:
            st.warning(f"Gini importance görüntülenemedi: {e}")

        # Permutation importances (on test set)
        try:
            from sklearn.inspection import permutation_importance
            X_test_tr = model.named_steps['prep'].transform(X_test)
            perm = permutation_importance(rf_fitted, X_test_tr, y_test, n_repeats=10,
                                          random_state=random_state, n_jobs=-1)
            feat_out = model.named_steps['prep'].get_feature_names_out()
            perm_df = (pd.DataFrame({'feature': feat_out, 'importance': perm.importances_mean})
                         .sort_values('importance', ascending=False).head(20))
            fig_perm, ax_perm = plt.subplots(figsize=(7,6))
            ax_perm.barh(perm_df['feature'][::-1], perm_df['importance'][::-1])
            ax_perm.set_title("Permutation Importances (Top 20)")
            st.pyplot(fig_perm)
        except Exception as e:
            st.warning(f"Permutation importance hesaplanamadı: {e}")

with tabs[4]:
    st.subheader("Çapraz Doğrulama (StratifiedKFold=5)")
    if 'model' not in st.session_state:
        st.warning("Önce modeli eğitmelisiniz (Model Eğitimi sekmesi).")
    else:
        from sklearn.metrics import make_scorer, balanced_accuracy_score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        from joblib import parallel_backend
        with parallel_backend('loky'):
            bal_acc = cross_val_score(imb_pipe, X, y, cv=cv,
                                      scoring=make_scorer(balanced_accuracy_score), n_jobs=-1)
            f1      = cross_val_score(imb_pipe, X, y, cv=cv, scoring='f1', n_jobs=-1)
            auc     = cross_val_score(imb_pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("CV Balanced Acc (mean)", f"{bal_acc.mean():.3f}")
        with c2:
            st.metric("CV F1 (mean)", f"{f1.mean():.3f}")
        with c3:
            st.metric("CV ROC-AUC (mean)", f"{auc.mean():.3f}")
        st.write("Balanced Acc:", np.round(bal_acc,3))
        st.write("F1:", np.round(f1,3))
        st.write("ROC-AUC:", np.round(auc,3))

with tabs[5]:
    st.subheader("Veri ve Model İndirme")
    # Preprocessed CSV
    st.download_button(
        label="Ön-işlenmiş veriyi CSV indir",
        data=X.to_csv(index=False).encode("utf-8"),
        file_name="toi_preprocessed.csv",
        mime="text/csv"
    )
    # Save model (if exists)
    if 'model' in st.session_state:
        import joblib, io
        memfile = io.BytesIO()
        joblib.dump(st.session_state['model'], memfile)
        memfile.seek(0)
        st.download_button(
            label="Eğitilmiş modeli indir (.joblib)",
            data=memfile.read(),
            file_name="toi_rf_smote_model.joblib",
            mime="application/octet-stream"
        )
    else:
        st.info("Model henüz eğitilmedi.")

st.caption("Not: Grafikler Matplotlib ile çizilir. Seaborn kullanılmaz. Büyük veri setlerinde ilk yükleme zaman alabilir.")
