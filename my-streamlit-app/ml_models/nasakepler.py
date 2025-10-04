import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib, json, os
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# ====================================================================
# BÖLÜM 1: Veri Yükleme ve Ön İşleme
# ====================================================================

# Örnek DataFrame Oluşturma (Gerçek veriye erişim yoksa)
try:
    path = "/content/drive/MyDrive/Makine_öğrenmesi/NASA/işlenmiş kepler.csv"
    df = pd.read_csv(path)
except FileNotFoundError:
    print("UYARI: Gerçek veri seti bulunamadı. Örnek DataFrame oluşturuluyor.")
    # Örnek veri oluşturma
    np.random.seed(42)
    N = 7000
    df = pd.DataFrame({
        'kepid': np.repeat(np.arange(N // 2), 2) if N % 2 == 0 else np.concatenate([np.repeat(np.arange((N-1) // 2), 2), [N//2]]),
        'kepoi_name': [f'K{i}' for i in range(N)],
        'koi_disposition': np.random.choice(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"], N, p=[0.1, 0.4, 0.5]),
        'koi_period': np.random.rand(N) * 100 + 1, 'koi_duration': np.random.rand(N) * 10 + 0.1,
        'koi_depth': np.random.rand(N) * 10000 + 10, 'koi_model_snr': np.random.rand(N) * 50 + 5,
        'koi_num_transits': np.random.randint(5, 500, N), 'koi_score': np.random.rand(N),
        'koi_teq': np.random.rand(N) * 2000 + 300, 'koi_prad': np.random.rand(N) * 5 + 0.5,
        'koi_sma': np.random.rand(N) * 2 + 0.1, 'koi_steff': np.random.rand(N) * 3000 + 4000,
        'koi_srad': np.random.rand(N) * 2 + 0.5, 'koi_smass': np.random.rand(N) * 1.5 + 0.5,
        'koi_insol': np.random.rand(N) * 100 + 1, # NaN eklenecek
        'koi_fpflag_nt': np.random.randint(0, 2, N), 'koi_fpflag_ss': np.random.randint(0, 2, N),
        'koi_fpflag_co': np.random.randint(0, 2, N), 'koi_fpflag_ec': np.random.randint(0, 2, N)
    })
    df.loc[::100, 'koi_insol'] = np.nan
    df.loc[::50, 'koi_depth'] = np.nan
    df.loc[::10, 'koi_prad'] = np.nan

print("Veri Önizlemesi:")
display(df.head())
print(f"Veri Boyutu: {df.shape}")

# Eksik Değer Doldurma (koi_insol ve koi_depth için erken doldurma)
median_koi_insol = df['koi_insol'].median()
df['koi_insol'] = df['koi_insol'].fillna(median_koi_insol)
# koi_depth için de median_koi_insol kullanılmış, median ile düzeltildi.
median_koi_depth = df['koi_depth'].median()
df['koi_depth'] = df['koi_depth'].fillna(median_koi_depth)
print(f"koi_depth eksik sayısı: {df['koi_depth'].isnull().sum()}")
print(f"koi_insol eksik sayısı: {df['koi_insol'].isnull().sum()}")

# Etiketleme ve filtreleme
label_map = {"CONFIRMED": 1, "CANDIDATE": 1, "FALSE POSITIVE": 0}
df["label"] = (
    df["koi_disposition"]
    .astype(str).str.upper().str.strip()
    .map(label_map)
)
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)
print("\nLabel dağılımı:")
print(df["label"].value_counts())

# İlk Özellik Seçimi (koi_insol çıkarıldı, çünkü manuel dolduruldu)
feature_cols = [
    "koi_period", "koi_duration", "koi_depth", "koi_model_snr",
    "koi_num_transits", "koi_score", "koi_teq", "koi_prad", "koi_sma",
    "koi_steff", "koi_srad", "koi_smass",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"
]
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"\nKullanılacak özellik sayısı: {len(feature_cols)}")
X = df[feature_cols].copy()
y = df["label"].copy()

# Veri Tipleri ve Eksik Değer İşleme
flag_cols = [c for c in ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"] if c in X.columns]
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")
for c in flag_cols:
    X[c] = X[c].fillna(0).astype("int64")

# Geriye kalan sayısal eksik değerleri median ile doldurma (Sadece ilk model eğitimi için)
X = X.fillna(X.median(numeric_only=True))
nunique = X.nunique(dropna=False)
keep_cols = nunique[nunique > 1].index
X = X[keep_cols].copy()
print("X shape:", X.shape, "| y shape:", y.shape)

# ====================================================================
# BÖLÜM 2: İlk Model Eğitimi (StratifiedKFold)
# ====================================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs, aps, models = [], [], []
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]
    model = lgb.LGBMClassifier(
        n_estimators=10000, learning_rate=0.05, max_depth=-1,
        subsample=0.8, colsample_bytree=0.8, class_weight='balanced', random_state=42
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    proba = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
    auc = roc_auc_score(y_va, proba)
    ap = average_precision_score(y_va, proba)
    aucs.append(auc); aps.append(ap); models.append(model)
    # print(f"Fold {fold}: AUC={auc:.3f}, AP={ap:.3f}, best_iter={model.best_iteration_}") # Yorum satırı kaldırıldı
print("\nStratifiedKFold Sonuçları:")
print("Mean AUC:", np.mean(aucs))
print("Mean AP :", np.mean(aps))
best_model = models[int(np.argmax(aucs))]

print("\nEn İyi Modelden İlk Özellik Önem Skorları (Gini):")
importances = best_model.feature_importances_
order = np.argsort(importances)[::-1][:20]
top20 = [(X.columns[i], int(importances[i])) for i in order]
for name, imp in top20:
    print(f"{name:25s} {imp}")

# ====================================================================
# BÖLÜM 3: GroupKFold ile Korelasyon ve Güvenilirlik Analizi
# ====================================================================

id_col = "kepid"
uniq_key = "kepoi_name"
# df'in duplicate'leri düşülmüş hali zaten yukarıda oluşturuldu.
if "kepoi_name" in df.columns:
    df = df.drop_duplicates(subset=["kepoi_name"]).copy()
if "kepid" not in df.columns:
     df['kepid'] = df[uniq_key].str.replace('K', '').astype(int) # kepoi_name'den basitçe türetildi
     
core_feats = [
    "koi_period","koi_duration","koi_depth","koi_model_snr",
    "koi_num_transits","koi_teq","koi_prad","koi_sma",
    "koi_steff","koi_srad","koi_smass"
]
core_feats = [c for c in core_feats if c in df.columns]
suspect_feats = [c for c in ["koi_score","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec"] if c in df.columns]

def eval_cv(feats, title):
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(int)
    groups = df[id_col]
    gkf = GroupKFold(n_splits=5)
    aucs, aps = [], []
    for tr, va in gkf.split(X, y, groups):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_va = imp.transform(X_va)
        model = lgb.LGBMClassifier(
            n_estimators=10000, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)], eval_metric="auc",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        proba = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:,1]
        aucs.append(roc_auc_score(y_va, proba))
        aps.append(average_precision_score(y_va, proba))
    print(f"{title} -> AUC={np.mean(aucs):.4f}, AP={np.mean(aps):.4f}")

print("\nGroupKFold Değerlendirmeleri (Güvenilirlik Testi):")
eval_cv(core_feats, "CORE only (transit + star phys)")
if suspect_feats:
    eval_cv(core_feats + suspect_feats, "CORE + suspect (score/flags)")
if suspect_feats:
    eval_cv(suspect_feats, "SUSPECT only")

print("\nÖzellik-Etiket Korelasyonları:")
for c in core_feats + suspect_feats:
    s = pd.to_numeric(df[c], errors="coerce")
    corr = np.corrcoef(s.fillna(s.median()), df["label"])[0,1]
    print(f"{c:15s} corr≈ {corr:.3f}")

# Şüpheli özelliklerin kaldırılması (Kullanıcının tercih ettiği gibi)
df.drop(columns=[c for c in ['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co'] if c in df.columns], inplace=True, errors='ignore')
suspect_feats = [c for c in suspect_feats if c not in ['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co']]

# ====================================================================
# BÖLÜM 4: Özellik Mühendisliği (Feature Engineering)
# ====================================================================

# Fonksiyonların yeniden tanımlanması
def add_engineered(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    eps = 1e-6
    if "koi_duration" in X.columns and "koi_period" in X.columns:
        X["dur_per_ratio"]   = X["koi_duration"] / (X["koi_period"] + eps)
    if "koi_depth" in X.columns and "koi_duration" in X.columns:
        X["depth_dur_ratio"] = X["koi_depth"] / (X["koi_duration"] + eps)
    if "koi_prad" in X.columns and "koi_srad" in X.columns:
        X["prad_srad_ratio"] = X["koi_prad"] / (X["koi_srad"] + eps)
    if "koi_teq" in X.columns and "koi_steff" in X.columns:
        X["teq_teff_ratio"]  = X["koi_teq"] / (X["koi_steff"] + eps)
    if "koi_sma" in X.columns and "koi_srad" in X.columns:
        X["sma_srad_ratio"]  = X["koi_sma"] / (X["koi_srad"] + eps)
    if "koi_depth" in X.columns and "koi_model_snr" in X.columns:
        X["depth_x_snr"]     = X["koi_depth"] * X["koi_model_snr"]
    if "koi_period" in X.columns and "koi_num_transits" in X.columns:
        X["period_x_transits"] = X["koi_period"] * X["koi_num_transits"]
    if "koi_prad" in X.columns:
        X["flag_large_prad"]   = (X["koi_prad"] > 20).astype(int)
    if "koi_duration" in X.columns and "koi_period" in X.columns:
        X["flag_invalid_dur"]  = (X["koi_duration"] > X["koi_period"]).astype(int)
    return X

def eval_model(X, y, label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, aps = [], []
    for tr, va in skf.split(X, y):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X.iloc[tr])
        X_va = imp.transform(X.iloc[va])
        model = lgb.LGBMClassifier(
            n_estimators=5000, learning_rate=0.05, class_weight="balanced", random_state=42
        )
        model.fit(X_tr, y.iloc[tr],
                  eval_set=[(X_va, y.iloc[va])],
                  eval_metric="auc",
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])
        p = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:,1]
        aucs.append(roc_auc_score(y.iloc[va], p))
        aps.append(average_precision_score(y.iloc[va], p))
    print(f"{label:<20s} | AUC={np.mean(aucs):.4f} | AP={np.mean(aps):.4f}")

# Core ve engineered özellik setlerini hazırla
core_cols = [c for c in core_feats if c in df.columns]
X_core = df[core_cols].apply(pd.to_numeric, errors="coerce")
y = df["label"].astype(int)

# Core + engineered
X_core_plus = add_engineered(X_core)
X_core_plus = X_core_plus.apply(pd.to_numeric, errors="coerce") # Yeni kolonlar için tekrar zorla

print("\nÖzellik Mühendisliği Değerlendirmesi (StratifiedKFold):")
eval_model(X_core, y, "Core only")
eval_model(X_core_plus, y, "Core + engineered")
# eval_model(X_eng, y, "Engineered only") # Orijinal kodda bu kısım için X_eng tekrar hesaplanmadığından güvenli olması için yoruma alındı.

# ====================================================================
# BÖLÜM 5: Özellik Önemi ile Sadeleştirme ve NİHAİ MODEL EĞİTİMİ
# ====================================================================

# Fonksiyonun yeniden tanımlanması
def mean_importance(X, y, n_splits=5):
    # Bu model, final model değil, sadece feature selection için kullanılır.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    feats = X.columns.tolist()
    imps = np.zeros(len(feats))
    for tr, va in skf.split(X, y):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X.iloc[tr]); ytr = y.iloc[tr]
        Xva = imp.transform(X.iloc[va]);     yva = y.iloc[va]
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05,
                               class_weight="balanced", random_state=42)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              eval_metric="auc",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])
        imps += m.feature_importances_
    imps /= n_splits
    return pd.Series(imps, index=feats).sort_values(ascending=False)

# 1) "Core + engineered" ile importanceları ölç
imp_series = mean_importance(X_core_plus, y)

# 2) Engineered olanlardan düşük önemlileri ele
engineered_cols = [c for c in X_core_plus.columns if c not in X_core.columns]
drop_eng = [c for c in engineered_cols if imp_series.get(c, 0) <= 0]

# 3) Sadeleştirilmiş nihai özellik seti
X_slim = X_core_plus.drop(columns=drop_eng, errors="ignore")
final_feats = X_slim.columns.tolist()

print("\nNihai Özellik Sadeleştirme:")
print("Atılan engineered kolon sayısı:", len(drop_eng))
print("Nihai kullanılan özellik sayısı:", len(final_feats))
print("Nihai feature listesi:\n", final_feats)

# ====================================================================
# BÖLÜM 6: NİHAİ MODELİN YENİDEN EĞİTİMİ VE KAYDEDİLMESİ (DÜZELTME)
# ====================================================================

# NİHAİ ÖZELLİK KÜMESİ ÜZERİNDE GroupKFold ile son model eğitimi
X_final = df[final_feats].apply(pd.to_numeric, errors="coerce") # Yeni X_final
y = df["label"].astype(int)
groups = df["kepid"] # GroupKFold için gruplar

gkf = GroupKFold(n_splits=5)
aucs_final, aps_final, models_final = [], [], []

for fold, (tr, va) in enumerate(gkf.split(X_final, y, groups), 1):
    X_tr, X_va = X_final.iloc[tr], X_final.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]
    imp_fold = SimpleImputer(strategy="median")
    X_tr_imp = imp_fold.fit_transform(X_tr)
    X_va_imp = imp_fold.transform(X_va)
    
    model = lgb.LGBMClassifier(
        n_estimators=10000, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        max_depth=-1, class_weight="balanced", random_state=42,
    )
    model.fit(
        X_tr_imp, y_tr,
        eval_set=[(X_va_imp, y_va)], eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    proba = model.predict_proba(X_va_imp, num_iteration=model.best_iteration_)[:, 1]
    aucs_final.append(roc_auc_score(y_va, proba))
    aps_final.append(average_precision_score(y_va, proba))
    models_final.append((model, imp_fold))
    print(f"Fold {fold} (Final): AUC={aucs_final[-1]:.4f}, AP={aps_final[-1]:.4f}, best_iter={model.best_iteration_}")

print("\nFinal Model (GroupKFold) Sonuçları:")
print("Mean AUC:", float(np.mean(aucs_final)))
print("Mean AP :", float(np.mean(aps_final)))

# Nihai modelin tüm veri üzerinde eğitilmesi
imp_final = SimpleImputer(strategy="median")
X_all = imp_final.fit_transform(X_final)
best_idx = int(np.argmax(aucs_final))
best_iter = models_final[best_idx][0].best_iteration_ or 1000 # En iyi iterasyon sayısını kullan

final_model_re = lgb.LGBMClassifier( # Düzeltilmiş final_model_re adı
    n_estimators=best_iter, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    max_depth=-1, class_weight="balanced", random_state=42,
)
final_model_re.fit(X_all, y)

# Kaydetme
os.makedirs("models", exist_ok=True)
joblib.dump(final_model_re, "models/kepler_final_lgbm.pkl")
joblib.dump(imp_final, "models/kepler_final_imputer.pkl")
with open("models/kepler_final_features.json", "w") as f:
    json.dump({"feature_cols": final_feats}, f, indent=2)
print("\nKaydedildi: models/kepler_final_lgbm.pkl + kepler_final_imputer.pkl + kepler_final_features.json")

# ====================================================================
# BÖLÜM 7: NİHAİ MODELİN DEĞERLENDİRME GRAFİKLERİ
# ====================================================================

X_eval = X_final.copy() # Değerlendirme için X_final kullanılır
X_eval_imp = imp_final.transform(X_eval) # Kaydedilen imputer kullanılır

# 1. Özellik Önem Grafiği (Yeni model)
print("\nNihai Model Özellik Önem Skorları:")
importances = final_model_re.feature_importances_
feat_names = final_feats
imp_series = pd.Series(importances, index=feat_names).sort_values(ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x=imp_series.values[:15], y=imp_series.index[:15], palette="viridis")
plt.title("Top 15 Feature Importances (Final Model)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 2. ROC Eğrisi
y_pred = final_model_re.predict_proba(X_eval_imp)[:,1]
fpr, tpr, _ = roc_curve(y, y_pred)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, y_pred):.3f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Final Model)")
plt.legend()
plt.grid(True)
plt.show()

# 3. Precision-Recall Eğrisi
prec, rec, _ = precision_recall_curve(y, y_pred)
ap = average_precision_score(y, y_pred)
plt.figure(figsize=(6,6))
plt.plot(rec, prec, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Final Model)")
plt.legend()
plt.grid(True)
plt.show()

# 4. Confusion Matrix
y_class = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y, y_class, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["False Positive","Planet"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Final Model, threshold=0.5)")
plt.show()

# 5. SHAP Özet Grafikleri
try:
    explainer = shap.TreeExplainer(final_model_re)
    # SHAP değerleri X_eval_imp (doldurulmuş/impute edilmiş veri) üzerinde hesaplanmalıdır.
    shap_values = explainer.shap_values(X_eval_imp) 
    
    # SHAP plotları için X'i DataFrame olarak tutmak daha iyidir
    X_eval_df = pd.DataFrame(X_eval_imp, columns=final_feats)
    
    # SHAP Bar Grafiği (Ortalama Mutlak SHAP Değeri)
    print("\nSHAP Değerleri - Bar Grafiği (Modelde En Önemli Özellikler):")
    shap.summary_plot(shap_values[1], X_eval_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.show()
    
    # SHAP Dot Grafiği (Özellik Etkisi ve Yönü)
    print("\nSHAP Değerleri - Dot Grafiği (Her Özelliğin Tahmine Etkisi):")
    shap.summary_plot(shap_values[1], X_eval_df, show=False)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"\nSHAP hesaplaması sırasında hata oluştu: {e}")
