import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

# Veri yükleme
path = "/content/drive/MyDrive/Makine_öğrenmesi/NASA/k2pandc_2025.10.02_15.23.29.csv"
df = pd.read_csv(
    path,
    comment="#",
    sep=None,
    engine="python",
    encoding="utf-8",
)
print("Orijinal veri boyutu:", df.shape)

# Kolon isimlerini temizleme
df.columns = (df.columns.str.strip()
              .str.replace(r"\s+", "_", regex=True)
              .str.lower())

# K2 ve transit filtreleme
m_k2 = df.get("disc_facility", "").astype(str).str.contains("k2", case=False, na=False)
m_tr = df.get("discoverymethod", "").astype(str).str.contains("transit", case=False, na=False)
df = df[m_k2 & m_tr] if (m_k2.any() and m_tr.any()) else df
print("Filtreleme sonrası boyut:", df.shape)

# Etiket sütununu bulma
label_col = None
for c in ["disposition", "archive_disposition", "koi_disposition", "tfopwg_disposition"]:
    if c in df.columns:
        label_col = c
        break

assert label_col is not None, "Disposition benzeri bir etiket sütunu bulunamadı."

# Hedef değişkeni oluşturma
y = (df[label_col].astype(str).str.upper()
     .map({"CONFIRMED": 1, "FALSE POSITIVE": 0, "CANDIDATE": 0}))  # Candidate'ları da FP olarak işaretle

# Sadece 0 ve 1 olan satırları al
valid_mask = y.isin([0, 1])
df = df[valid_mask].copy()
y = y[valid_mask].astype(int)

print("Son veri boyutu:", len(df))
print("Pozitif oranı (Confirmed):", y.mean().round(3))
print("Sınıf dağılımı:", dict(zip(['FP', 'Confirmed'], np.bincount(y))))

# Grup sütununu bulma
group_col = None
for g in ["pl_name", "hostname", "epic", "epic_id", "kepid", "tic_id"]:
    if g in df.columns and df[g].notna().any():
        group_col = g
        break
print("Group column:", group_col)

# Yardımcı fonksiyonlar
def cols_present(cols, in_df):
    return [c for c in cols if c in in_df.columns]

def safe_drop(in_df, cols):
    cols2 = cols_present(cols, in_df)
    return in_df.drop(columns=cols2, errors='ignore')

def find_high_corr_pairs(X, thr=0.98):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if len(num) < 2:
        return []
    
    corr = X[num].corr(method="spearman").abs()
    pairs = []
    for i, a in enumerate(num):
        for b in num[i+1:]:
            if corr.loc[a, b] >= thr:
                pairs.append((a, b, float(corr.loc[a, b])))
    return pairs

def prune_by_corr(X, pairs):
    drop = set()
    for a, b, _ in pairs:
        if a not in drop and b not in drop:
            drop.add(b)
    return safe_drop(X, list(drop)), list(drop)

# Özellik seçimi - leakage önleme
patterns = [
    "disposition", "disp_", "_disposition", "disposition_score",
    "refname", "disc_year", "disc_facility", "rowupdate", "pl_pubdate",
    "default_flag", "_flag", "score_", "_score"
]
hard_drop = [c for c in df.columns if any(p in c.lower() for p in patterns)]
if group_col: 
    hard_drop.append(group_col)
hard_drop = list(set(hard_drop + [label_col]))

X = safe_drop(df, hard_drop)
print("Hard-drop edilen kolon sayısı:", len(set(hard_drop)))
print("Kalan feature sayısı:", X.shape[1])

# Eksik veri analizi - leakage kontrolü
if len(y.unique()) > 1:  # En az iki sınıf olduğundan emin ol
    miss_by_class = df.assign(y=y).groupby(y).apply(lambda d: X.reindex(d.index).isna().mean())
    miss_diff = (miss_by_class.loc[1] - miss_by_class.loc[0]).abs().sort_values(ascending=False)
    leaky_missing = miss_diff[miss_diff >= 0.60]
    X = safe_drop(X, list(leaky_missing.index))
    print("NaN-gap nedeniyle atılan:", len(leaky_missing))

# Yüksek korelasyonlu özellikleri temizleme
pairs = find_high_corr_pairs(X, thr=0.98)
X, dropped_corr = prune_by_corr(X, pairs)
print("Yüksek korelasyon nedeniyle atılan:", len(dropped_corr))
print("Kalan feature:", X.shape[1])

# Sayısal kolonları seç
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
X_num = X[num_cols].copy()

print("Model eğitimi için kullanılacak özellik sayısı:", len(num_cols))

# Model ve pipeline oluşturma
rf = RandomForestClassifier(
    n_estimators=400, 
    random_state=42, 
    n_jobs=-1,
    class_weight="balanced_subsample", 
    max_features="sqrt"
)

pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("rf", rf)
])

# Cross-validation
if group_col and df[group_col].notna().all():
    groups = df[group_col]
    gkf = GroupKFold(n_splits=min(5, len(groups.unique())))  # Grup sayısına göre split ayarla
    splits = list(gkf.split(X_num, y, groups))
    scores_gkf = cross_val_score(pipe, X_num, y, cv=splits, scoring="roc_auc")
    print("GroupKFold ROC-AUC:", np.round(scores_gkf, 3), "mean=", scores_gkf.mean().round(3))
else:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_kf = cross_val_score(pipe, X_num, y, cv=kf, scoring="roc_auc")
    print("KFold ROC-AUC:", np.round(scores_kf, 3), "mean=", scores_kf.mean().round(3))

# Manuel cross-validation için veri hazırlığı
X_work = X_num.copy()
y_work = y.copy()
groups_work = df[group_col] if (group_col and group_col in df.columns) else None

# Manuel CV ile detaylı değerlendirme
y_true_all, y_pred_all, y_prob_all = [], [], []

if group_col and groups_work is not None and groups_work.notna().all():
    gkf = GroupKFold(n_splits=min(5, len(groups_work.unique())))
    splits = list(gkf.split(X_work, y_work, groups_work))
else:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(X_work, y_work))

for tr, te in splits:
    Xtr, Xte = X_work.iloc[tr], X_work.iloc[te]
    ytr, yte = y_work.iloc[tr], y_work.iloc[te]
    
    pipe.fit(Xtr, ytr)
    p = pipe.predict(Xte)
    pr = pipe.predict_proba(Xte)[:, 1]
    
    y_true_all.extend(yte)
    y_pred_all.extend(p)
    y_prob_all.extend(pr)

cm = confusion_matrix(y_true_all, y_pred_all)
auc = roc_auc_score(y_true_all, y_prob_all)
print("\nConfusion matrix:")
print(cm)
print("ROC-AUC:", round(auc, 3))

# Feature importance
pipe.fit(X_work, y_work)  # Tüm veri ile tekrar eğit
model = pipe.named_steps["rf"]
feat_imp = pd.Series(model.feature_importances_, index=X_work.columns).sort_values(ascending=False)
print("\nTop 20 important features:")
print(feat_imp.head(20))

# Leakage şüphesi olan özellikleri kontrol et
leak_suspects = [c for c in X_work.columns
                 if any(p in c.lower() for p in [
                     "sy_pnum", "pnum", "err", "error", "dist", "radeerr",
                     "st_teff", "st_rad", "st_rader", "st_mass", "fpp"
                 ])]
print("\nŞüpheli leak kolonları:", leak_suspects)

# Leakage özellikleri çıkarıp tekrar deneme
if leak_suspects:
    X_clean = X_work.drop(columns=leak_suspects, errors='ignore')
    print("Leak özellikler çıkarıldıktan sonra kalan özellik sayısı:", X_clean.shape[1])
    
    rf_clean = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    
    pipe_clean = Pipeline([
        ("imp", SimpleImputer(strategy="median")), 
        ("rf", rf_clean)
    ])
    
    if group_col and groups_work is not None and groups_work.notna().all():
        gkf = GroupKFold(n_splits=min(5, len(groups_work.unique())))
        splits = list(gkf.split(X_clean, y, groups_work))
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(X_clean, y))
    
    scores = []
    for tr, te in splits:
        pipe_clean.fit(X_clean.iloc[tr], y.iloc[tr])
        prob = pipe_clean.predict_proba(X_clean.iloc[te])[:, 1]
        scores.append(roc_auc_score(y.iloc[te], prob))
    
    print("Yeni ROC-AUC (leak özellikler çıkarıldı):", np.round(scores, 3), "mean=", np.mean(scores).round(3))
else:
    print("Şüpheli leak kolonu bulunamadı.")
