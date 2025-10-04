import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, RocCurveDisplay, precision_recall_curve,
                             average_precision_score, PrecisionRecallDisplay, f1_score,
                             make_scorer, balanced_accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
pd.set_option('display.max_columns', None)

# Gerekli importlar eklendi

# ====================================================================
# BÖLÜM 1: Veri Yükleme ve Hazırlık (Orijinal hali korunmuştur)
# ====================================================================

# Sürücü bağlama ve veri okuma kısmı (Colab ortamına özeldir, burada çalıştırılmaz)
# path = "/content/drive/MyDrive/Makine_öğrenmesi/NASA/işlenmiş TOI.csv"
# from google.colab import drive
# drive.mount('/content/drive')
# df = pd.read_csv(path)

# ÖRNEK VERİ OLUŞTURMA (Kodun bu kısımda çalışabilmesi için)
try:
    path = "/content/drive/MyDrive/Makine_öğrenmesi/NASA/işlenmiş TOI.csv"
    from google.colab import drive
    drive.mount('/content/drive')
    df = pd.read_csv(path)
except:
    # Gerçek veri yoksa örnek bir DataFrame oluştur
    print("ÖRNEK VERİ SETİ OLUŞTURULUYOR: Gerçek veri yolu bulunamadı.")
    data = {
        'rowid': range(100), 'toi': range(1, 101), 'toipfx': ['T'] * 100,
        'tid': range(1000, 1100), 'ctoi_alias': [f'C{i}' for i in range(100)],
        'rastr': ['10h'] * 100, 'decstr': ['+30d'] * 100,
        'tfopwg_disp': ['FP'] * 50 + ['PLANET'] * 50,
        'pl_trandep': np.random.rand(100), 'pl_trandurh': np.random.rand(100) * 10,
        'pl_orbper': np.random.rand(100) * 100, 'pl_insol': np.random.rand(100) * 5,
        'st_rad': np.random.rand(100), 'st_teff': np.random.rand(100) * 5000 + 3000,
        # Diğer sayısal kolonlar
        'feature1': np.random.rand(100), 'feature2': np.random.randn(100)
    }
    df = pd.DataFrame(data)
    df.loc[::10, ['pl_trandep', 'pl_trandurh']] = np.nan # Eksik değerler ekle

display(df.head())
display(df.columns)

id_cols = ['rowid','toi','toipfx','tid','ctoi_alias']
drop_text_cols = ['rastr','decstr']
target_col = 'tfopwg_disp'
assert target_col in df.columns, "tfopwg_disp yok!"
labels = df[target_col].astype(str).str.upper().str.strip()
y = np.where(labels == 'FP', 0, 1)
class_names = ['FP', 'PLANET']
print("Sınıf dağılımı (0=FP, 1=PLANET):", np.bincount(y))
X = df.drop(columns=[c for c in (id_cols + drop_text_cols + [target_col]) if c in df.columns], errors='ignore')
num_cols = X.select_dtypes(include=['number']).columns.tolist()
X = X[num_cols].copy()
print(f"Kullanılan {len(num_cols)} sayısal özellik.")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====================================================================
# BÖLÜM 2: İlk Model Eğitimi (Orijinal hali korunmuştur)
# ====================================================================

# Önişleme Adımları
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocess = ColumnTransformer(
    transformers=[('num', numeric_transformer, num_cols)],
    remainder='drop'
)

# Pipeline Tanımlama
smote = SMOTE(random_state=42, sampling_strategy='auto')
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
imb_pipe = ImbPipeline(steps=[
    ('prep', preprocess),
    ('smote', smote),
    ('model', rf)
])

# Eğitim ve SMOTE Kontrolü
imb_pipe.fit(X_train, y_train)
Xt_tr = preprocess.fit_transform(X_train, y_train)
Xs, ys = smote.fit_resample(Xt_tr, y_train)
unique, counts = np.unique(ys, return_counts=True)
print("SMOTE sonrası eğitimde dağılım:", dict(zip(unique, counts)))

# Değerlendirme
y_pred  = imb_pipe.predict(X_test)
y_proba = imb_pipe.predict_proba(X_test)[:, 1]
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(5,5))
disp.plot(ax=ax, values_format='d', colorbar=False)
ax.set_title("Confusion Matrix (Binary)")
plt.tight_layout(); plt.show()
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.3f}")
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve (PLANET = positive)"); plt.show()
ap = average_precision_score(y_test, y_proba)
print(f"Average Precision (AP): {ap:.3f}")
PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.title("Precision–Recall (PLANET = positive)"); plt.show()

# Eşik Optimizasyonu
ths = np.linspace(0.05, 0.95, 181)
best_t_pos, best_f1_pos = 0.5, -1
for t in ths:
    y_pred_t = (y_proba >= t).astype(int)
    f1t = f1_score(y_test, y_pred_t)
    if f1t > best_f1_pos:
        best_f1_pos, best_t_pos = f1t, t
print(f"[PLANET=1] En iyi threshold: {best_t_pos:.2f} | F1: {best_f1_pos:.3f}")
fp_proba = 1 - y_proba
best_t_fp, best_f1_fp = 0.5, -1
for t in ths:
    y_pred_fp = (fp_proba >= t).astype(int)
    f1t = f1_score((y_test==0).astype(int), y_pred_fp)
    if f1t > best_f1_fp:
        best_f1_fp, best_t_fp = f1t, t
print(f"[FP=1]      En iyi threshold: {best_t_fp:.2f} | F1(FP): {best_f1_fp:.3f}")

y_pred_best = (fp_proba >= best_t_fp).astype(int)
cm2 = confusion_matrix((y_test==0).astype(int), y_pred_best, labels=[0,1])
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=['Not-FP','FP'])
fig, ax = plt.subplots(figsize=(5,5))
disp2.plot(ax=ax, values_format='d', colorbar=False)
ax.set_title(f"Confusion Matrix @ threshold={best_t_fp:.2f} (FP=positive)")
plt.tight_layout(); plt.show()

# Çapraz Doğrulama (Cross-Validation)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
bal_acc = cross_val_score(imb_pipe, X, y, cv=cv,
                          scoring=make_scorer(balanced_accuracy_score), n_jobs=-1)
f1      = cross_val_score(imb_pipe, X, y, cv=cv, scoring='f1', n_jobs=-1)
auc     = cross_val_score(imb_pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print("CV Balanced Acc:", np.round(bal_acc,3), "| mean:", bal_acc.mean().round(3))
print("CV F1 (pos=PLANET):", np.round(f1,3),     "| mean:", f1.mean().round(3))
print("CV ROC-AUC:", np.round(auc,3),           "| mean:", auc.mean().round(3))

# Önem Skorları (Feature Importance)
X_test_tr = imb_pipe.named_steps['prep'].transform(X_test)
rf_fitted  = imb_pipe.named_steps['model']
feature_names = num_cols # Orijinal özellik adları
perm = permutation_importance(rf_fitted, X_test_tr, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_df = (pd.DataFrame({'feature': feature_names, 'perm_importance': perm.importances_mean})
             .sort_values('perm_importance', ascending=False).head(25))
print("\nPermutation Importances (Initial Model):")
display(perm_df)
imp_df = (pd.DataFrame({'feature': feature_names, 'importance': rf_fitted.feature_importances_})
             .sort_values('importance', ascending=False).head(25))
print("\nGini Importances (Initial Model):")
display(imp_df)
topk = 20
fig, ax = plt.subplots(figsize=(7,6))
ax.barh(perm_df['feature'].head(topk)[::-1], perm_df['perm_importance'].head(topk)[::-1])
ax.set_title("Permutation Importances (Top 20) - Initial")
plt.tight_layout(); plt.show()
fig, ax = plt.subplots(figsize=(7,6))
ax.barh(imp_df['feature'].head(topk)[::-1], imp_df['importance'].head(topk)[::-1])
ax.set_title("RandomForest Gini Importances (Top 20) - Initial")
plt.tight_layout(); plt.show()

# ====================================================================
# BÖLÜM 3: Özellik Mühendisliği (Feature Engineering)
# ====================================================================

df_fe = df.copy()
if 'pl_trandep' in df_fe.columns:
    df_fe['log_trandep'] = np.log1p(df_fe['pl_trandep'])
if 'pl_trandurh' in df_fe.columns and 'pl_orbper' in df_fe.columns:
    df_fe['dur_per_ratio'] = df_fe['pl_trandurh'] / (df_fe['pl_orbper'] + 1e-6)
if 'pl_insol' in df_fe.columns and 'st_rad' in df_fe.columns:
    df_fe['insol_rad2'] = df_fe['pl_insol'] / (df_fe['st_rad']**2 + 1e-6)
if 'st_teff' in df_fe.columns:
    df_fe['log_teff'] = np.log1p(df_fe['st_teff'])
print("\nYeni eklenen özellikler:", [c for c in df_fe.columns if c not in df.columns])

# YENİ VERİ SETİNİ OLUŞTUR
labels = df_fe['tfopwg_disp'].astype(str).str.upper().str.strip()
y = np.where(labels == 'FP', 0, 1)
class_names = ['FP','PLANET']
X = df_fe.drop(columns=[c for c in (id_cols + drop_text_cols + [target_col]) if c in df_fe.columns], errors='ignore')
num_cols = X.select_dtypes(include=['number']).columns.tolist()
X = X[num_cols].copy()
print("Yeni özellik sayısı:", len(num_cols))

# ====================================================================
# BÖLÜM 4: Hata Düzeltme ve Hiperparametre Optimizasyonu (RandomizedSearchCV)
# ====================================================================

# DÜZELTME: Yeni num_cols kullanılarak preprocess nesnesi yeniden tanımlanmalıdır.
numeric_transformer_new = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocess_new = ColumnTransformer(
    transformers=[('num', numeric_transformer_new, num_cols)], # ARTIK YENİ num_cols KULLANILIYOR
    remainder='drop'
)
print("Yeni preprocess nesnesi, genişletilmiş özellik listesiyle oluşturuldu.")

# DÜZELTME: Yeni preprocess ile imb_pipe yeniden tanımlanmalıdır.
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
param_dist = {
    'model__n_estimators': randint(200, 800),
    'model__max_depth': randint(3, 30),
    'model__min_samples_split': randint(2, 10),
    'model__min_samples_leaf': randint(1, 5),
    'model__max_features': ['sqrt', 'log2', None]
}
imb_pipe = ImbPipeline(steps=[
    ('prep', preprocess_new), # DÜZELTİLMİŞ PREPROCESS KULLANILIYOR
    ('smote', SMOTE(random_state=42)),
    ('model', rf)
])

rs = RandomizedSearchCV(
    imb_pipe,
    param_distributions=param_dist,
    n_iter=30,
    scoring='f1',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Arama işlemini yeni, genişletilmiş X ve y üzerinde gerçekleştir
rs.fit(X, y)

print("En iyi parametreler:", rs.best_params_)
print("En iyi CV skoru (F1):", rs.best_score_)

# ====================================================================
# BÖLÜM 5: En İyi Modelin Değerlendirilmesi
# ====================================================================

best_model = rs.best_estimator_

# En iyi model için train/test seti yeniden ayrılır (yeni X ve y kullanılır)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# En iyi model, yeniden ayrılmış eğitim verisine fit edilir
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:,1]

print("\n=== Final Model Classification Report ===")
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))
print("ROC-AUC:", roc_auc_score(y_test, y_proba).round(3))

# Permutation Importance (Yeni Model)
rf_fitted = best_model.named_steps['model']
X_test_tr = best_model.named_steps['prep'].transform(X_test)

# feature_names_out listesinden 'num__' öneki temizlenir.
feature_names_out = best_model.named_steps['prep'].get_feature_names_out()
# Sadece özellik ismini almak için num__ ön eki silinir.
feature_names_for_plot = [name.split('__')[1] for name in feature_names_out] 

perm = permutation_importance(rf_fitted, X_test_tr, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({'feature': feature_names_for_plot, 'importance': perm.importances_mean}) \
             .sort_values('importance', ascending=False).head(20)

print("\nPermutation Importances (Final Model):")
display(perm_df)

perm_df.plot.barh(x='feature', y='importance', figsize=(8,6), legend=False)
plt.title("Permutation Importances (Top 20) - Final Model")
plt.tight_layout(); plt.show()
