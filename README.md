from google.colab import drive
import shutil, os

drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive"
os.chdir("/content")

for fname in ["train_guide.csv", "test_guide.csv", "building_info_guide.csv", "sample_submission guide.csv"]:
    src = os.path.join(DATA_DIR, fname)
    dst = os.path.join("/content", fname)
    shutil.copy2(src, dst)
    print(f"[COPIED] {src} -> {dst}")

!pip -q install xgboost optuna

import os, gc, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor

SEED = 42
np.random.seed(SEED)

PATH_TRAIN = "train_guide.csv"
PATH_TEST  = "test_guide.csv"
PATH_INFO  = "building_info_guide.csv"
PATH_SUB_CANDIDATES = ["sample_submission_guide.csv", "sample_submission guide.csv"]


# 설정 (가중치/이상치/포리에/CDH)
PEAK_HOURS = list(range(9, 19))   # 9~18시 inclusive
PEAK_WEIGHT = 2.0

ROLL_WIN_H = 24
MAD_K = 5.0
RUN_INTERP_MAX = 2
RUN_DROP_MIN  = 3

DAILY_K  = 3
WEEKLY_K = 2

CDH_BASE_TEMP = 26
CDH_WINDOW_H  = 12

TARGET = "전력소비량(kWh)"


# 유틸
def to_datetime_yyyymmdd_hh(s):
    s = s.astype(str).str.strip()
    y = s.str.slice(0,4).astype(int)
    m = s.str.slice(4,6).astype(int)
    d = s.str.slice(8,10).astype(int) if ':' in s.iloc[0] else s.str.slice(6,8).astype(int)
    hh = s.str.slice(9,11).astype(int) if ':' not in s.iloc[0] else s.str.slice(11,13).astype(int)
    return pd.to_datetime(dict(year=y, month=m, day=d, hour=hh))

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    out = 2.0 * np.abs(y_pred - y_true) / np.maximum(denom, 1e-9)
    return 100.0 * np.mean(out)

def add_time_features(df, ts_col="일시_dt"):
    df["hour"] = df[ts_col].dt.hour
    df["weekday"] = df[ts_col].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["sin_hour"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["cos_hour"] = np.cos(2*np.pi*df["hour"]/24.0)
    return df

def add_periodic_fourier(df, ts_col="일시_dt",
                         daily_period=24, daily_K=3,
                         weekly_period=24*7, weekly_K=2):
    hour = df[ts_col].dt.hour
    dow = df[ts_col].dt.weekday
    hour_of_week = (dow * 24 + hour).astype(int)  # 0~167
    for k in range(1, daily_K+1):
        df[f"fourier_day_sin_k{k}"] = np.sin(2*np.pi*k*hour/daily_period)
        df[f"fourier_day_cos_k{k}"] = np.cos(2*np.pi*k*hour/daily_period)
    for k in range(1, weekly_K+1):
        df[f"fourier_week_sin_k{k}"] = np.sin(2*np.pi*k*hour_of_week/weekly_period)
        df[f"fourier_week_cos_k{k}"] = np.cos(2*np.pi*k*hour_of_week/weekly_period)
    df["hour_of_week"] = hour_of_week
    return df

def stull_wet_bulb(Ta, RH):
    Ta = np.asarray(Ta, dtype=float)
    RH = np.asarray(RH, dtype=float)
    return (Ta*np.arctan(0.151977*np.sqrt(RH+8.313659))
            + np.arctan(Ta+RH)
            - np.arctan(RH-1.67633)
            + 0.00391838*(RH**1.5)*np.arctan(0.023101*RH)
            - 4.686035)

def heat_index_from_Ta_RH(Ta, RH):
    Tw = stull_wet_bulb(Ta, RH)
    return (-0.2442 + 0.55399*Tw + 0.45535*Ta - 0.0022*(Tw**2) + 0.00278*(Tw*Ta) + 3.0)

def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -------- 이상치 처리 --------
def _mad(series):
    med = series.median()
    return np.median(np.abs(series - med))

def detect_outlier_mask(y, window=24, k=5.0):
    s = pd.Series(y).astype(float)
    roll_med = s.rolling(window=window, center=True, min_periods=max(3, window//3)).median()
    resid = s - roll_med
    roll_mad = resid.rolling(window=window, center=True, min_periods=max(3, window//3)).apply(_mad, raw=False)
    global_mad = _mad(resid.dropna()) or 1e-6
    roll_mad = roll_mad.fillna(global_mad)
    z = np.abs(resid) / (1.4826 * roll_mad + 1e-6)
    return (z > k).values

def split_runs(bool_mask):
    idx = np.where(bool_mask)[0]
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = i
            prev = i
    runs.append((start, prev))
    return runs

def repair_or_drop_block(series, start, end, strategy="interp"):
    s = series.copy()
    if strategy == "interp":
        s.iloc[start:end+1] = np.nan
        s = s.interpolate(method="linear", limit_direction="both")
    return s

def clean_outliers_per_building(df, bid_col, ycol, ts_col,
                               roll_win=24, k=5.0,
                               run_interp_max=2, run_drop_min=3):
    out_frames, drop_indices = [], []
    for bid, g in df.sort_values(ts_col).groupby(bid_col, sort=False):
        g = g.copy()
        y = g[ycol].astype(float).values
        mask = detect_outlier_mask(y, window=roll_win, k=k)
        runs = split_runs(mask)
        if not runs:
            out_frames.append(g); continue
        s = g[ycol].astype(float)
        for (st, ed) in runs:
            length = ed - st + 1
            if length <= run_interp_max:
                s = repair_or_drop_block(s, st, ed, strategy="interp")
            elif length >= run_drop_min:
                drop_indices.extend(g.iloc[st:ed+1].index.tolist())
            else:
                s = repair_or_drop_block(s, st, ed, strategy="interp")
        g[ycol] = s.values
        out_frames.append(g)
    out_df = pd.concat(out_frames, axis=0)
    if drop_indices:
        out_df = out_df.drop(index=drop_indices)
    return out_df.reset_index(drop=True)

# -------- CDH(누적 냉방부하)
def compute_cdh_array(xs, base_temp=26, window=12):
    ys = []
    for i in range(len(xs)):
        if i < window - 1:
            window_vals = xs[:(i+1)]
        else:
            window_vals = xs[(i - window + 1):(i + 1)]
        excess = np.maximum(0, window_vals - base_temp)
        ys.append(np.sum(excess))
    return np.array(ys)

def add_cdh_indicator(df, temp_col='기온(°C)', group_col='건물번호', base_temp=26, window=12):
    df = df.copy()
    def apply_cdh(group):
        temps = group[temp_col].values
        cdh_vals = compute_cdh_array(temps, base_temp=base_temp, window=window)
        return pd.Series(cdh_vals, index=group.index)
    df['CDH'] = df.groupby(group_col).apply(apply_cdh).reset_index(level=0, drop=True)
    return df


# 1)Load
train = pd.read_csv(PATH_TRAIN)
test  = pd.read_csv(PATH_TEST)
info  = pd.read_csv(PATH_INFO)

sub = None
for cand in PATH_SUB_CANDIDATES:
    if os.path.exists(cand):
        sub = pd.read_csv(cand); print(f"[INFO] Loaded submission template: {cand}"); break
if sub is None:
    raise FileNotFoundError("sample_submission 파일명을 확인하세요: sample_submission_guide.csv 또는 sample_submission guide.csv")


# 2) Preprocess
info = info.replace("-", 0)
info = safe_numeric(info, ["연면적(m2)", "냉방면적(m2)", "태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"])

train["일시_dt"] = to_datetime_yyyymmdd_hh(train["일시"])
test["일시_dt"]  = to_datetime_yyyymmdd_hh(test["일시"])

# join by 건물번호
train = train.merge(info, on="건물번호", how="left")
test  = test.merge(info, on="건물번호", how="left")


# 3) 이상치 처리 (train만)
train = clean_outliers_per_building(
    df=train,
    bid_col="건물번호", ycol=TARGET, ts_col="일시_dt",
    roll_win=ROLL_WIN_H, k=MAD_K,
    run_interp_max=RUN_INTERP_MAX, run_drop_min=RUN_DROP_MIN
)


# 4) Feature engineering

# 시간 파생 + Fourier
train = add_time_features(train, "일시_dt")
test  = add_time_features(test, "일시_dt")
train = add_periodic_fourier(train, "일시_dt", daily_K=DAILY_K, weekly_K=WEEKLY_K)
test  = add_periodic_fourier(test,  "일시_dt", daily_K=DAILY_K, weekly_K=WEEKLY_K)

# 체감온도
def heat_index_from_Ta_RH(Ta, RH):
    Tw = stull_wet_bulb(Ta, RH)
    return (-0.2442 + 0.55399*Tw + 0.45535*Ta - 0.0022*(Tw**2) + 0.00278*(Tw*Ta) + 3.0)

def stull_wet_bulb(Ta, RH):
    Ta = np.asarray(Ta, dtype=float)
    RH = np.asarray(RH, dtype=float)
    return (Ta*np.arctan(0.151977*np.sqrt(RH+8.313659))
            + np.arctan(Ta+RH)
            - np.arctan(RH-1.67633)
            + 0.00391838*(RH**1.5)*np.arctan(0.023101*RH)
            - 4.686035)

train["체감온도"] = heat_index_from_Ta_RH(train["기온(°C)"], train["습도(%)"])
test["체감온도"]  = heat_index_from_Ta_RH(test["기온(°C)"],  test["습도(%)"])

# --- CDH(누적 냉방부하) ---
train = add_cdh_indicator(train, temp_col='기온(°C)', group_col='건물번호',
                          base_temp=CDH_BASE_TEMP, window=CDH_WINDOW_H)
test  = add_cdh_indicator(test,  temp_col='기온(°C)', group_col='건물번호',
                          base_temp=CDH_BASE_TEMP, window=CDH_WINDOW_H)

# --- 집단통계 ---
grp_type_hour = (train
                 .groupby(["건물유형","hour"])[TARGET]
                 .agg(type_hour_mean="mean", type_hour_std="std")
                 .reset_index())
train = train.merge(grp_type_hour, on=["건물유형","hour"], how="left")
test  = test.merge(grp_type_hour,  on=["건물유형","hour"], how="left")

grp_type_wkd_hour = (train
                     .groupby(["건물유형","is_weekend","hour"])[TARGET]
                     .agg(type_wkd_hour_mean="mean", type_wkd_hour_std="std")
                     .reset_index())
train = train.merge(grp_type_wkd_hour, on=["건물유형","is_weekend","hour"], how="left")
test  = test.merge(grp_type_wkd_hour,  on=["건물유형","is_weekend","hour"], how="left")

grp_bld = (train.groupby("건물번호")[TARGET]
           .agg(bld_mean="mean", bld_std="std")
           .reset_index())
train = train.merge(grp_bld, on="건물번호", how="left")
test  = test.merge(grp_bld,  on="건물번호", how="left")

# 집단통계 결측 보정
global_mean = train[TARGET].mean()
for col in ["type_hour_mean", "type_wkd_hour_mean", "bld_mean"]:
    train[col] = train[col].fillna(global_mean)
    test[col]  = test[col].fillna(global_mean)
for col in ["type_hour_std", "type_wkd_hour_std", "bld_std"]:
    train[col] = train[col].fillna(0.0)
    test[col]  = test[col].fillna(0.0)

# --- 교호작용: is_weekend × 건물유형(원핫)
type_dummies = pd.get_dummies(train["건물유형"], prefix="type")
type_cols = type_dummies.columns.tolist()
train[type_cols] = type_dummies
test[type_cols]  = pd.get_dummies(test["건물유형"], prefix="type").reindex(columns=type_cols, fill_value=0)
for c in type_cols:
    inter_col = f"{c}_x_weekend"
    train[inter_col] = train[c] * train["is_weekend"]
    test[inter_col]  = test[c]  * test["is_weekend"]


# 5) Column curation (+ 앙상블용 메타 보존)
meta_train = train[["건물번호","건물유형","일시_dt"]].copy()
meta_test  = test[["건물번호","건물유형","일시_dt"]].copy()

drop_cols = [
    "일시","건물번호","num_date_time",
    "강수량(mm)","일조(hr)","일사(MJ/m2)",
    "태양광용량(kW)","ESS저장용량(kWh)","PCS용량(kW)",
    "건물유형"
]
if "요일" in train.columns: drop_cols.append("요일")
if "요일" in test.columns:  drop_cols.append("요일")

train = train.drop(columns=[c for c in drop_cols if c in train.columns])
test  = test.drop(columns=[c for c in drop_cols if c in test.columns])

y = train[TARGET].values
X = train.drop(columns=[TARGET])

obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
if obj_cols:
    print("[WARN] object cols dropped:", obj_cols)
    X = X.drop(columns=obj_cols)
    test = test.drop(columns=[c for c in obj_cols if c in test.columns])

# 6) 홀드아웃 분할
VAL_START = pd.to_datetime("2024-08-18 00:00:00")
VAL_END   = pd.to_datetime("2024-08-24 23:00:00")

mask_tr  = (meta_train["일시_dt"] < VAL_START).values
mask_val = ((meta_train["일시_dt"] >= VAL_START) & (meta_train["일시_dt"] <= VAL_END)).values

def ensure_no_dt(df):
    return df.drop(columns=[c for c in ("일시_dt",) if c in df.columns], errors="ignore")

X_tr_glob   = ensure_no_dt(X.loc[mask_tr])
y_tr_glob   = y[mask_tr]
X_val_glob  = ensure_no_dt(X.loc[mask_val])
y_val_glob  = y[mask_val]

hours_tr = meta_train.loc[mask_tr, "일시_dt"].dt.hour.values
w_tr = np.where(np.isin(hours_tr, PEAK_HOURS), PEAK_WEIGHT, 1.0)

print(f"Shapes | X_train: {X_tr_glob.shape}  X_val: {X_val_glob.shape}")

# 7) Optuna: XGB 하이퍼파라미터
def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
    }
    model = XGBRegressor(**params)
    model.fit(X_tr_glob, y_tr_glob, sample_weight=w_tr)
    pred = model.predict(X_val_glob)
    return smape(y_val_glob, pred)

N_TRIALS_XGB = 8  # << 아주 작게
print("Tuning XGBoost (small trials)...")
study_xgb = optuna.create_study(direction="minimize", study_name="xgb_vert_smape")
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS_XGB, show_progress_bar=True)
print("Best XGB:", study_xgb.best_value, study_xgb.best_params)

best_params = dict(study_xgb.best_params)
best_params.update({"tree_method":"hist", "random_state":SEED, "n_jobs":-1, "verbosity":0})

def fit_xgb(X_tr, y_tr, w=None, xgb_params=None):
    p = best_params if xgb_params is None else xgb_params
    m = XGBRegressor(**p)
    m.fit(X_tr, y_tr, sample_weight=w)
    return m

# 8) 수직 앙상블(전역/유형/건물)

# 전역
global_model = fit_xgb(X_tr_glob, y_tr_glob, w=w_tr)
val_pred_global = global_model.predict(X_val_glob)

# 유형별
type_tr  = meta_train.loc[mask_tr,  "건물유형"].values
type_val = meta_train.loc[mask_val, "건물유형"].values
type_models = {}
val_pred_type = np.zeros_like(y_val_glob, dtype=float)

for t in np.unique(type_tr):
    idx_tr_t = (type_tr == t)
    if idx_tr_t.sum() < 50:  # 표본 적으면 스킵
        continue
    Xm = X_tr_glob.loc[idx_tr_t]
    ym = y_tr_glob[idx_tr_t]
    wm = w_tr[idx_tr_t]
    m = fit_xgb(Xm, ym, wm)
    type_models[t] = m
    idx_val_t = (type_val == t)
    if idx_val_t.sum() > 0:
        val_pred_type[idx_val_t] = m.predict(X_val_glob.loc[idx_val_t])

fallback = (val_pred_type == 0)
if fallback.any():
    val_pred_type[fallback] = val_pred_global[fallback]

# 건물별
bid_tr  = meta_train.loc[mask_tr,  "건물번호"].values
bid_val = meta_train.loc[mask_val, "건물번호"].values
bld_models = {}
val_pred_bld = np.zeros_like(y_val_glob, dtype=float)

for bid in np.unique(bid_tr):
    idx_tr_b = (bid_tr == bid)
    if idx_tr_b.sum() < 24:
        continue
    Xm = X_tr_glob.loc[idx_tr_b]
    ym = y_tr_glob[idx_tr_b]
    wm = w_tr[idx_tr_b]
    m = fit_xgb(Xm, ym, wm)
    bld_models[bid] = m
    idx_val_b = (bid_val == bid)
    if idx_val_b.sum() > 0:
        val_pred_bld[idx_val_b] = m.predict(X_val_glob.loc[idx_val_b])

fallback = (val_pred_bld == 0)
if fallback.any():
    val_pred_bld[fallback] = val_pred_global[fallback]

# 9) Optuna: (전역/유형/건물) 가중치 최적화

stack_val = np.vstack([val_pred_global, val_pred_type, val_pred_bld])  # (3, N)

def objective_w(trial):
    a = trial.suggest_float("a", -2.0, 2.0)
    b = trial.suggest_float("b", -2.0, 2.0)
    c = trial.suggest_float("c", -2.0, 2.0)
    w = np.exp([a,b,c]); w = w/np.sum(w)
    pred = w[0]*stack_val[0] + w[1]*stack_val[1] + w[2]*stack_val[2]
    return smape(y_val_glob, pred)

N_TRIALS_W = 20  # << 아주 작게
print("Tuning blend weights (small trials)...")
study_w = optuna.create_study(direction="minimize", study_name="blend_w_smape")
study_w.optimize(objective_w, n_trials=N_TRIALS_W, show_progress_bar=True)

aw, bw, cw = study_w.best_params["a"], study_w.best_params["b"], study_w.best_params["c"]
W = np.exp(np.array([aw,bw,cw])); W = W / W.sum()
WG, WT, WB = W.tolist()
val_pred_ens = WG*val_pred_global + WT*val_pred_type + WB*val_pred_bld
print(f"[VAL] Best SMAPE: {smape(y_val_glob, val_pred_ens):.4f}  | Weights (G/T/B): {WG:.3f}/{WT:.3f}/{WB:.3f}")

# 10) 테스트 예측 (풀데이터 재학습)
X_full_glob = ensure_no_dt(X)
hours_full  = meta_train["일시_dt"].dt.hour.values
w_full      = np.where(np.isin(hours_full, PEAK_HOURS), PEAK_WEIGHT, 1.0)
global_model_full = fit_xgb(X_full_glob, y, w=w_full)

X_test_glob = ensure_no_dt(test.copy())
pred_global_test = global_model_full.predict(X_test_glob)

# 유형별 풀데이터
type_full = meta_train["건물유형"].values
pred_type_test = np.zeros_like(pred_global_test, dtype=float)
for t in np.unique(type_full):
    idx_full_t = (type_full == t)
    if idx_full_t.sum() < 50:
        continue
    Xm = X_full_glob.loc[idx_full_t]
    ym = y[idx_full_t]
    wm = w_full[idx_full_t]
    m = fit_xgb(Xm, ym, wm)
    idx_test_t = (meta_test["건물유형"].values == t)
    if idx_test_t.sum() > 0:
        pred_type_test[idx_test_t] = m.predict(X_test_glob.loc[idx_test_t])

fallback = (pred_type_test == 0)
if fallback.any():
    pred_type_test[fallback] = pred_global_test[fallback]

# 건물별 풀데이터
bid_full = meta_train["건물번호"].values
pred_bld_test = np.zeros_like(pred_global_test, dtype=float)
for bid in np.unique(bid_full):
    idx_full_b = (bid_full == bid)
    if idx_full_b.sum() < 24:
        continue
    Xm = X_full_glob.loc[idx_full_b]
    ym = y[idx_full_b]
    wm = w_full[idx_full_b]
    m = fit_xgb(Xm, ym, wm)
    idx_test_b = (meta_test["건물번호"].values == bid)
    if idx_test_b.sum() > 0:
        pred_bld_test[idx_test_b] = m.predict(X_test_glob.loc[idx_test_b])

fallback = (pred_bld_test == 0)
if fallback.any():
    pred_bld_test[fallback] = pred_global_test[fallback]

# 최종 블렌딩
test_pred_ens = WG*pred_global_test + WT*pred_type_test + WB*pred_bld_test

# 11) 제출 파일 생성
submission = sub.copy()
if "answer" in submission.columns:
    submission["answer"] = test_pred_ens
else:
    pred_col = submission.columns[-1]
    submission[pred_col] = test_pred_ens

OUT_PATH = "./submission_vert_xgb_optuna.csv"
submission.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(submission.head())
print(f"[Saved] {OUT_PATH}")

# 메모리 정리
del X_tr_glob, y_tr_glob, X_val_glob, y_val_glob, global_model, global_model_full
gc.collect()

