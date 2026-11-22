# %%
# =============================================================================
# LightGBM 梯度提升决策树建模与评估
# (LightGBM Modeling, Gain Selection & Optuna Optimization)
# =============================================================================
# 模块功能：
# 实现基于 LightGBM 的 GBDT 提升树建模流程：
# 1. 特征类型适配（保留连续值与缺失分支，类别特征转为 category 原生支持）；
# 2. 特征初筛（同值率 > 95% 过滤，Spearman |r| >= 0.95 宽容度去共线）；
# 3. 5 折交叉验证 OOF 计算分裂增益，保留累计增益前 95% 的特征；
# 4. Optuna 贝叶斯超参数调优（优化树深度、叶子数与 L1/L2 正则项）；
# 5. 5 折模型集成与评估（训练集 OOF、验证集评估及测试集预测概率 PSI）。
# =============================================================================

# %%
import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import toad
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

# 文件路径配置
DATA_PATH = Path.cwd() / "data"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
OUTPUTS_PATH = Path.cwd() / "outputs"
MODEL_OUTPUT_PATH = OUTPUTS_PATH / "models"
MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# 1. 数据集读取与特征类型转换（直接对接统一特征工程全量宽表）
# =============================================================================
print("正在加载统一特征工程全量宽表...")
train = pd.read_csv(PROCESSED_DATA_PATH / 'train_features_full.csv')
valid = pd.read_csv(PROCESSED_DATA_PATH / 'valid_features_full.csv')
test = pd.read_csv(PROCESSED_DATA_PATH / 'test_features_full.csv')

y_train = train['TARGET'].copy()
y_valid = valid['TARGET'].copy()

ignore_cols = ['SK_ID_CURR', 'TARGET']
all_feature_cols = [c for c in train.columns if c not in ignore_cols]

# 统一类别特征为 category 类型，适配 LightGBM 原生接口
cat_cols = train[all_feature_cols].select_dtypes(exclude='number').columns.tolist()
for col in cat_cols:
    train[col] = train[col].astype('category')
    valid[col] = pd.Categorical(valid[col], categories=train[col].cat.categories)
    test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)

print(f"数据加载完成: 训练集 {train.shape}, 验证集 {valid.shape}, 测试集 {test.shape}")
print(f"初始候选特征数: {len(all_feature_cols)} (其中类别型: {len(cat_cols)}, 数值型: {len(all_feature_cols) - len(cat_cols)})")

# %%
# =============================================================================
# 2. 特征初筛（同值率 > 0.95 过滤，Spearman 宽阈值去共线）
# =============================================================================
# (1) 剔除极高同值特征（同值率 > 0.95）
primary_value_ratios = pd.Series({
    col: train[col].value_counts(dropna=False).iloc[0] / len(train)
    for col in all_feature_cols
})
high_pvr_cols = primary_value_ratios[primary_value_ratios > 0.95].index.tolist()

# (2) 相关性初筛：Spearman 阈值设为 0.95（树模型抗共线性较强，仅剔除极高度相关变量）
orig_num_cols = [c for c in all_feature_cols if c not in cat_cols and c not in high_pvr_cols]
features_corr = train[orig_num_cols].corr(method='spearman')

result = features_corr.stack().reset_index()
result.columns = ['feature1', 'feature2', 'corr']
result = result[np.abs(result['corr']).between(0.95, 1, inclusive='left')]

# 优先保留训练集缺失率更低的特征
missing_dict = train[orig_num_cols].isna().mean().to_dict()
result['missing1'] = result['feature1'].map(missing_dict)
result['missing2'] = result['feature2'].map(missing_dict)
high_corr = result.loc[
    (result['missing1'] > result['missing2']) | 
    ((result['missing1'] == result['missing2']) & (result['feature1'] > result['feature2'])),
    'feature1'
].tolist()

delete_features = set(high_pvr_cols + high_corr)
candidate_feature_cols = [c for c in all_feature_cols if c not in delete_features]

print(f"✓ 粗筛完成: 剔除同值率>95%字段 {len(high_pvr_cols)} 个、极端共线(|r|>=0.95)字段 {len(set(high_corr))} 个")
print(f"  剩余供增益初筛的候选特征数: {len(candidate_feature_cols)}")

del high_pvr_cols, high_corr, result, features_corr, orig_num_cols, delete_features, primary_value_ratios, missing_dict
gc.collect()

# %%
# =============================================================================
# 3. 5 折交叉验证与累计分裂增益筛选 (Gain Importance Top 95%)
# =============================================================================
X = train[candidate_feature_cols]
y = y_train

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

baseline_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'max_depth': 5,
    'num_leaves': 16,
    'min_child_samples': 50,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': 123,
    'importance_type': 'gain',
    'n_jobs': -1,
    'verbose': -1
}

print("\n正在运行 5 折交叉验证基准模型计算各特征分裂增益 (Gain Importance)...")
feature_importance = np.zeros(X.shape[1])
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[valid_idx], y.iloc[valid_idx]

    clf = lgb.LGBMClassifier(**baseline_params)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    feature_importance += clf.feature_importances_ / skf.n_splits

importance_df = pd.DataFrame({
    'feature': candidate_feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False).reset_index(drop=True)

# 计算累计分裂增益百分比
importance_df['cum_importance'] = (importance_df['importance'] / importance_df['importance'].sum()).cumsum()

# 筛选累计贡献前 95% 的核心特征
selected_feature_cols = importance_df.loc[
    importance_df['cum_importance'] <= 0.95, 'feature'
].tolist()

# 兜底：若刚好截止项小于 95%，补充下一项使其达到或稍超过 95%
if len(selected_feature_cols) < len(importance_df):
    next_feature = importance_df.loc[len(selected_feature_cols), 'feature']
    if next_feature not in selected_feature_cols:
        selected_feature_cols.append(next_feature)

importance_df.to_csv(MODEL_OUTPUT_PATH / 'lgb_feature_importance.csv', index=False)
print(f"✓ 分裂增益精筛完成！累计增益前 95% 特征数: {len(selected_feature_cols)} (原始候选: {len(candidate_feature_cols)})")
print(f"  增益特征重要性榜单已落盘至: {MODEL_OUTPUT_PATH / 'lgb_feature_importance.csv'}")

del X, feature_importance, baseline_params
gc.collect()

# %%
# =============================================================================
# 4. Optuna 贝叶斯超参数优化（树结构与正则化参数调优）
# =============================================================================
X_final = train[selected_feature_cols].copy()

def objective(trial):
    max_depth = trial.suggest_int('max_depth', 3, 5)
    num_leaves_raw = trial.suggest_int('num_leaves', 4, 30)
    # 限制 num_leaves 小于 2^max_depth - 1，避免满二叉树过拟合
    num_leaves = min(num_leaves_raw, (2 ** max_depth) - 2)
    
    subsample = trial.suggest_categorical('subsample', [0.8, 0.9, 1.0])
    colsample_bytree = trial.suggest_categorical('colsample_bytree', [0.8, 0.9, 1.0])
    reg_alpha = trial.suggest_float('reg_alpha', 1.0, 1000.0, log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 1.0, 1000.0, log=True)

    cv_scores = []
    for train_idx, valid_idx in skf.split(X_final, y):
        X_tr, y_tr = X_final.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_final.iloc[valid_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            learning_rate=0.05,
            n_estimators=1000,
            min_child_samples=50,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            subsample_freq=1,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=123,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        val_preds = model.predict_proba(X_va)[:, 1]
        cv_scores.append(roc_auc_score(y_va, val_preds))

    return np.mean(cv_scores)

print("\n正在启动 Optuna 进行 30 次 TPE 贝叶斯超参数调优...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=123))
study.optimize(objective, n_trials=30, timeout=1800, show_progress_bar=False)

best_depth = study.best_params['max_depth']
best_leaves = min(study.best_params['num_leaves'], (2 ** best_depth) - 2)

print(f"✓ Optuna 搜索完成！最佳 5 折 CV AUC: {study.best_value:.5f}")
print("  最佳超参数组合:")
print(f"    - max_depth: {best_depth}")
print(f"    - num_leaves: {best_leaves}")
print(f"    - subsample: {study.best_params['subsample']}")
print(f"    - colsample_bytree: {study.best_params['colsample_bytree']}")
print(f"    - reg_alpha: {study.best_params['reg_alpha']:.4f}")
print(f"    - reg_lambda: {study.best_params['reg_lambda']:.4f}")

del objective
gc.collect()

# %%
# =============================================================================
# 5. 最优超参数 5 折集成训练与推理
# =============================================================================
final_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'min_child_samples': 50,
    'max_depth': best_depth,
    'num_leaves': best_leaves,
    'subsample': study.best_params['subsample'],
    'subsample_freq': 1,
    'colsample_bytree': study.best_params['colsample_bytree'],
    'reg_alpha': study.best_params['reg_alpha'],
    'reg_lambda': study.best_params['reg_lambda'],
    'random_state': 123,
    'n_jobs': -1,
    'verbose': -1
}

print(f"\n正在使用最优超参数在精选的 {len(selected_feature_cols)} 个核心特征上执行 5 折完整集成训练...")

X_valid_final = valid[selected_feature_cols].copy()
X_test_final = test[selected_feature_cols].copy()

oof_preds = np.zeros(len(train))
valid_preds = np.zeros(len(valid))
test_preds = np.zeros(len(test))

def calc_ks(y_true, y_prob):
    """计算 KS (Kolmogorov-Smirnov) 统计量"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return np.max(np.abs(tpr - fpr))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_final, y), 1):
    X_tr, y_tr = X_final.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X_final.iloc[valid_idx], y.iloc[valid_idx]

    clf = lgb.LGBMClassifier(**final_params)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    val_pred = clf.predict_proba(X_va)[:, 1]
    oof_preds[valid_idx] = val_pred

    # 累加验证集与测试集预测概率 (5折集成)
    valid_preds += clf.predict_proba(X_valid_final)[:, 1] / skf.n_splits
    test_preds += clf.predict_proba(X_test_final)[:, 1] / skf.n_splits

    fold_auc = roc_auc_score(y_va, val_pred)
    fold_ks = calc_ks(y_va, val_pred)
    print(f"  Fold {fold} | AUC: {fold_auc:.5f} | KS: {fold_ks:.5f}")

# (1) 训练集 5 折 OOF 评估
cv_auc = roc_auc_score(y, oof_preds)
cv_ks = calc_ks(y, oof_preds)

# (2) 验证集评估
valid_auc = roc_auc_score(y_valid, valid_preds)
valid_ks = calc_ks(y_valid, valid_preds)

# (3) 测试集跨客群 PSI 评估 (以训练集 OOF 预测概率为基准分箱)
prob_combiner = toad.transform.Combiner()
prob_combiner.fit(pd.Series(oof_preds, name='prob'), method='quantile', n_bins=10)
test_psi = toad.metrics.PSI(oof_preds, test_preds, combiner=prob_combiner)

# 汇总并保存评估报告
eval_summary_df = pd.DataFrame({
    '数据集': ['训练集 (Train 5-Fold OOF)', '验证集 (Valid)', '测试集 (Test)'],
    '样本量': [len(train), len(valid), len(test)],
    'AUC': [f"{cv_auc:.4f}", f"{valid_auc:.4f}", "-"],
    'KS': [f"{cv_ks:.4f}", f"{valid_ks:.4f}", "-"],
    '跨客群预测概率 PSI': ["基准 (0.00000)", "-", f"{test_psi:.5f}"]
})
eval_summary_df.to_csv(MODEL_OUTPUT_PATH / 'lgb_eval_summary.csv', index=False)

print("\n" + "=" * 70)
print("✓ [LightGBM 建模与评估完成 - 汇总账本]")
print(eval_summary_df.to_string(index=False))
print("=" * 70)

del train, valid, test, X_final, X_valid_final, X_test_final
del oof_preds, valid_preds, test_preds, prob_combiner, eval_summary_df
gc.collect()
