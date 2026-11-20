# %%
# =============================================================================
# LightGBM 梯度提升决策树建模与评估（LightGBM Modeling & Evaluation）
# =============================================================================
# 数据隔离与评估规范（严格遵循 00_stratified_split.py 约定）：
# 1. 开发训练集（Train Dev, 80%）：
#    读取 data/interim/application_train_dev.csv。作为单表高阶特征工程规则定义、
#    分组统计特征（如行业/职业均值）拟合、特征粗筛与 Gain 精筛、Optuna 超参数搜索、
#    5 折交叉验证训练及 OOF 评估的唯一开发输入。
# 2. 闭卷验证集（Valid Blind, 20%）：
#    读取 data/interim/application_valid_blind.csv。在建模完成前完全独立封存，
#    不参与任何规则拟合、特征筛选或超参数调优；仅使用 Train Dev 学习的规则执行
#    纯特征转换（transform）与 5 折模型集成推理，用于独立客观评估 AUC 与 KS。
# 3. 官方未标注测试集（Official Test, 4.8 万条）：
#    读取 data/raw/application_test.csv。无 TARGET 标签，直接输入训练好的模型输出预测分值，
#    用于评估跨客群模型预测概率总分 PSI，检验模型面对未知分布时的稳定性。
#    严禁在特征工程与筛选阶段利用测试集计算单特征 PSI 进行特征准入过滤（彻底杜绝数据泄露）。
# =============================================================================

# %%
import gc
import os
import sys
from pathlib import Path

# Windows 环境下确保当前虚拟环境的 Library/bin 优先加载，防止 LAPACK/OpenBLAS DLL 符号解析冲突
dll_path = os.path.join(sys.prefix, "Library", "bin")
if os.path.exists(dll_path):
    os.add_dll_directory(dll_path)
    os.environ["PATH"] = dll_path + os.pathsep + os.environ.get("PATH", "")

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import toad
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

# 文件路径配置
DATA_PATH = Path.cwd() / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
INTERIM_DATA_PATH = DATA_PATH / "interim"

OUTPUTS_PATH = Path.cwd() / "outputs"
EDA_OUTPUT_PATH = OUTPUTS_PATH / "eda"
MODEL_OUTPUT_PATH = OUTPUTS_PATH / "models"
MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# 1. 数据集读取与特殊清洗
# =============================================================================
# 开发训练集（唯一开发输入）
train_dev = pd.read_csv(INTERIM_DATA_PATH / 'application_train_dev.csv')

# 独立闭卷验证集（独立封存评估）
valid_blind = pd.read_csv(INTERIM_DATA_PATH / 'application_valid_blind.csv')

# 官方未标注测试集（跨客群分布参照）
application_test = pd.read_csv(RAW_DATA_PATH / 'application_test.csv')

cols_eda = pd.read_csv(EDA_OUTPUT_PATH / 'cols_eda.csv')

# 提取房屋相关列（标准化信息）
house_cols = cols_eda.loc[
    cols_eda['Description'].str.startswith('Normalized information'), 'Row'
].tolist()


def clean_anomalies(df):
    """异常值替换与标记（全表无参数变换）"""
    df['FLAG_DAYS_EMPLOYED_ANOMALOUS'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', 'F')
    if 'OBS_30_CNT_SOCIAL_CIRCLE' in df.columns:
        df.loc[df['OBS_30_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
    if 'OBS_60_CNT_SOCIAL_CIRCLE' in df.columns:
        df.loc[df['OBS_60_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_60_CNT_SOCIAL_CIRCLE'] = np.nan


# -- 高阶单表特征衍生 --
def add_advanced_features(df, house_cols_list):
    """针对申请主表的高阶特征工程（包含还款能力、期限代理、时间交叉、EXT外部评分深挖及质量缺失指标）"""
    # 1. 还款能力与负债压力
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['FREE_INCOME_PROXY'] = df['AMT_INCOME_TOTAL'] - df['AMT_ANNUITY']
    df['ANNUITY_TO_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['GOODS_TO_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    df['DOWN_PAYMENT_PROXY'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])

    # 2. 期限代理与利率成本
    df['ESTIMATED_TENURE'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['ANNUITY_TO_GOODS_RATIO'] = df['AMT_ANNUITY'] / df['AMT_GOODS_PRICE']

    # 3. 稳定性与时间交叉比率
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (-df['DAYS_BIRTH'] / 365)
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['REG_TO_BIRTH_RATIO'] = df['DAYS_REGISTRATION'] / df['DAYS_BIRTH']
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['PHONE_TO_EMPLOYED_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED'].replace(0, np.nan)
    df['AGE_AT_EMPLOYMENT'] = (df['DAYS_BIRTH'] - df['DAYS_EMPLOYED']) / 365
    df['REG_TO_ID_RATIO'] = df['DAYS_REGISTRATION'] / df['DAYS_ID_PUBLISH'].replace(0, np.nan)

    # 4. 外部征信评分深挖 (单表提分核心)
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_WEIGHTED'] = (
        df['EXT_SOURCE_1'] * 2 + df['EXT_SOURCE_2'] * 3 + df['EXT_SOURCE_3'] * 5
    )
    df['EXT_SOURCE_NAN_CNT'] = df[ext_cols].isna().sum(axis=1)
    df['EXT_MEAN_TO_BIRTH'] = df['EXT_SOURCE_MEAN'] / (-df['DAYS_BIRTH'] / 365)
    df['EXT_MEAN_TO_CREDIT_RATIO'] = df['EXT_SOURCE_MEAN'] / df['CREDIT_TO_INCOME_RATIO']

    # 5. 风险信号聚合与同源缺失标志位
    avg_house_cols = [col for col in house_cols_list if col.endswith('AVG') and col in df.columns]
    if avg_house_cols:
        df['HOUSE_INFO_MISSING_RATE'] = df[avg_house_cols].isna().mean(axis=1)

    doc_cols = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22) if f"FLAG_DOCUMENT_{i}" in df.columns]
    df['DOCUMENTS_PROVIDED_CNT'] = df[doc_cols].sum(axis=1)

    addr_cols = [
        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION'
    ]
    df['ADDRESS_MISMATCH_MEAN'] = df[addr_cols].mean(axis=1)
    df['DEF_OBS_RATIO_30'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)
    df['DEF_OBS_RATIO_60'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['OBS_60_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)

    # 整行总缺失特征计数（质量与配合度代理）
    df['TOTAL_NAN_COUNT'] = df.isna().sum(axis=1)


# 对三份数据集统一执行异常值清洗与高阶衍生
for df_obj in [train_dev, valid_blind, application_test]:
    clean_anomalies(df_obj)
    add_advanced_features(df_obj, house_cols)

# 6. 分组统计特征（同行业/同职业相对水平）：统计量仅在开发训练集上拟合，杜绝验证集与测试集穿越
for grp_col in ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']:
    grp_stats = train_dev.groupby(grp_col, observed=True)['AMT_INCOME_TOTAL'].agg(['mean']).reset_index()
    grp_stats = grp_stats.rename(columns={'mean': f'{grp_col}_INCOME_MEAN'})

    train_dev = train_dev.merge(grp_stats, on=grp_col, how='left')
    train_dev[f'INCOME_TO_{grp_col}_RATIO'] = (
        train_dev['AMT_INCOME_TOTAL'] / train_dev[f'{grp_col}_INCOME_MEAN']
    )

    valid_blind = valid_blind.merge(grp_stats, on=grp_col, how='left')
    valid_blind[f'INCOME_TO_{grp_col}_RATIO'] = (
        valid_blind['AMT_INCOME_TOTAL'] / valid_blind[f'{grp_col}_INCOME_MEAN']
    )

    application_test = application_test.merge(grp_stats, on=grp_col, how='left')
    application_test[f'INCOME_TO_{grp_col}_RATIO'] = (
        application_test['AMT_INCOME_TOTAL'] / application_test[f'{grp_col}_INCOME_MEAN']
    )

del grp_col, grp_stats, house_cols, clean_anomalies, add_advanced_features
gc.collect()

# %%
# =============================================================================
# 2. 特征筛选（基于开发训练集 Train Dev 开展粗筛与精筛）
# =============================================================================
ignore_cols = ['SK_ID_CURR', 'TARGET']
all_feature_cols = [c for c in train_dev.columns if c not in ignore_cols]
orig_num_cols = [
    c for c in all_feature_cols
    if c in train_dev.select_dtypes(include='number').columns and c not in ignore_cols
]

# -- 粗筛 --
# 1. 基础清洗：剔除极高同值特征（同值率 > 0.95）
high_primary_value_ratio = cols_eda.loc[
    cols_eda['primary_value_ratio'] > 0.95, 'Row'
].tolist()

derived_cols = [col for col in all_feature_cols if col not in cols_eda['Row'].tolist()]
for d_col in derived_cols:
    val_counts = train_dev[d_col].value_counts(normalize=True, dropna=False)
    if val_counts.iloc[0] > 0.95:
        high_primary_value_ratio.append(d_col)

del d_col, val_counts

# =============================================================================
# 2. 稳定性过滤（原 PSI > 0.25 特征初筛已彻底移除）
# -----------------------------------------------------------------------------
# 【数据泄露防范批注 / Data Leakage Prevention Note】:
# 原代码在此处利用测试集 (application_test) 计算了原始特征与衍生特征的 PSI 指标，
# 并基于 `high_psi` (PSI > 0.25) 进行了特征粗筛剔除。
# 现已彻底去除在特征筛选阶段对 PSI 的筛选，原因如下：
# 1. 严格禁止测试集数据泄露（Zero Data Leakage 原则）：
#    测试集用于评估模型在未知分布上的真实泛化能力，建模与特征工程阶段应将其完全视作“不可见数据”。
#    若在特征筛选时引入测试集计算单特征 PSI 并以此剔除漂移特征，属于无监督数据穿越（Unsupervised Data Leakage），
#    使模型构建过程受到未来/盲测数据分布的先验污染。
# 2. 避免对未知分布的过拟合与虚假高估稳定性：
#    针对特定测试集人工剔除漂移特征，会导致入模特征强行适配该测试集，使最终测试集评估出的
#    模型预测概率 PSI 和区分度（AUC/KS）呈现过于乐观的虚假繁荣，在实际生产部署面对真实客群漂移时
#    将产生严重的泛化性能崩溃。
# 3. PSI 的角色应严格限定于模型级事后评估与上线监控：
#    PSI 是模型交付后的整体健康度监控指标，应在模型最终训练完成并冻结后，通过评估模型输出的
#    违约概率/评分在测试集上的宏观分布变化来衡量模型稳定性，绝不应用于前置反向指导特征准入。
#    因此，此处不再执行基于测试集的特征 PSI 过滤。
# =============================================================================

# 3. 多重共线性：对数值特征采用 Spearman 相关系数衡量单调相关性，阈值设置为 0.95
num_candidates_for_corr = [
    col for col in orig_num_cols
    if col not in high_primary_value_ratio
]
features_corr = train_dev[num_candidates_for_corr].corr(method='spearman')

result = features_corr.stack().reset_index()
result.columns = ['feature1', 'feature2', 'corr']
result = result[np.abs(result['corr']).between(0.95, 1, inclusive='left')]

# 【数据泄露防范批注 / Data Leakage Prevention Note】:
# 原代码在消除多重共线性特征时，利用测试集计算的 psi1 与 psi2 进行比较 (result['psi1'] > result['psi2'])，
# 剔除在测试集上 PSI 较高的共线特征。这同样利用了测试集信息，构成数据泄露。
# 现已重构为纯粹基于开发训练集自身信息（缺失率）消除共线性：
# 针对高相关性特征对 (|r| >= 0.95)，优先保留开发集上缺失率更低（信息更完整）的特征；
# 若缺失率相同，按字段名称字典序做确定性剔除，彻底杜绝测试集信息渗透。
missing_dict = train_dev[orig_num_cols].isna().mean().to_dict()
result['missing1'] = result['feature1'].map(missing_dict)
result['missing2'] = result['feature2'].map(missing_dict)
high_corr = result.loc[
    (result['missing1'] > result['missing2']) | 
    ((result['missing1'] == result['missing2']) & (result['feature1'] > result['feature2'])),
    'feature1'
].tolist()

del result, missing_dict, num_candidates_for_corr

# 粗筛需要剔除的特征清单（仅基于开发训练集自身的数据质量与共线性指标）
delete_feature = set(high_primary_value_ratio + high_corr)

# 初筛后的候选特征集（包含保留的优质原始特征与衍生特征）
candidate_feature_cols = [
    col for col in all_feature_cols
    if col not in delete_feature
]

del orig_num_cols, high_primary_value_ratio, features_corr
del ignore_cols, all_feature_cols, derived_cols, high_corr, delete_feature, cols_eda
gc.collect()

# %%
# -- 精筛 --
# 使用开发训练集五折交叉验证汇总 Gain，降低单次切分带来的重要性波动
X = train_dev[candidate_feature_cols].copy()
y = train_dev['TARGET']

categorical_cols = X.select_dtypes(include=['object', 'category']).columns
X[categorical_cols] = X[categorical_cols].astype('category')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
gain_importance = np.zeros(X.shape[1], dtype=float)

params = {
    # 1. 树结构复杂度控制 (防过拟合)
    'max_depth': 5,             # 限制树深为5，严格约束特征交互阶数，防止拟合长尾稀疏噪声
    'num_leaves': 16,           # 设为16远低于2^5=32，显著降低树复杂度与叶节点方差
    'min_child_samples': 50,    # 单叶节点最小样本数设为50，保证叶子节点违约率置信度
    # 2. 随机采样引入 (增强泛化与特征多样性)
    'subsample': 0.8,           # 每棵树随机采样80%样本，提高抗噪性
    'subsample_freq': 1,        # 每轮都采样
    'colsample_bytree': 0.8,    # 每棵树随机抽取80%特征，防止强特征垄断
    # 3. 正则化约束 (稀疏化与平滑权重)
    'reg_alpha': 1.0,           # L1 正则化
    'reg_lambda': 1.0,          # L2 正则化
    # 4. 学习步长与迭代控制
    'learning_rate': 0.05,
    'n_estimators': 1000,
    # 5. 业务目标与评估指标
    'objective': 'binary',
    'metric': 'auc',
    'class_weight': None,
    # 6. 工程与计算配置
    'random_state': 123,
    'importance_type': 'gain',  # 按分裂增益评估
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_train_f, X_valid_f = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_f, y_valid_f = y.iloc[train_idx], y.iloc[valid_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_valid_f, y_valid_f)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    fold_gain = model.booster_.feature_importance(importance_type='gain')
    fold_gain_sum = fold_gain.sum()
    if fold_gain_sum > 0:
        gain_importance += fold_gain / fold_gain_sum / skf.n_splits

importance_df = pd.DataFrame({
    'feature': X.columns,
    'gain_importance': gain_importance,
}).sort_values('gain_importance', ascending=False, ignore_index=True)

# 计算累计 Gain，保留前 95% 累计重要性的核心特征
importance_df['cumulative_gain'] = importance_df['gain_importance'].cumsum()

selected_feature_cols = importance_df.loc[
    importance_df['cumulative_gain'].shift(1, fill_value=0) < 0.95, 'feature'
].tolist()

print(f"\n精筛完成: 总候选特征 {len(candidate_feature_cols)} 个，保留核心特征 {len(selected_feature_cols)} 个。")
print("\nTop 15 重要特征排名:")
print(importance_df.head(15)[['feature', 'gain_importance', 'cumulative_gain']])

# 保存特征重要性表至 outputs/models
MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
importance_df.to_csv(MODEL_OUTPUT_PATH / 'lgb_feature_importance.csv', index=False)

del X, categorical_cols, candidate_feature_cols, gain_importance, importance_df
del fold, train_idx, valid_idx, X_train_f, X_valid_f, y_train_f, y_valid_f, model, fold_gain, fold_gain_sum
gc.collect()

# %%
# =============================================================================
# 3. 最终模型超参数调优（Optuna TPE 贝叶斯优化）
# =============================================================================
def calc_ks(y_true, y_prob):
    """计算二分类模型的 KS (Kolmogorov-Smirnov) 统计量"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return np.max(np.abs(tpr - fpr))


X_final = train_dev[selected_feature_cols].copy()
cat_cols_final = X_final.select_dtypes(include=['object', 'category']).columns
X_final[cat_cols_final] = X_final[cat_cols_final].astype('category')


def objective(trial):
    max_depth = trial.suggest_int('max_depth', 3, 5)
    max_leaves_bound = (2 ** max_depth) - 1
    num_leaves = trial.suggest_int('num_leaves', 3, max_leaves_bound - 1)
    subsample = trial.suggest_categorical('subsample', [0.8, 0.9, 1.0])
    colsample_bytree = trial.suggest_categorical('colsample_bytree', [0.8, 0.9, 1.0])
    reg_alpha = trial.suggest_float('reg_alpha', 1.0, 1000.0, log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 1.0, 1000.0, log=True)

    trial_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'min_child_samples': 50,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'subsample': subsample,
        'subsample_freq': 1,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'random_state': 123,
        'n_jobs': -1,
        'verbose': -1
    }

    cv_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
    oof_preds_tune = np.zeros(len(X_final))

    for tr_idx, va_idx in cv_tune.split(X_final, y):
        X_tr_t, y_tr_t = X_final.iloc[tr_idx], y.iloc[tr_idx]
        X_va_t, y_va_t = X_final.iloc[va_idx], y.iloc[va_idx]

        model_t = lgb.LGBMClassifier(**trial_params)
        model_t.fit(
            X_tr_t, y_tr_t,
            eval_set=[(X_va_t, y_va_t)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        oof_preds_tune[va_idx] = model_t.predict_proba(X_va_t)[:, 1]

    return roc_auc_score(y, oof_preds_tune)


print("启动 Optuna 超参数搜索 (TPE 算法)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=123))
study.optimize(objective, n_trials=30, show_progress_bar=False)

print(f"\n超参数搜索完成！搜索阶段最佳 OOF 验证 AUC: {study.best_value:.5f}")
print("搜索获得的最优超参数组合:")
for k, v in study.best_params.items():
    if isinstance(v, float):
        print(f"  - {k}: {v:.4f}")
    else:
        print(f"  - {k}: {v}")

best_alpha = study.best_params['reg_alpha']
best_lambda = study.best_params['reg_lambda']

if best_alpha > 50.0:
    print(f"\n⚠️ [特征提示] 搜索得到的 reg_alpha 较大 ({best_alpha:.2f} > 50)。模型可能依赖 L1 截断弱特征，建议进一步排查共线性与噪声！")
else:
    print(f"\n✓ [特征检查正常] reg_alpha = {best_alpha:.2f}，处于较小稳健区间。")

if best_lambda > 100.0:
    print(f"⚠️ [模型提示] 搜索得到的 reg_lambda 较大 ({best_lambda:.2f} > 100)，说明叶节点权重存在极端值并受到强 L2 惩罚。")
else:
    print(f"✓ [模型检查正常] reg_lambda = {best_lambda:.2f}，处于平稳区间。")

del k, v, best_alpha, best_lambda, objective

# %%
# =============================================================================
# 4. 组装最优超参数并在开发集 5 折训练，同步进行闭卷验证与官方测试评估
# =============================================================================
final_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,       # 经验确定
    'n_estimators': 1000,        # 经验确定
    'min_child_samples': 50,     # 数据集规模确定
    'max_depth': study.best_params['max_depth'],
    'num_leaves': study.best_params['num_leaves'],
    'subsample': study.best_params['subsample'],
    'subsample_freq': 1,
    'colsample_bytree': study.best_params['colsample_bytree'],
    'reg_alpha': study.best_params['reg_alpha'],
    'reg_lambda': study.best_params['reg_lambda'],
    'random_state': 123,
    'n_jobs': -1,
    'verbose': -1
}

print(f"\n开始使用搜索出的最佳超参数在精筛后的 {len(selected_feature_cols)} 个核心特征上进行 5 折完整训练与本地/闭卷验证...")

# 准备闭卷验证集特征数据（类别特征对齐开发集分类维度）
X_valid_final = valid_blind[selected_feature_cols].copy()
y_valid = valid_blind['TARGET']
for col in cat_cols_final:
    X_valid_final[col] = pd.Categorical(X_valid_final[col], categories=X_final[col].cat.categories)

# 准备官方未标注测试集特征数据
X_test_final = application_test[selected_feature_cols].copy()
for col in cat_cols_final:
    X_test_final[col] = pd.Categorical(X_test_final[col], categories=X_final[col].cat.categories)

oof_preds = np.zeros(len(train_dev))
valid_preds = np.zeros(len(valid_blind))
test_preds = np.zeros(len(application_test))

fold_aucs = []
fold_kss = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_final, y), 1):
    X_tr, y_tr = X_final.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X_final.iloc[valid_idx], y.iloc[valid_idx]

    clf = lgb.LGBMClassifier(**final_params)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    val_pred = clf.predict_proba(X_va)[:, 1]
    oof_preds[valid_idx] = val_pred

    # 累加闭卷验证集与官方测试集预测概率（5 折集成）
    valid_preds += clf.predict_proba(X_valid_final)[:, 1] / skf.n_splits
    test_preds += clf.predict_proba(X_test_final)[:, 1] / skf.n_splits

    fold_auc = roc_auc_score(y_va, val_pred)
    fold_ks = calc_ks(y_va, val_pred)
    fold_aucs.append(fold_auc)
    fold_kss.append(fold_ks)
    print(f"Fold {fold} | AUC: {fold_auc:.5f} | KS: {fold_ks:.5f}")

# (1) 开发训练集 5 折 OOF 评估
cv_auc = roc_auc_score(y, oof_preds)
cv_ks = calc_ks(y, oof_preds)

# (2) 独立闭卷验证集评估 (Valid Blind, 20%)：仅评估 AUC 与 KS，不计算 PSI
valid_auc = roc_auc_score(y_valid, valid_preds)
valid_ks = calc_ks(y_valid, valid_preds)

# (3) 官方未标注测试集跨客群 PSI 评估：使用开发集 OOF 预测概率分布作为基准分箱
prob_combiner = toad.transform.Combiner()
prob_combiner.fit(pd.Series(oof_preds, name='prob'), method='quantile', n_bins=10)
test_psi = toad.metrics.PSI(oof_preds, test_preds, combiner=prob_combiner)

# 汇总评估指标并落盘至 outputs/models
eval_summary_df = pd.DataFrame({
    '数据集': ['开发训练集 (Train Dev 5-Fold OOF)', '闭卷验证集 (Valid Blind)', '官方测试集 (Official Test)'],
    '样本量': [len(train_dev), len(valid_blind), len(application_test)],
    'AUC': [f"{cv_auc:.4f}", f"{valid_auc:.4f}", "-"],
    'KS': [f"{cv_ks:.4f}", f"{valid_ks:.4f}", "-"],
    '跨客群预测概率 PSI': ["基准 (0.00000)", "-", f"{test_psi:.5f}"]
})
eval_summary_df.to_csv(MODEL_OUTPUT_PATH / 'lgb_eval_summary.csv', index=False)

#                             数据集     样本量     AUC      KS   跨客群预测概率 PSI
# 0  开发训练集 (Train Dev 5-Fold OOF)  246008  0.7676  0.3988  基准 (0.00000)
# 1           闭卷验证集 (Valid Blind)   61503  0.7657  0.3979             -
# 2         官方测试集 (Official Test)   48744       -       -       0.00170

# 清理最终训练与验证过程中的临时变量与特征集副本
del fold, train_idx, valid_idx, X_tr, y_tr, X_va, y_va, clf, val_pred
del X_final, cat_cols_final, X_valid_final, X_test_final, skf, params, final_params, y, y_valid, calc_ks, study
gc.collect()
