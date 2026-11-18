# %%
# =============================================================================
# 逻辑回归标准评分卡建模与评估（Scorecard Modeling & Evaluation）
# =============================================================================
# 数据隔离与评估规范（严格遵循 00_stratified_split.py 与 03_feature_engineering.py 约定）：
# 1. 开发训练集（Train Dev, 80%）：
#    读取 data/processed/train_fe.csv（由 03_feature_engineering.py 基于
#    application_train_dev.csv 加工完成，包含初筛后的 40 个候选特征）。
#    作为特征分箱拟合（fit）、WOE 编码、多变量共线性剔除、逐步回归筛选及评分卡模型训练的唯一输入。
# 2. 闭卷验证集（Valid Blind, 20%）：
#    读取 data/interim/application_valid_blind.csv。在建模完成前完全独立封存，
#    不参与任何规则拟合；仅执行与 03 脚本严格一致的特征清洗与衍生变换（transform），
#    用于独立客观评估 AUC 与 KS。
# 3. 官方未标注测试集（Official Test, 4.8 万条）：
#    读取 data/processed/test_fe.csv（由 03 脚本加工落盘）或从 raw 表执行无监督转换。
#    无 TARGET 标签，直接输入训练好的模型输出预测分值，用于评估跨客群模型总分 PSI。
#
# -- 单箱样本约束 --
# 单箱样本占比原则上 >= 5%（开发集样本 246,008 时，至少约 12,300 个样本）。
# 每个分类或分箱中的坏样本数建议不少于 20~30 个；样本过少时应合并分箱，避免 WOE 估计不稳定。
#
# -- WOE 分箱质量 --
# 1. 组间差异：相邻分类/分箱的 |ΔWOE| 建议 >= 0.10，严格标准可取 0.15；差异过小时考虑合并。
# 2. 单箱幅度：|WOE| 通常以 0.10~1.50 为较合理范围。
#    - |WOE| 长期接近 0（如 -0.05~+0.05）：该箱与总体违约率接近，区分信息有限。
#    - |WOE| 过大（如 > 2.0，尤其 > 3.0）：检查箱内样本/坏样本是否过少，并考虑合并分箱。
# 3. 趋势与稳定性：有序变量的 WOE 应单调递增或递减；无序变量应检查各箱排序在训练集、测试集及 OOT 样本间是否一致。
#    不应出现训练集与测试集（或跨时间样本）间 WOE 方向反转。
# =============================================================================

# %%
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import toad

from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 文件路径配置
DATA_PATH = Path.cwd() / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
INTERIM_DATA_PATH = DATA_PATH / "interim"
PROCESSED_DATA_PATH = DATA_PATH / "processed"

OUTPUTS_PATH = Path.cwd() / "outputs"
EDA_OUTPUT_PATH = OUTPUTS_PATH / "eda"
MODEL_OUTPUT_PATH = OUTPUTS_PATH / "models"
MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# 1. 数据集读取与特征准备
# =============================================================================
# (1) 开发训练集（Train Dev, 80%）：读取由 03 脚本初筛落盘的入模候选特征及 TARGET
train_fe = pd.read_csv(PROCESSED_DATA_PATH / 'train_fe.csv')
X_train = train_fe.drop(columns='TARGET').copy()
y_train = train_fe['TARGET'].copy()
del train_fe

# (2) 闭卷验证集（Valid Blind, 20%）：封存集，含 TARGET
valid_blind = pd.read_csv(INTERIM_DATA_PATH / 'application_valid_blind.csv')
y_valid = valid_blind['TARGET'].copy()

# 读取 EDA 全量特征质量账本以获取房屋特征列表（与 03 脚本完全一致）
cols_eda = pd.read_csv(EDA_OUTPUT_PATH / 'cols_eda.csv')
house_cols = cols_eda.loc[
    cols_eda['Description'].str.startswith('Normalized information'), 'Row'
].tolist()

# 学历有序类别映射（强制单调性）
edu_mapping = {
    'Lower secondary': 1,
    'Secondary / secondary special': 2,
    'Incomplete higher': 3,
    'Higher education': 4,
    'Academic degree': 5
}
X_train['NAME_EDUCATION_TYPE'] = X_train['NAME_EDUCATION_TYPE'].map(edu_mapping)


def transform_features(df_raw, house_cols_list):
    """
    无监督特征映射与转换管道：
    对闭卷验证集 (Valid Blind) 和官方测试集 (Official Test) 执行与 03_feature_engineering.py
    严格一致的异常值清洗、高阶业务衍生与有序类别数值化映射。
    全程不进行任何规则拟合或参数学习，杜绝数据穿越。
    """
    df = df_raw.copy()

    # 1. 异常值清洗与特殊标记（与 03 脚本完全一致）
    df['FLAG_DAYS_EMPLOYED_ANOMALOUS'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', 'F')
    df.loc[df['OBS_30_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
    df.loc[df['OBS_60_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_60_CNT_SOCIAL_CIRCLE'] = np.nan

    # 2. 高阶业务特征衍生（与 03 脚本完全一致）
    # 2.1 还款能力与负债压力
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['FREE_INCOME_PROXY'] = df['AMT_INCOME_TOTAL'] - df['AMT_ANNUITY']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])

    # 2.2 贷款条件与期限结构
    df['ANNUITY_TO_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['ESTIMATED_TENURE'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['GOODS_TO_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    df['DOWN_PAYMENT_PROXY'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    df['ANNUITY_TO_GOODS_RATIO'] = df['AMT_ANNUITY'] / df['AMT_GOODS_PRICE']

    # 2.3 外部征信评分深挖
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
    df['EXT_MEAN_TO_CREDIT_RATIO'] = df['EXT_SOURCE_MEAN'] / df['CREDIT_TO_INCOME_RATIO']

    # 2.4 时间变化、稳定性与生命周期交叉
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (-df['DAYS_BIRTH'] / 365)
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['REG_TO_BIRTH_RATIO'] = df['DAYS_REGISTRATION'] / df['DAYS_BIRTH']
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['AGE_AT_EMPLOYMENT'] = (-df['DAYS_BIRTH'] + df['DAYS_EMPLOYED']) / 365
    df['PHONE_TO_EMPLOYED_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED'].replace(0, np.nan)

    # 2.5 缺失模式与数据采集指示
    avg_house_cols = [c for c in house_cols_list if c.endswith('AVG')]
    df['HOUSE_INFO_MISSING_RATE'] = df[avg_house_cols].isna().mean(axis=1)
    df['FLAG_OWN_CAR_MISSING'] = df['OWN_CAR_AGE'].isna().astype(int)
    df['TOTAL_NAN_COUNT'] = df.isna().sum(axis=1)

    # 2.6 身份真实性、反欺诈与社交圈风险
    doc_cols = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]
    df['DOCUMENTS_PROVIDED_CNT'] = df[doc_cols].sum(axis=1)

    addr_cols = [
        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION'
    ]
    df['ADDRESS_MISMATCH_MEAN'] = df[addr_cols].mean(axis=1)

    df['DEF_OBS_RATIO_30'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)
    df['DEF_OBS_RATIO_60'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['OBS_60_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)

    # 2.7 征信查询趋势与稳定性改造
    bureau_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    df['BUREAU_REQ_TOTAL'] = df[bureau_cols].sum(axis=1)
    df['FLAG_BUREAU_REQ'] = (df['AMT_REQ_CREDIT_BUREAU_YEAR'] > 0).astype(int)
    short_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON'
    ]
    df['BUREAU_REQ_SHORT_TO_LONG'] = df[short_cols].sum(axis=1) / (df['AMT_REQ_CREDIT_BUREAU_YEAR'] + 1)

    # 3. 有序类别特征映射
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].map(edu_mapping)

    return df


# 对独立闭卷验证集执行与 03 严格一致的无监督转换
X_valid = transform_features(valid_blind, house_cols)

# (3) 官方未标注测试集（Official Test, 4.8 万条）：跨客群参照，无 TARGET
# 若 03 脚本已落盘 processed/test_fe.csv 则直接加载，否则从 raw 表执行转换
test_fe_path = PROCESSED_DATA_PATH / 'test_fe.csv'
X_test = pd.read_csv(test_fe_path)
X_test['NAME_EDUCATION_TYPE'] = X_test['NAME_EDUCATION_TYPE'].map(edu_mapping)

del valid_blind, house_cols, cols_eda
gc.collect()

features = X_train.columns.tolist()
num_features = X_train.select_dtypes(include='number').columns.tolist()
cat_features = X_train.select_dtypes(include='object').columns.tolist()

# %%
# =============================================================================
# 2. 精细分箱与 WOE 编码（仅在开发训练集 Train Dev 上拟合）
# =============================================================================
# 数值特征：强制单调 (auto_asc_desc 自动判断单调增或单调减，杜绝锯齿波动)
# 针对典型非线性关系特征（如金额或比率）允许单个拐点 (monotonic_trend="auto")
# 类别特征：optbinning 自动按各分类 Bad Rate 排序并合并相似风险类别
u_shape_cols = [c for c in ["AMT_GOODS_PRICE", "ANNUITY_TO_CREDIT_RATIO", "GOODS_TO_CREDIT_RATIO"] if c in num_features]

# min_bin_n_event=30：落实注释中“每个分箱坏样本数不少于 20~30 个”的约束
binning_fit_params = {}
for col in num_features:
    if col in u_shape_cols:
        binning_fit_params[col] = {
            "monotonic_trend": "auto",            # 允许 1 个合理峰值/谷值拐点
            "max_n_bins": 4,                      # 限制 4 箱以内，防多拐点锯齿
            "min_bin_n_event": 30,                # 单箱至少 30 个坏样本
        }
    else:
        binning_fit_params[col] = {
            "monotonic_trend": "auto_asc_desc",   # 严格单调递增或递减
            "min_bin_n_event": 30,
        }

for col in cat_features:
    binning_fit_params[col] = {
        "min_bin_n_event": 30,
    }

del u_shape_cols

# 实例化精细分箱器并在开发训练集上 fit
fine_binning_process = BinningProcess(
    variable_names=features,
    categorical_variables=cat_features,
    max_n_prebins=30,                             # 预分箱数提升至 30，提高切点精度
    min_prebin_size=0.02,                         # 预分箱最小占比 2% (需小于 min_bin_size)
    min_n_bins=2,
    max_n_bins=5,
    min_bin_size=0.05,
    max_pvalue=0.05,
    max_pvalue_policy="consecutive",
    binning_fit_params=binning_fit_params,
    selection_criteria=None,
    n_jobs=-1
)

fine_binning_process.fit(X_train, y_train)

binning_summary = fine_binning_process.summary().rename(columns={'name': 'feature'})

# 提取 WOE 编码数据集（供共线性检验与逐步回归使用）
X_train_woe = pd.DataFrame(
    fine_binning_process.transform(X_train, metric="woe", metric_missing="empirical"),
    columns=features,
    index=X_train.index
)

# 剔除极低覆盖度且分布极度倾斜的两箱房屋离散类别（如存在）
drop_initial_cat = [c for c in ['HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE'] if c in X_train_woe.columns]
if drop_initial_cat:
    X_train_woe = X_train_woe.drop(columns=drop_initial_cat)
    features = [c for c in features if c not in drop_initial_cat]
    binning_summary = binning_summary[~binning_summary['feature'].isin(drop_initial_cat)]
del drop_initial_cat

# %%
# =============================================================================
# 3. 多变量筛选（相关性、VIF、逐步回归与系数合理性检验）
# =============================================================================
# optbinning woe 定义为 ln(good%/bad%)，好客户占比越高，WOE 越大
# 预测目标为违约概率 P(Y=1) 时，正常入模特征的逻辑回归系数必须为负，正系数代表共线性反转，必须剔除

# -- WOE 相关性筛选 (|r| >= 0.80) --
train_woe_corr = X_train_woe.corr()
result = train_woe_corr.stack().reset_index()
result.columns = ['val1', 'val2', 'corr']
result = result[np.abs(result['corr']).between(0.8, 1, inclusive='left')]
del train_woe_corr

# 保留 IV 值更大的特征，剔除共线低 IV 特征
iv_dict = dict(zip(binning_summary['feature'], binning_summary['iv']))
result['iv1'] = result['val1'].map(iv_dict)
result['iv2'] = result['val2'].map(iv_dict)

delete_corr_feature = list(set(result.loc[result['iv1'] < result['iv2'], 'val1'].tolist()))
del binning_summary, iv_dict, result

features = [col for col in features if col not in delete_corr_feature]
X_train_woe = X_train_woe.drop(columns=delete_corr_feature)

# -- VIF 多重共线性迭代剔除 (阈值: VIF <= 5) --
while True:
    X_with_const = sm.add_constant(X_train_woe)
    vif = pd.Series(
        [variance_inflation_factor(X_with_const.values, i) for i in range(1, X_with_const.shape[1])],
        index=features
    )
    del X_with_const
    max_vif = vif.max()
    if max_vif > 5.0:
        worst_col = vif.idxmax()
        X_train_woe = X_train_woe.drop(columns=[worst_col])
        features.remove(worst_col)
    else:
        del vif
        break

# -- 逐步回归筛选特征 (AIC 准则) --
stepwise_df = toad.selection.stepwise(
    X_train_woe.join(y_train),
    target='TARGET',
    estimator='lr',
    direction='both',
    criterion='aic'
)
final_model_feats = [col for col in stepwise_df.columns if col != "TARGET"]
del features, stepwise_df

# -- 检查入模特征逻辑回归系数是否反转 (大于 0) --
# 在当前入模候选集合上拟合逻辑回归，迭代剔除系数大于 0（符号反转）的特征，直到所有系数均为负
while True:
    logit_mod = sm.Logit(y_train, sm.add_constant(X_train_woe[final_model_feats])).fit(disp=False)
    coefs = logit_mod.params.drop('const')
    pos_coefs = coefs[coefs > 0]
    del logit_mod
    if len(pos_coefs) > 0:
        worst_feature = pos_coefs.idxmax()
        final_model_feats.remove(worst_feature)
    else:
        del coefs, pos_coefs
        break

del X_train_woe
gc.collect()

num_final_feats = [c for c in final_model_feats if c in num_features]
cat_final_feats = [c for c in final_model_feats if c in cat_features]

# %%
# =============================================================================
# 4. 训练标准评分卡并映射 PDO 刻度
# =============================================================================
# 为最终入模特征专门构建 BinningProcess
scorecard_bp = BinningProcess(
    variable_names=final_model_feats,
    categorical_variables=cat_final_feats,
    max_n_prebins=30,
    min_prebin_size=0.02,
    min_n_bins=2,
    max_n_bins=5,
    min_bin_size=0.05,
    max_pvalue=0.05,
    max_pvalue_policy="consecutive",
    binning_fit_params={c: binning_fit_params[c] for c in final_model_feats},
    selection_criteria=None,
    n_jobs=-1
)

# 设置标准评分卡刻度参数：
# - scaling_method="pdo_odds"
# - pdo=20：好坏比 (Odds) 每翻一倍，分数增加 20 分
# - scorecard_points=600：基准分 600 分
base_odds = (1 - y_train.mean()) / y_train.mean()

lr_estimator = LogisticRegression(
    penalty=None,        # 关闭正则化，保持标准线性 Logit 尺度
    solver='lbfgs',
    max_iter=1000,
    tol=1e-6,
    random_state=123
)

scorecard = Scorecard(
    binning_process=scorecard_bp,
    estimator=lr_estimator,
    scaling_method="pdo_odds",
    scaling_method_params={
        "pdo": 20,
        "odds": base_odds,
        "scorecard_points": 600
    },
    rounding=True        # 每个分箱得分四舍五入为整数，便于线上部署
)

del base_odds, lr_estimator, scorecard_bp, binning_fit_params

# Scorecard.fit 接收开发训练集的入模特征原始值
scorecard.fit(X_train[final_model_feats], y_train)

# 导出标准评分卡各分箱加减分明细表至 outputs/models
scorecard_table = scorecard.table(style="detailed")
scorecard_table.to_csv(MODEL_OUTPUT_PATH / 'final_scorecard_table.csv', index=False)

# %%
# =============================================================================
# 5. 模型综合评估（AUC、KS 与 PSI，严格按照 00_stratified_split.py 规定分工）
# =============================================================================
# 预测违约概率 (用于算 AUC / KS) 与 信用评分 (用于算跨客群总分 PSI)
train_prob = scorecard.predict_proba(X_train[final_model_feats])[:, 1]
valid_prob = scorecard.predict_proba(X_valid[final_model_feats])[:, 1]

train_score = scorecard.score(X_train[final_model_feats])
test_score = scorecard.score(X_test[final_model_feats])


def calc_ks(y_true, y_prob):
    """计算 KS (Kolmogorov-Smirnov) 统计量"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return np.max(np.abs(tpr - fpr))


# (1) AUC 与 KS 评估：开发训练集 (Train Dev) vs 独立闭卷验证集 (Valid Blind)
train_auc = roc_auc_score(y_train, train_prob)
valid_auc = roc_auc_score(y_valid, valid_prob)

train_ks = calc_ks(y_train, train_prob)
valid_ks = calc_ks(y_valid, valid_prob)

# (2) 官方测试集跨客群模型总分 PSI 评估：使用 toad 的 Combiner 在开发集总分上拟合 10 等频分箱切点
score_combiner = toad.transform.Combiner()
score_combiner.fit(pd.Series(train_score, name='score'), method='quantile', n_bins=10)

# 官方未标注测试集跨客群总分 PSI（根据 00_stratified_split.py 说明，评估面对未知分布时的跨客群稳定性）
test_score_psi = toad.metrics.PSI(train_score, test_score, combiner=score_combiner)

# 汇总评估指标并落盘
eval_summary_df = pd.DataFrame({
    '数据集': ['开发训练集 (Train Dev)', '闭卷验证集 (Valid Blind)', '官方测试集 (Official Test)'],
    '样本量': [len(X_train), len(X_valid), len(X_test)],
    'AUC': [f"{train_auc:.4f}", f"{valid_auc:.4f}", "-"],
    'KS': [f"{train_ks:.4f}", f"{valid_ks:.4f}", "-"],
    '跨客群总分 PSI': ["基准 (0.00000)", "-", f"{test_score_psi:.5f}"]
})
eval_summary_df.to_csv(MODEL_OUTPUT_PATH / 'scorecard_eval_summary.csv', index=False)

#                      数据集     样本量     AUC      KS     跨客群总分 PSI
# 0      开发训练集 (Train Dev)  246008  0.7517  0.3749  基准 (0.00000)
# 1    闭卷验证集 (Valid Blind)   61503  0.7470  0.3681             -
# 2  官方测试集 (Official Test)   48744       -       -       0.00157
