# %%
# =============================================================================
# 逻辑回归标准信用评分卡建模与评估
# (Scorecard Modeling & Screening Pipeline)
# =============================================================================
# 模块功能：
# 实现基于逻辑回归的标准信用评分卡建模流程：
# 1. 学历有序类别数值编码；
# 2. 数据质量粗筛（剔除同值率 > 95% 或缺失率 > 90% 的字段，IV >= 0.02 准入）；
# 3. 房屋同源特征去重与相关性初筛（Spearman |r| >= 0.80，保留高 IV 特征）；
# 4. OptBinning 分箱（单箱占比 >= 5%，坏样本 >= 30，单调性约束）；
# 5. WOE 编码与多变量共线性排查（相关性过滤、VIF <= 5 迭代剔除）；
# 6. 双向逐步回归（AIC 准则）与系数符号合理性检验（排除符号反转特征）；
# 7. PDO 刻度映射（基准分 600 分，PDO = 20，输出整数加减分明细）；
# 8. 独立验证集评估（AUC、KS）与测试集总分跨客群稳定性评估（PSI）。
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
PROCESSED_DATA_PATH = DATA_PATH / "processed"
OUTPUTS_PATH = Path.cwd() / "outputs"
EDA_OUTPUT_PATH = OUTPUTS_PATH / "eda"
MODEL_OUTPUT_PATH = OUTPUTS_PATH / "models"
MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# 1. 数据集读取与候选特征池准备
# =============================================================================
print("正在加载统一特征工程全量宽表...")
# 正在加载统一特征工程全量宽表...
train_df = pd.read_csv(PROCESSED_DATA_PATH / 'train_features_full.csv')
valid_df = pd.read_csv(PROCESSED_DATA_PATH / 'valid_features_full.csv')
test_df = pd.read_csv(PROCESSED_DATA_PATH / 'test_features_full.csv')

y_train = train_df['TARGET'].copy()
y_valid = valid_df['TARGET'].copy()

# 学历有序类别数值编码（在特征筛选前完成单调化映射）
edu_mapping = {
    'Lower secondary': 1,
    'Secondary / secondary special': 2,
    'Incomplete higher': 3,
    'Higher education': 4,
    'Academic degree': 5
}
for df_obj in [train_df, valid_df, test_df]:
    df_obj['NAME_EDUCATION_TYPE'] = df_obj['NAME_EDUCATION_TYPE'].map(edu_mapping)

ignore_cols = ['SK_ID_CURR', 'TARGET']
candidate_features = [c for c in train_df.columns if c not in ignore_cols]

print(f"数据加载完成: 训练集 {train_df.shape}, 验证集 {valid_df.shape}, 测试集 {test_df.shape}")
print(f"评分卡初始候选特征数: {len(candidate_features)}")
# 数据加载完成: 训练集 (246008, 164), 验证集 (61503, 164), 测试集 (48744, 163)
# 评分卡初始候选特征数: 162

# 读取原始特征画像与说明表提取房屋相关字段
raw_features_summary = pd.read_csv(EDA_OUTPUT_PATH / 'raw_features_summary.csv')
house_cols = raw_features_summary.loc[
    raw_features_summary['Description'].str.startswith('Normalized information'), 'Row'
].tolist()
del raw_features_summary

# %%
# =============================================================================
# 2. 评分卡特征漏斗初筛（质量、IV、房屋去重与相关性）
# =============================================================================
# (1) 数据质量粗筛：同值率 > 0.95 或 缺失率 > 0.90
missing_rates = train_df[candidate_features].isna().mean()
primary_value_ratios = pd.Series({
    col: train_df[col].value_counts(dropna=False).iloc[0] / len(train_df)
    for col in candidate_features
})

high_pvr_cols = primary_value_ratios[primary_value_ratios > 0.95].index.tolist()
high_missing_cols = missing_rates[missing_rates > 0.90].index.tolist()
quality_drop_cols = sorted(list(set(high_pvr_cols + high_missing_cols)))

filtered_candidates = [c for c in candidate_features if c not in quality_drop_cols]
print(f"✓ 质量初筛完成: 剔除极高同值/缺失字段 {len(quality_drop_cols)} 个，剩余 {len(filtered_candidates)} 个")
# ✓ 质量初筛完成: 剔除极高同值/缺失字段 21 个，剩余 141 个

# 区分数值型特征与类别型特征
num_filtered = train_df[filtered_candidates].select_dtypes(include='number').columns.tolist()
cat_filtered = [c for c in filtered_candidates if c not in num_filtered]

# (2) 批量最优分箱计算特征 IV（仅在训练集上拟合）
binning_fit_params = {
    col: {"monotonic_trend": "auto", "cat_cutoff": 0.05}
    for col in filtered_candidates
}

binning_process = BinningProcess(
    variable_names=filtered_candidates,
    categorical_variables=cat_filtered,
    max_n_bins=8,
    min_bin_size=0.02,
    special_codes=[365243, 'XNA'],
    binning_fit_params=binning_fit_params,
    selection_criteria=None,
    n_jobs=-1
)
binning_process.fit(train_df[filtered_candidates], y_train)

iv_summary_df = binning_process.summary().rename(columns={'name': 'feature'})
iv_dict = dict(zip(iv_summary_df['feature'], iv_summary_df['iv']))
del binning_process
gc.collect()

# (3) IV >= 0.02 筛选
metric_passed_stats = pd.DataFrame({
    'Row': filtered_candidates,
    'iv': [iv_dict.get(c, np.nan) for c in filtered_candidates]
})
metric_passed_stats = metric_passed_stats[metric_passed_stats['iv'] >= 0.02].sort_values(by='iv', ascending=False).reset_index(drop=True)
passed_cols = metric_passed_stats['Row'].tolist()
print(f"✓ IV 阈值准入 (IV >= 0.02): 保留特征 {len(passed_cols)} 个")
# ✓ IV 阈值准入 (IV >= 0.02): 保留特征 79 个

# (4) 房屋特征同源后缀去重与强相关面积字段去共线
passed_house_cols = [c for c in passed_cols if c in house_cols and c in num_filtered]
house_redundant_drops = []
house_df = metric_passed_stats[metric_passed_stats['Row'].isin(passed_house_cols)].copy()
house_df['prefix'] = house_df['Row'].str.split('_').str[0]
house_df = house_df.sort_values(by=['prefix', 'iv'], ascending=[True, False])
house_keep = house_df.groupby('prefix')['Row'].head(1).tolist()
house_suffix_drops = [c for c in passed_house_cols if c not in house_keep]
house_redundant_drops.extend(house_suffix_drops)

# 跨维度规模共线（APARTMENTS, ELEVATORS, LIVINGAREA, TOTALAREA 相关性 > 0.75）
apt_rep = [c for c in house_keep if c.startswith('APARTMENTS_')]
cross_dim_cands = apt_rep + [c for c in ['ELEVATORS_AVG', 'LIVINGAREA_AVG', 'TOTALAREA_MODE'] if c in house_keep]
cross_dim_iv = metric_passed_stats[metric_passed_stats['Row'].isin(cross_dim_cands)].sort_values(by='iv', ascending=False)
if len(cross_dim_iv) > 0:
    cross_dim_drops = cross_dim_iv.iloc[1:]['Row'].tolist()
    house_redundant_drops.extend(cross_dim_drops)

cols_after_house = [c for c in passed_cols if c not in house_redundant_drops]
del house_df, house_keep, house_suffix_drops, house_redundant_drops

# (5) 贪心相关性过滤 (|r| > 0.80，优先保留高 IV 特征)
num_passed = train_df[cols_after_house].select_dtypes(include='number').columns.tolist()
cat_passed = [c for c in cols_after_house if c not in num_passed]

num_stats = metric_passed_stats[metric_passed_stats['Row'].isin(num_passed)].sort_values(by='iv', ascending=False)
sorted_num_cols = num_stats['Row'].tolist()

spearman_corr = train_df[sorted_num_cols].corr(method='spearman')

CORR_THRESHOLD = 0.80
kept_num_cols = []
for col in sorted_num_cols:
    is_collinear = False
    for kept in kept_num_cols:
        r = spearman_corr.loc[col, kept]
        if abs(r) > CORR_THRESHOLD:
            is_collinear = True
            break
    if not is_collinear:
        kept_num_cols.append(col)

candidate_model_feats = kept_num_cols + cat_passed
print(f"✓ 贪心共线性剔除 (|r| > 0.80): 产生 {len(candidate_model_feats)} 个入模候选特征")
# ✓ 贪心共线性剔除 (|r| > 0.80): 产生 44 个入模候选特征

del spearman_corr, num_stats, sorted_num_cols, num_passed, cat_passed, cols_after_house, metric_passed_stats

# %%
# =============================================================================
# 3. 约束分箱（OptBinning）
# =============================================================================
features = candidate_model_feats.copy()
num_features = train_df[features].select_dtypes(include='number').columns.tolist()
cat_features = [c for c in features if c not in num_features]

# 部分非线性比率特征放宽单调性限制（允许单峰/单谷），最多 4 箱
u_shape_cols = [c for c in ["AMT_GOODS_PRICE", "ANNUITY_TO_CREDIT_RATIO", "GOODS_TO_CREDIT_RATIO"] if c in num_features]

fine_binning_fit_params = {}
for col in num_features:
    if col in u_shape_cols:
        fine_binning_fit_params[col] = {
            "monotonic_trend": "auto",            # 允许 1 个峰值/谷值拐点
            "max_n_bins": 4,                      # 限制 4 箱以内，防多拐点锯齿
            "min_bin_n_event": 30,                # 单箱至少 30 个坏样本
        }
    else:
        fine_binning_fit_params[col] = {
            "monotonic_trend": "auto_asc_desc",   # 严格单调递增或递减
            "min_bin_n_event": 30,
        }

for col in cat_features:
    fine_binning_fit_params[col] = {
        "min_bin_n_event": 30,
    }

del u_shape_cols

# 实例化精细分箱器并在训练集上拟合
print("正在执行精细约束分箱 (OptBinning)...")
# 正在执行精细约束分箱 (OptBinning)...
fine_binning_process = BinningProcess(
    variable_names=features,
    categorical_variables=cat_features,
    max_n_prebins=30,
    min_prebin_size=0.02,
    min_n_bins=2,
    max_n_bins=5,
    min_bin_size=0.05,
    max_pvalue=0.05,
    max_pvalue_policy="consecutive",
    binning_fit_params=fine_binning_fit_params,
    selection_criteria=None,
    n_jobs=-1
)
fine_binning_process.fit(train_df[features], y_train)

binning_summary = fine_binning_process.summary().rename(columns={'name': 'feature'})

# 提取 WOE 编码数据集（供后续共线性排查与逐步回归使用）
X_train_woe = pd.DataFrame(
    fine_binning_process.transform(train_df[features], metric="woe", metric_missing="empirical"),
    columns=features,
    index=train_df.index
)

# 剔除极低覆盖度且分布极度倾斜的房屋类别
drop_initial_cat = [c for c in ['HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE'] if c in X_train_woe.columns]
if drop_initial_cat:
    X_train_woe = X_train_woe.drop(columns=drop_initial_cat)
    features = [c for c in features if c not in drop_initial_cat]
    binning_summary = binning_summary[~binning_summary['feature'].isin(drop_initial_cat)]
del drop_initial_cat

# %%
# =============================================================================
# 4. 多变量共线性排查、逐步回归与系数合理性校验
# =============================================================================
print("正在执行 WOE 相关性过滤、VIF 诊断与逐步回归...")
# 正在执行 WOE 相关性过滤、VIF 诊断与逐步回归...
# -- (1) WOE 相关性筛选 (|r| >= 0.80) --
train_woe_corr = X_train_woe.corr()
corr_pairs = train_woe_corr.stack().reset_index()
corr_pairs.columns = ['val1', 'val2', 'corr']
corr_pairs = corr_pairs[np.abs(corr_pairs['corr']).between(0.8, 1, inclusive='left')]
del train_woe_corr

iv_dict_fine = dict(zip(binning_summary['feature'], binning_summary['iv']))
corr_pairs['iv1'] = corr_pairs['val1'].map(iv_dict_fine)
corr_pairs['iv2'] = corr_pairs['val2'].map(iv_dict_fine)

delete_corr_feature = list(set(corr_pairs.loc[corr_pairs['iv1'] < corr_pairs['iv2'], 'val1'].tolist()))
del binning_summary, iv_dict_fine, corr_pairs

features = [col for col in features if col not in delete_corr_feature]
X_train_woe = X_train_woe.drop(columns=delete_corr_feature)
del delete_corr_feature

# -- (2) 常数截距 VIF 迭代剔除 (阈值: VIF <= 5.0) --
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

print(f"✓ VIF 诊断完成 (全部特征 VIF <= 5.0)，剩余特征数: {len(features)}")
# ✓ VIF 诊断完成 (全部特征 VIF <= 5.0)，剩余特征数: 32

# -- (3) 双向逐步回归 (AIC 准则) --
stepwise_df = toad.selection.stepwise(
    X_train_woe.join(y_train),
    target='TARGET',
    estimator='lr',
    direction='both',
    criterion='aic'
)
final_model_feats = [col for col in stepwise_df.columns if col != "TARGET"]
del features, stepwise_df

# -- (4) 逻辑回归系数符号合理性检验与反转剔除 --
# optbinning 的 WOE 定义为 ln(good% / bad%)，好客户占比越高则 WOE 越大。
# 预测目标为违约（TARGET=1）时，正常系数应为负；正系数表明存在共线性反转，需剔除以防风险方向倒挂。
while True:
    logit_mod = sm.Logit(y_train, sm.add_constant(X_train_woe[final_model_feats])).fit(disp=False)
    coefs = logit_mod.params.drop('const')
    pos_coefs = coefs[coefs > 0]
    del logit_mod
    if len(pos_coefs) > 0:
        worst_feature = pos_coefs.idxmax()
        print(f"  ⚠️ 发现符号反转变量 (系数为正): {worst_feature}，予以剔除以防风险倒挂")
        final_model_feats.remove(worst_feature)
    else:
        del coefs, pos_coefs
        break
#   ⚠️ 发现符号反转变量 (系数为正): PHONE_TO_EMPLOYED_RATIO，予以剔除以防风险倒挂
#   ⚠️ 发现符号反转变量 (系数为正): INCOME_TO_OCCUPATION_TYPE_RATIO，予以剔除以防风险倒挂
#   ⚠️ 发现符号反转变量 (系数为正): OCCUPATION_TYPE_INCOME_MEAN，予以剔除以防风险倒挂
#   ⚠️ 发现符号反转变量 (系数为正): REG_CITY_NOT_WORK_CITY，予以剔除以防风险倒挂
#   ⚠️ 发现符号反转变量 (系数为正): NAME_INCOME_TYPE，予以剔除以防风险倒挂
#   ⚠️ 发现符号反转变量 (系数为正): EXT_SOURCE_MAX，予以剔除以防风险倒挂

del X_train_woe
gc.collect()

print(f"✓ 逐步回归与符号校验完成！最终入模特征数: {len(final_model_feats)}")
# ✓ 逐步回归与符号校验完成！最终入模特征数: 25

# %%
# =============================================================================
# 5. 训练标准信用评分卡与 PDO 刻度映射
# =============================================================================
num_final_feats = [c for c in final_model_feats if c in num_features]
cat_final_feats = [c for c in final_model_feats if c in cat_features]

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
    binning_fit_params={c: fine_binning_fit_params[c] for c in final_model_feats},
    selection_criteria=None,
    n_jobs=-1
)

base_odds = (1 - y_train.mean()) / y_train.mean()

lr_estimator = LogisticRegression(
    penalty=None,        # 不使用正则化，保持标准 Logit 参数估计
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
    rounding=True        # 分箱得分四舍五入取整
)

del base_odds, lr_estimator, scorecard_bp, fine_binning_fit_params

# Scorecard.fit 接收训练集的原始特征值
print("正在拟合标准评分卡并执行 PDO=20, 600 基准分刻度映射...")
# 正在拟合标准评分卡并执行 PDO=20, 600 基准分刻度映射...
scorecard.fit(train_df[final_model_feats], y_train)

# 导出标准评分卡明细表
scorecard_table = scorecard.table(style="detailed")
scorecard_table.to_csv(MODEL_OUTPUT_PATH / 'final_scorecard_table.csv', index=False)
print(f"✓ 标准评分卡明细表已导出至: {MODEL_OUTPUT_PATH / 'final_scorecard_table.csv'}")
# ✓ 标准评分卡明细表已导出至: D:\0.My-Projects\home-credit-default-risk\outputs\models\final_scorecard_table.csv

# %%
# =============================================================================
# 6. 模型效果综合评估（AUC、KS 与跨客群总分 PSI）
# =============================================================================
print("正在进行验证集评估与测试集总分 PSI 检验...")
# 正在进行验证集评估与测试集总分 PSI 检验...
train_prob = scorecard.predict_proba(train_df[final_model_feats])[:, 1]
valid_prob = scorecard.predict_proba(valid_df[final_model_feats])[:, 1]

train_score = scorecard.score(train_df[final_model_feats])
test_score = scorecard.score(test_df[final_model_feats])

def calc_ks(y_true, y_prob):
    """计算 KS (Kolmogorov-Smirnov) 统计量"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return np.max(np.abs(tpr - fpr))

train_auc = roc_auc_score(y_train, train_prob)
valid_auc = roc_auc_score(y_valid, valid_prob)

train_ks = calc_ks(y_train, train_prob)
valid_ks = calc_ks(y_valid, valid_prob)

# 测试集总分 PSI 评估 (以训练集总分等频分箱为基准)
score_combiner = toad.transform.Combiner()
score_combiner.fit(pd.Series(train_score, name='score'), method='quantile', n_bins=10)
test_score_psi = toad.metrics.PSI(train_score, test_score, combiner=score_combiner)

# 汇总并保存评估报告
eval_summary_df = pd.DataFrame({
    '数据集': ['训练集 (Train)', '验证集 (Valid)', '测试集 (Test)'],
    '样本量': [len(train_df), len(valid_df), len(test_df)],
    'AUC': [f"{train_auc:.4f}", f"{valid_auc:.4f}", "-"],
    'KS': [f"{train_ks:.4f}", f"{valid_ks:.4f}", "-"],
    '跨客群总分 PSI': ["基准 (0.00000)", "-", f"{test_score_psi:.5f}"]
})
eval_summary_df.to_csv(MODEL_OUTPUT_PATH / 'scorecard_eval_summary.csv', index=False)

print("\n" + "=" * 70)
print("✓ [标准信用评分卡建模与评估完成 - 汇总账本]")
print(eval_summary_df.to_string(index=False))
print("=" * 70)
# ======================================================================
# ✓ [标准信用评分卡建模与评估完成 - 汇总账本]
#         数据集    样本量    AUC     KS    跨客群总分 PSI
# 训练集 (Train) 246008 0.7523 0.3747 基准 (0.00000)
# 验证集 (Valid)  61503 0.7466 0.3665            -
#  测试集 (Test)  48744      -      -      0.00131
# ======================================================================

del train_df, valid_df, test_df, y_train, y_valid, train_prob, valid_prob, train_score, test_score
del scorecard, score_combiner, eval_summary_df, scorecard_table
gc.collect()
