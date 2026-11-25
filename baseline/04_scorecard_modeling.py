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
# 【调参说明】
# - primary_value_ratios > 0.95（同值率阈值 = 95%）：
#   说明：单个取值（含缺失）在训练集中占比超过 95% 时直接剔除。
#   原因：高度同质化的特征几乎没有区分度，且后续精细分箱要求单箱最小占比 >= 5%，
#         同值率 > 95% 的特征无法切出至少 2 个有效分箱（其余箱样本不足 5%），需提前淘汰。
# - missing_rates > 0.90（缺失率阈值 = 90%）：
#   说明：缺失率超过 90% 的字段直接剔除。
#   原因：缺失超过 90% 意味着有效观测不足 10%，不仅统计估计方差极大、极易过拟合，在评分卡中也难以稳定部署；
#         放宽至 90%（而非更严的 70%）是为了在粗筛期保留少数高价值稀疏衍生字段，交由后续 IV 检验决定去留。
missing_rates = train_df[candidate_features].isna().mean()
primary_value_ratios = pd.Series({
    col: train_df[col].value_counts(dropna=False, normalize=True).iloc[0]
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

# (2) 批量最优分箱计算特征 IV（仅在训练集上拟合，防止验证/测试集标签泄漏）
# 【调参说明 - 逐特征拟合参数】
# - monotonic_trend="auto"（数值型）：允许自动拟合单调或单峰/单谷（U型/倒U型）趋势。
#   原因：初筛期旨在客观评估特征的潜在区分度（IV），放宽单调约束可防止误杀具有合理非线性风险结构的比率特征。
# - cat_cutoff=0.01（类别型）：样本占比低于 1% 的长尾稀有类别自动合并为 'others'。
#   原因：部分类别字段（如 OCCUPATION_TYPE 等）存在极小占比长尾类，合并 < 1% 类别可防止小样本分片导致虚高 IV。
binning_fit_params = {
    col: {"monotonic_trend": "auto"}
    for col in num_filtered
}
for col in cat_filtered:
    binning_fit_params[col] = {"cat_cutoff": 0.01}

# 【调参说明 - BinningProcess 全局超参数】
binning_process = BinningProcess(
    variable_names=filtered_candidates,
    categorical_variables=cat_filtered,
    max_n_prebins=30,           # 预分箱最大箱数设为 30（高于默认 20）：保留足够细粒度的局部违约率变化，为后续最优合并提供丰富切分点
    min_prebin_size=0.02,       # 预分箱最小占比设为 2%（约 4920 样本）：保证每个初始切片样本充足，避免小样本噪声干扰最优分箱求解
    max_n_bins=8,               # 最大分箱数设为 8（比入模阶段的 5 箱更宽）：初筛期充分释放特征的信息价值，避免过度压缩箱数导致高价值特征被误筛
    min_bin_size=0.02,          # 单箱最小占比设为 2%（比入模阶段的 5% 更宽）：允许捕捉少数极端风险客群对 IV 的贡献
    special_codes=['XNA'],      # 特殊值列表：Home Credit 数据中 'XNA' 代表“未知/未适用”，单独成箱可防止混入常规类别或缺失箱扭曲 WOE
    binning_fit_params=binning_fit_params,  # 传入上方定制的趋势与长尾合并参数
    selection_criteria=None,    # 内部筛选准则设为 None：关闭 BinningProcess 自动丢弃，保留全量 IV 供后续显式漏斗可控筛选
    n_jobs=-1                   # 并行线程数设为 -1：调用全部 CPU 核心并行求解，缩短 141 个特征的批量分箱耗时
)
binning_process.fit(train_df[filtered_candidates], y_train)

iv_summary_df = binning_process.summary().rename(columns={'name': 'feature'})
iv_dict = dict(zip(iv_summary_df['feature'], iv_summary_df['iv']))
del binning_process
gc.collect()

# (3) IV >= 0.02 筛选
# 【调参说明】
# - IV 准入门槛 >= 0.02：
#   说明：剔除信息价值（IV）低于 0.02 的特征。
#   原因：IV < 0.02 是风控评分卡领域公认的“无预测能力/极弱预测能力”行业经验阈值，可有效过滤噪声变量、精简入模候选池并降低过拟合风险。
metric_passed_stats = pd.DataFrame({
    'Row': filtered_candidates,
    'iv': [iv_dict.get(c, np.nan) for c in filtered_candidates]
})
metric_passed_stats = metric_passed_stats[metric_passed_stats['iv'] >= 0.02].sort_values(by='iv', ascending=False).reset_index(drop=True)
passed_cols = metric_passed_stats['Row'].tolist()
print(f"✓ IV 阈值准入 (IV >= 0.02): 保留特征 {len(passed_cols)} 个")
# ✓ IV 阈值准入 (IV >= 0.02): 保留特征 79 个

# (4) 房屋特征同源后缀去重与强相关面积字段去共线
# 【调参说明】
# - 同前缀（_AVG / _MODE / _MEDI）仅保留 IV 最高者：同一物理指标的三种统计量高度冗余，择优保留 1 个即可消除同源共线。
# - 跨维度规模字段（APARTMENTS / ELEVATORS_AVG / LIVINGAREA_AVG / TOTALAREA_MODE）仅保留 IV 最高者：
#   这些字段均反映住宅体量（相关性 > 0.75），提前按业务含义去重可防止逻辑回归系数互相抵消。
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
# 【调参说明】
# - method='spearman'（Spearman 秩相关）：
#   原因：评分卡后续要经过分箱与单调 WOE 映射，Spearman 对极端值不敏感且能捕获单调非线性相关，比 Pearson 更贴合分箱前的特征去共线。
# - CORR_THRESHOLD = 0.80（相关性过滤阈值）：
#   说明：按 IV 从高到低贪心保留，若与已选特征 |r| > 0.80 则剔除。
#   原因：逻辑回归对多重共线性敏感（易致系数方差膨胀与符号倒挂），因此阈值比 LightGBM 脚本（0.95）更严；
#         设为 0.80 而非过低的 0.60，可保留存在部分互补信息的特征，将精细共线治理留给后续基于 WOE 的 VIF 检验。
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
# 【调参说明 - 逐特征精细约束】
# - u_shape_cols（AMT_GOODS_PRICE, ANNUITY_TO_CREDIT_RATIO, GOODS_TO_CREDIT_RATIO）：
#   - monotonic_trend="auto"：这 3 个金额/比率特征在信贷业务中天然存在“两头高或中间高”的 U 型/倒 U 型风险形态，
#     放宽单调性允许最多 1 个峰值/谷值拐点，避免强行拉直导致业务信号丢失。
#   - max_n_bins=4：在放宽单调性的同时将最大箱数从 5 收紧至 4，限制分段自由度以防出现多拐点锯齿状过拟合。
# - 其余数值型特征 monotonic_trend="auto_asc_desc"：
#   - 强制严格单调递增或递减（自动判定方向）。这是标准评分卡可解释性的核心要求，确保特征单调变化时分数同步单调变化、跨期更稳健。
# - min_bin_n_event=30（全特征统一）：
#   - 强制单箱至少包含 30 个违约（坏）样本。在 24.6 万样本、约 8% 违约率下，30 个坏样本是保证箱内违约率与 WOE 估计具有统计稳健性的下限，
#     避免高风险小箱因样本过少出现极端 WOE 波动。
# - cat_cutoff=0.01（类别型特征）：
#   - 合并占比低于 1% 的长尾稀有类别为 'others'，防止稀有类别单独成箱引起线上打分剧烈波动。
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
        "cat_cutoff": 0.01,
        "min_bin_n_event": 30,
    }

del u_shape_cols

# 实例化精细分箱器并在训练集上拟合
print("正在执行精细约束分箱 (OptBinning)...")
# 正在执行精细约束分箱 (OptBinning)...
# 【调参说明 - 精细分箱全局超参数】
fine_binning_process = BinningProcess(
    variable_names=features,
    categorical_variables=cat_features,
    max_n_prebins=30,                 # 预分箱上限设为 30：与初筛一致，提供充足细粒度切分点供带统计检验的最优规划合并
    min_prebin_size=0.02,             # 预分箱最小占比设为 2%：保证初始切片统计稳定，防止噪声切片影响全局切分
    min_n_bins=2,                     # 最小分箱数设为 2：强制每个入模特征至少分出 2 档，保证具备基础风险分层能力（单箱无区分度）
    max_n_bins=5,                     # 最大分箱数收紧为 5（初筛为 8）：评分卡经验上 3~5 箱最稳健，既刻画风险梯度又防分箱过碎导致 PSI 漂移
    min_bin_size=0.05,                # 单箱最小占比收紧为 5%（初筛为 2%，约 1.23 万样本）：保障每档得分有充足客群覆盖度，大幅提升跨集/跨期稳定性
    max_pvalue=0.05,                  # 相邻箱差异显著性 p 值上限设为 0.05（95% 置信度）：相邻分箱违约率若无统计显著差异则强制合并，抑制噪声切分
    max_pvalue_policy="consecutive",  # p 值检验策略设为 "consecutive"：逐对检验相邻分箱，既贴合单调排列逻辑又避免跨箱过度合并
    special_codes=['XNA'],            # 与初筛一致，将 'XNA' 划为独立特殊箱，防止与常规值或缺失值混并扭曲 WOE
    binning_fit_params=fine_binning_fit_params,  # 传入上方逐特征定制的单调约束、箱数上限与最小坏样本数
    selection_criteria=None,          # 关闭内置自动剔除，交由后续 WOE 共线性、VIF 与逐步回归统一精筛
    n_jobs=-1                         # 全核并行加速拟合
)
fine_binning_process.fit(train_df[features], y_train)

binning_summary = fine_binning_process.summary().rename(columns={'name': 'feature'})

# 提取 WOE 编码数据集（供后续共线性排查与逐步回归使用）
# 【调参说明 - WOE 转换参数】
# - metric="woe"：将分箱转为 WOE 值，完成非线性特征的线性化与无量纲化以适配逻辑回归。
# - metric_special="empirical" / metric_missing="empirical"：
#   对特殊值箱（'XNA'）和缺失值箱直接使用各自箱内的经验 WOE，而非强行归零或填均值；
#   原因是在信用风控中“缺失”和“未知（XNA）”往往携带明确的风险信息，经验 WOE 能完整保留其区分能力。
X_train_woe = pd.DataFrame(
    fine_binning_process.transform(
        train_df[features],
        metric="woe",
        metric_special="empirical",
        metric_missing="empirical",
    ),
    columns=features,
    index=train_df.index
)

# 剔除极低覆盖度且分布极度倾斜的房屋类别
# 【调参说明】
# - 人工剔除 HOUSETYPE_MODE、EMERGENCYSTATE_MODE：这两个特征缺失/单值倾斜极严重且整体信息量弱（IV 仅约 0.02 出头），
#   与已保留的核心房屋特征高度冗余，人工移除可进一步精简评分卡规则表并避免稀疏类别波动。
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
# 【调参说明】
# - between(0.8, 1, inclusive='left')（即 0.80 <= |r| < 1.0）：
#   说明：在完成分箱和 WOE 编码后的空间中，再次剔除相关性 >= 0.80 对中 IV 较低的特征。
#   原因：原始数值/类别经非线性分箱和特殊值 WOE 映射后，原本未超标的特征对在 WOE 空间可能表现出高度线性相关，
#         必须进行二次共线过滤以减轻逻辑回归共线性压力。
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
# 【调参说明】
# - max_vif > 5.0（方差膨胀因子阈值 = 5.0，对应辅助回归 R^2 = 0.80）：
#   说明：每轮仅剔除 VIF 最大的一个特征并重新计算，直至全量特征 VIF <= 5.0。
#   原因：单轮仅删最差特征可防止同时误删一对共线特征；阈值设为 5.0（严格于常规统计学的 10.0）是风控评分卡标准做法，
#         能有效抑制多变量复合共线、保持回归系数稳定性并大幅降低符号反转概率。
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
# 【调参说明】
# - estimator='lr'：指定基模型为逻辑回归，与最终评分卡模型形式严格对齐。
# - direction='both'：采用向前引入与向后剔除结合的双向搜索，新特征进入后可自动淘汰不再显著的已有特征，避免单向搜索陷入局部最优。
# - criterion='aic'：使用赤池信息准则（AIC = 2k - 2ln(L)）而非 BIC（k*ln(N) - 2ln(L)）。
#   原因：训练集样本量达 24.6 万，BIC 的惩罚系数 ln(N) ≈ 12.4 过重，容易过早截断有效特征导致欠拟合；
#         AIC 惩罚项固定为 2，更适合大样本评分卡在拟合优度与变量精简度之间取得良好平衡。
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
# 【调参说明】
# - 剔除条件 coefs > 0（逐轮剔除最大正系数变量）：
#   说明：optbinning 的 WOE 定义为 ln(good% / bad%)，好客户占比越高则 WOE 越大。
#   原因：预测目标为违约（TARGET=1）时，正常逻辑回归系数应严格为负；正系数表明存在残余共线性引发的风险方向倒挂，
#         逐个剔除正系数最大者并重拟合，可保证每个入模特征的评分方向与风险含义完全一致。
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
#   ⚠️ 发现符号反转变量 (系数为正): EXT_SOURCE_MAX，予以剔除以防风险倒挂

del X_train_woe
gc.collect()

print(f"✓ 逐步回归与符号校验完成！最终入模特征数: {len(final_model_feats)}")
# ✓ 逐步回归与符号校验完成！最终入模特征数: 26

# %%
# =============================================================================
# 5. 训练标准信用评分卡与 PDO 刻度映射
# =============================================================================
num_final_feats = [c for c in final_model_feats if c in num_features]
cat_final_feats = [c for c in final_model_feats if c in cat_features]

# 【调参说明 - 终模分箱器】
# 各项分箱超参数与第 3 步 fine_binning_process 完全保持一致，仅将特征范围收敛为最终通过校验的 26 个特征，
# 确保 Scorecard 内部端到端封装的分箱与 WOE 空间与逐步回归/符号检验阶段完全等价。
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
    special_codes=['XNA'],
    binning_fit_params={c: fine_binning_fit_params[c] for c in final_model_feats},
    selection_criteria=None,
    n_jobs=-1
)

# 【调参说明 - 基准好坏比】
# 使用训练集全局平均好坏比 (1-p)/p（约 11.39:1）作为 PDO 刻度的基准 Odds，
# 使违约风险处于全行平均水平的客群恰好对应 600 基准分，方便业务与基准风险水位对齐。
base_odds = (1 - y_train.mean()) / y_train.mean()

# 【调参说明 - 逻辑回归估计器】
lr_estimator = LogisticRegression(
    penalty=None,        # 关闭正则化（无偏极大似然估计）：前期已经过相关性、VIF <= 5、AIC 与符号检查充分降维，无需正则惩罚压缩系数；无偏估计才能使对数几率与 PDO 分数刻度严格对应
    solver='lbfgs',      # 拟牛顿法求解器：原生支持 penalty=None，且在 26 维连续 WOE 特征、24.6 万样本下收敛快、数值稳定
    max_iter=1000,       # 最大迭代次数提高至 1000（默认 100）：确保配合高精度 tol=1e-6 时优化器有足够步数完全收敛，避免早停警告或系数偏差
    tol=1e-6,            # 收敛容差收紧至 1e-6（默认 1e-4）：提高回归系数精度，防止小数偏差经 PDO 放大后影响整数分箱得分边界
    random_state=123     # 固定随机种子 123：与项目其余建模脚本一致，保证结果完全可复现
)

# 【调参说明 - 评分卡 PDO 刻度映射参数】
scorecard = Scorecard(
    binning_process=scorecard_bp,
    estimator=lr_estimator,
    scaling_method="pdo_odds",   # 采用金融评分卡标准“翻倍好坏比（PDO）”对数几率线性刻度映射
    scaling_method_params={
        "pdo": 20,               # PDO = 20：好坏比每翻倍一次，评分增加 20 分。20 分是风控行业经典惯例，能拉开适中分数跨度且方便按每 20 分一档划定审批阈值
        "odds": base_odds,       # 基准好坏比锚定为训练集平均好坏比
        "scorecard_points": 600  # 基准分设为 600 分：行业通用信用分中心锚点，使总分自然落在约 400~800 区间，符合风控业务认知习惯
    },
    rounding=True        # 分箱得分四舍五入取整：生成简洁可落地的整数评分卡规则表，便于业务系统部署与人工复核
)

del base_odds, lr_estimator, scorecard_bp, fine_binning_fit_params

# Scorecard.fit 接收训练集的原始特征值
print("正在拟合标准评分卡并执行 PDO=20, 600 基准分刻度映射...")
# 正在拟合标准评分卡并执行 PDO=20, 600 基准分刻度映射...
scorecard.fit(
    train_df[final_model_feats],
    y_train,
    metric_special="empirical",  # 与第 3 步一致，特殊值 'XNA' 采用箱内经验 WOE 参与回归与计分
    metric_missing="empirical",  # 与第 3 步一致，缺失值采用箱内经验 WOE 参与回归与计分，完整保留缺失风险信号
)

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
# 【调参说明】
# - method='quantile'（等频分箱）：保证训练集基准下各分箱占比均约 10%，避免等距分箱在得分两端的长尾稀疏区出现空箱或极小占比导致 PSI 计算失真。
# - n_bins=10（10 个十分位箱）：风控领域监测总分分布稳定性（PSI）的标准粒度，兼顾分布变化敏感度与每箱样本充足性。
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
# 训练集 (Train) 246008 0.7510 0.3733 基准 (0.00000)
# 验证集 (Valid)  61503 0.7456 0.3655            -
#  测试集 (Test)  48744      -      -      0.00133
# ======================================================================

del train_df, valid_df, test_df, y_train, y_valid, train_prob, valid_prob, train_score, test_score
del scorecard, score_combiner, eval_summary_df, scorecard_table
gc.collect()
