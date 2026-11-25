# %%
# =============================================================================
# 探索性数据分析（EDA）：基于训练集 Train
# =============================================================================
# 数据隔离要求：
# 1. 仅读取 data/interim/application_train_split.csv 进行 EDA；验证集
#    application_valid.csv 不参与统计与规则拟合，避免信息泄露。
# 2. 测试集 application_test.csv 仅用于字段和类型对齐校验，不参与单变量统计。
# 3. 样本量、主键唯一性及切分质量审计已在 00_stratified_split.py 完成，此处不重复。
# =============================================================================

# 库
import gc
from pathlib import Path

import pandas as pd

from optbinning import BinningProcess

# 文件路径
DATA_PATH = Path.cwd() / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
INTERIM_DATA_PATH = DATA_PATH / "interim"
OUTPUTS_PATH = Path.cwd() / "outputs"
EDA_OUTPUT_PATH = OUTPUTS_PATH / "eda"
EDA_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 主申请表：训练集（00_stratified_split.py 的输出）与测试集
train = pd.read_csv(INTERIM_DATA_PATH / 'application_train_split.csv')
test = pd.read_csv(RAW_DATA_PATH / 'application_test.csv')

application_feature_descriptions = pd.read_csv(
    RAW_DATA_PATH / 'HomeCredit_columns_description.csv',
    usecols=['Table', 'Row', 'Description'],
    encoding='latin1'
)
application_feature_descriptions = application_feature_descriptions.loc[
    application_feature_descriptions['Table'] == 'application_{train|test}.csv'
].reset_index(drop=True)

# %%
"一、训练样本规模与测试集字段对齐（Sample & Schema Check）"
# 样本量、主键唯一性、训练集/验证集无重叠、坏账率一致性等切分校验
# 已由 00_stratified_split.py 完成，审计结果见
# outputs/validation/split_audit.csv；本脚本不再重复校验。

# -- 数据规模 --
train.shape
# (246008, 122)
# 训练集为原始训练表按 TARGET 分层抽取的 80%，共 246008 条、122 个字段

test.shape
# (48744, 121)
# test 样本量为 48744，含有 121 个字段

# 检查测试集字段是否恰好等于训练集业务特征字段
set(test.columns) == (set(train.columns) - {'TARGET'})
# True
# 字段一致，仅相差标签列 TARGET

# -- 测试集重复性 --
# 训练集的重复样本与主键校验随全量主键唯一性检查一并完成，此处不再重复
test.duplicated().any()
# False
" -> 测试集无重复样本"

# %%
"二、数据质量与单变量概览（Data Quality & Univariate Profiling）"
# 本节仅统计字段质量与分布特征，不直接修改或剔除字段；
# 具体清洗策略在后续特征工程中结合业务定义处理。

# -- 数据类型与字段说明 --
cols_train_types = train.dtypes.rename('dtypes')
cols_test_types = test.dtypes.rename('dtypes')

raw_features_summary = application_feature_descriptions.copy()
raw_features_summary['dtypes'] = raw_features_summary['Row'].map(cols_train_types)

# 检查训练集与测试集业务特征字段类型是否一致
cols_train_types.drop('TARGET').equals(cols_test_types)
# True

# 主键不作为业务特征，标签也不进入单变量特征质量筛查
feature_columns = [
    column for column in train.columns
    if column not in {'SK_ID_CURR', 'TARGET'}
]
numeric_feature_columns = train[feature_columns].select_dtypes(
    include='number'
).columns.tolist()

# -- 形成单变量质量汇总：缺失率、最大同值率、分位数 --
# 筛查标准：缺失率 >90% 或同值率 >95% 列入复核清单，需结合业务含义决定是否保留。
univariate_summary = pd.DataFrame(index=feature_columns)
univariate_summary.index.name = 'Row'
univariate_summary['missing_rate'] = train[feature_columns].isna().mean()
univariate_summary['primary_value_ratio'] = pd.Series({
    column: train[column].value_counts(dropna=False).iloc[0] / len(train)
    for column in feature_columns
})

numeric_moments = train[numeric_feature_columns].agg(
    ['count', 'mean', 'std']
).T
univariate_summary = univariate_summary.join(numeric_moments)

# -- 异常值识别 --
# 计算数值字段的0.01、0.05、0.5、0.95和0.99分位数
numeric_quantiles = train[numeric_feature_columns].quantile(
    [0.01, 0.05, 0.50, 0.95, 0.99]
).T
numeric_quantiles.columns = ['p1', 'p5', 'p50', 'p95', 'p99']
univariate_summary = univariate_summary.join(numeric_quantiles)

raw_features_summary = raw_features_summary.merge(
    univariate_summary.reset_index(),
    on='Row',
    how='left'
)

# 释放字段类型说明、测试集与单变量质量汇总中间变量
del application_feature_descriptions, cols_train_types, cols_test_types, test
del feature_columns, numeric_feature_columns, numeric_moments, numeric_quantiles, univariate_summary
gc.collect()

# 在 DAYS_EMPLOYED 中发现 365243 特殊编码
# 分位数：p1=-10900, p5=-6744, p50=-1216, p95=365243, p99=365243
# 说明：365243 为特殊标记值（约 1000 年），后续特征工程中需转为 NaN 并保留异常标记。

# -- 缺失率分析 --
missing_summary = train.isna().mean().reset_index()
missing_summary.columns = ['Row', 'missing_rate']

# 没有缺失的列
present_columns = missing_summary.loc[missing_summary['missing_rate'] == 0, 'Row'].tolist()

# 缺失的列
# 保留完整汇总表，另用非零缺失率子集进行后续缺失值分析
missing_summary_nonzero = missing_summary[
    missing_summary['missing_rate'] > 0
].reset_index(drop=True)
missing_columns = missing_summary_nonzero['Row'].tolist()

missing_summary_nonzero.sort_values(by='missing_rate', ascending=False).head(20)
#                          Row  missing_rate
# 27           COMMONAREA_MODE      0.698473
# 13            COMMONAREA_AVG      0.698473
# 41           COMMONAREA_MEDI      0.698473
# 49  NONLIVINGAPARTMENTS_MEDI      0.694278
# 35  NONLIVINGAPARTMENTS_MODE      0.694278
# 21   NONLIVINGAPARTMENTS_AVG      0.694278
# 51        FONDKAPREMONT_MODE      0.683681
# 47     LIVINGAPARTMENTS_MEDI      0.683518
# 19      LIVINGAPARTMENTS_AVG      0.683518
# 33     LIVINGAPARTMENTS_MODE      0.683518
# 31            FLOORSMIN_MODE      0.678315
# 17             FLOORSMIN_AVG      0.678315
# 45            FLOORSMIN_MEDI      0.678315
# 40          YEARS_BUILD_MEDI      0.664791
# 26          YEARS_BUILD_MODE      0.664791
# 12           YEARS_BUILD_AVG      0.664791
# 3                OWN_CAR_AGE      0.659853
# 46             LANDAREA_MEDI      0.593595
# 32             LANDAREA_MODE      0.593595
# 18              LANDAREA_AVG      0.593595

# 缺失率相同的字段组（由当前训练集统计，按缺失率降序）
# 缺失率相同只表示缺失比例相同，缺失位置是否一致由下方逻辑继续核验
# 69.8473%: COMMONAREA_AVG, COMMONAREA_MODE, COMMONAREA_MEDI
# 69.4278%: NONLIVINGAPARTMENTS_AVG, NONLIVINGAPARTMENTS_MODE, NONLIVINGAPARTMENTS_MEDI
# 68.3518%: LIVINGAPARTMENTS_AVG, LIVINGAPARTMENTS_MODE, LIVINGAPARTMENTS_MEDI
# 67.8315%: FLOORSMIN_AVG, FLOORSMIN_MODE, FLOORSMIN_MEDI
# 66.4791%: YEARS_BUILD_AVG, YEARS_BUILD_MODE, YEARS_BUILD_MEDI
# 59.3595%: LANDAREA_AVG, LANDAREA_MODE, LANDAREA_MEDI
# 58.4859%: BASEMENTAREA_AVG, BASEMENTAREA_MODE, BASEMENTAREA_MEDI
# 55.1271%: NONLIVINGAREA_AVG, NONLIVINGAREA_MODE, NONLIVINGAREA_MEDI
# 53.2507%: ELEVATORS_AVG, ELEVATORS_MODE, ELEVATORS_MEDI
# 50.7154%: APARTMENTS_AVG, APARTMENTS_MODE, APARTMENTS_MEDI
# 50.2988%: ENTRANCES_AVG, ENTRANCES_MODE, ENTRANCES_MEDI
# 50.1431%: LIVINGAREA_AVG, LIVINGAREA_MODE, LIVINGAREA_MEDI
# 49.7240%: FLOORSMAX_AVG, FLOORSMAX_MODE, FLOORSMAX_MEDI
# 48.7525%: YEARS_BEGINEXPLUATATION_AVG, YEARS_BEGINEXPLUATATION_MODE, YEARS_BEGINEXPLUATATION_MEDI
# 13.5731%: AMT_REQ_CREDIT_BUREAU_HOUR, AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_WEEK,
#           AMT_REQ_CREDIT_BUREAU_MON, AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_YEAR
# 0.3301%: OBS_30_CNT_SOCIAL_CIRCLE, DEF_30_CNT_SOCIAL_CIRCLE,
#          OBS_60_CNT_SOCIAL_CIRCLE, DEF_60_CNT_SOCIAL_CIRCLE

# 进一步探究缺失模式
for rate, group in missing_summary_nonzero.groupby('missing_rate'):
    if len(group) > 1:
        temp_list = group['Row'].tolist()
        temp_df = train[temp_list]
        
        if len(temp_df.isna().T.drop_duplicates()) != 1:
            print(f"{temp_list}的缺失模式不同")

# 没有输出结果

# 校验结果：
# 相同缺失率的字段组内，样本层面的缺失位置完全重合（同缺失、同有值），
# 表明这些字段通常来自同一数据采集来源。
# 特征工程中同组字段只需保留一个缺失标记衍生变量；原字段数值仍按各自特征保留。

missing_summary_nonzero = missing_summary_nonzero.merge(
    raw_features_summary[['Row', 'Description']],
    on='Row',
    how='left')

# 缺失率较高的特征主要集中在房屋标准化信息字段，属于非随机缺失（未填报或无相关资产）。

raw_features_summary.to_csv(EDA_OUTPUT_PATH / 'raw_features_summary.csv', index=False)

# 清理缺失率统计与模式核验中间变量
del missing_summary, present_columns, missing_summary_nonzero, missing_columns
del rate, group, temp_list, temp_df
gc.collect()

# %%
"三、双变量分析与风险单调性检验（Bivariate & Risk Discrimination）"
# 对 TARGET 以外的全部业务特征统一最优分箱与 IV 统计摸底，再按 IV 降序查看
# 使用 optbinning 批量最优分箱，缺失值与特殊编码单独分箱
# 摸底统计阶段仅计算基准 IV，不剔除特征

"3.1 全量业务特征 IV 计算与降序排名"
# 1. 区分特征列表与类别型/数值型变量
ignore_cols = ['SK_ID_CURR', 'TARGET']
feature_cols = [c for c in train.columns if c not in ignore_cols]
cat_cols = train[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in feature_cols if c not in cat_cols]

# 2. 配置特殊值编码（Home Credit 原始表尚未清洗，365243 代表退休/无业特殊编码，XNA 代表未知类别）
special_codes = [365243, 'XNA']

# 3. 配置底层分箱规则（透传给每个特征的 OptimalBinning）
# EDA 阶段底层分箱配置：放开单调性限制，允许发现峰值/谷值以评估理论最大 IV
eda_fit_params = {
    col: {
        "monotonic_trend": "auto",          # 设为 auto 允许单峰/单谷，便于摸底特征的最大可达 IV
    }
    for col in num_cols
}
for col in cat_cols:
    eda_fit_params[col] = {
        "cat_cutoff": 0.01                  # 占比低于 1% 的长尾稀有类别合并为 others，避免压低高基数特征 IV
    }

# 4. 实例化 EDA 专用 BinningProcess
binning_process = BinningProcess(
    variable_names=feature_cols,
    categorical_variables=cat_cols,
    max_n_prebins=30,                            # 预分箱上限与 min_prebin_size 配合，使 min_bin_size=0.02 生效
    min_prebin_size=0.02,                        # 预分箱最小样本占比设为 2%，避免被默认 5% 架空
    max_n_bins=8,                                # 最大分箱数（EDA 阶段设为 8 箱观察细分趋势）
    min_bin_size=0.02,                           # 单箱最小样本占比设为 2% 捕捉尾部样本
    special_codes=special_codes,                 # 特殊编码单独成箱
    binning_fit_params=eda_fit_params,
    selection_criteria=None,                     # EDA 阶段不剔除变量，全量输出体检结果
    n_jobs=-1
)

binning_process.fit(
    train[feature_cols],
    train['TARGET']
)

# summary() 提供各变量的 IV、分箱数及求解状态
iv_summary = binning_process.summary().rename(columns={'name': 'feature'})

iv_summary = iv_summary.sort_values(by='iv', ascending=False).reset_index(drop=True)

# 结果加入 raw_features_summary
raw_features_summary = raw_features_summary.join(iv_summary.set_index('feature'), on='Row')

raw_features_summary.to_csv(EDA_OUTPUT_PATH / 'raw_features_summary.csv', index=False)

# 清理本单元格的临时配置变量与中间表
del ignore_cols, num_cols, cat_cols, special_codes, eda_fit_params, iv_summary
gc.collect()

# %%
"3.2 具备预测力特征概览（IV >= 0.02）"
high_iv_features = raw_features_summary.loc[
    raw_features_summary['iv'] >= 0.02, ['Row', 'iv']
].sort_values(by='iv', ascending=False)
# 按iv值大小降序排列
#                              Row      iv
# 43                  EXT_SOURCE_3  0.3401
# 42                  EXT_SOURCE_2  0.3266
# 41                  EXT_SOURCE_1  0.1575
# 18                 DAYS_EMPLOYED  0.1162
# 10               AMT_GOODS_PRICE  0.0934
# 17                    DAYS_BIRTH  0.0883
# 28               OCCUPATION_TYPE  0.0780
#  8                    AMT_CREDIT  0.0622
# 40             ORGANIZATION_TYPE  0.0597
# 12              NAME_INCOME_TYPE  0.0562
# 13           NAME_EDUCATION_TYPE  0.0516
# 31   REGION_RATING_CLIENT_W_CITY  0.0499
# 95        DAYS_LAST_PHONE_CHANGE  0.0491
# 30          REGION_RATING_CLIENT  0.0465
# 20               DAYS_ID_PUBLISH  0.0431
#  3                   CODE_GENDER  0.0396
# 51                 FLOORSMAX_AVG  0.0382
# 79                FLOORSMAX_MEDI  0.0380
# 65                FLOORSMAX_MODE  0.0374
# 88                TOTALAREA_MODE  0.0367
# 16    REGION_POPULATION_RELATIVE  0.0362
#  9                   AMT_ANNUITY  0.0352
# 55                LIVINGAREA_AVG  0.0345
# 83               LIVINGAREA_MEDI  0.0344
# 23                FLAG_EMP_PHONE  0.0335
# 69               LIVINGAREA_MODE  0.0332
# 44                APARTMENTS_AVG  0.0319
# 49                 ELEVATORS_AVG  0.0317
# 72               APARTMENTS_MEDI  0.0317
# 77                ELEVATORS_MEDI  0.0314
# 58               APARTMENTS_MODE  0.0311
# 38        REG_CITY_NOT_WORK_CITY  0.0307
# 63                ELEVATORS_MODE  0.0305
# 97               FLAG_DOCUMENT_3  0.0294
# 60  YEARS_BEGINEXPLUATATION_MODE  0.0291
# 19             DAYS_REGISTRATION  0.0291
# 74  YEARS_BEGINEXPLUATATION_MEDI  0.0290
# 46   YEARS_BEGINEXPLUATATION_AVG  0.0290
# 50                 ENTRANCES_AVG  0.0270
# 78                ENTRANCES_MEDI  0.0268
# 89            WALLSMATERIAL_MODE  0.0260
# 64                ENTRANCES_MODE  0.0259
# 21                   OWN_CAR_AGE  0.0241
# 37        REG_CITY_NOT_LIVE_CITY  0.0231
# 90           EMERGENCYSTATE_MODE  0.0223
# 14            NAME_FAMILY_STATUS  0.0221
# 57             NONLIVINGAREA_AVG  0.0220
# 85            NONLIVINGAREA_MEDI  0.0218
# 87                HOUSETYPE_MODE  0.0218
# 45              BASEMENTAREA_AVG  0.0217
# 71            NONLIVINGAREA_MODE  0.0216
# 73             BASEMENTAREA_MEDI  0.0216
# 59             BASEMENTAREA_MODE  0.0208

# 清理高 IV 特征查看临时变量
del high_iv_features
gc.collect()

# %%
# 数据隔离说明：
# 1. EDA 仅基于训练集探索，不计算测试集的单变量 PSI，避免信息泄露；
# 2. 训练集与测试集的分布差异评估由 02_adversarial_validation.py 独立完成；
# 3. 模型总分与预测概率的跨客群 PSI 在下游建模（04/05 脚本）完成后评估。

# 清理训练集大表、分箱器及所有剩余变量，释放内存
del train, raw_features_summary, feature_cols, binning_process
gc.collect()

