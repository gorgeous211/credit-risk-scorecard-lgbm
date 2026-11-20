# %%
# 风控建模中最重要的8类变量

# 1. 还款能力变量（核心）
# 反映客户有没有钱还款，是信用风险最基础的判断维度。
# - 收入：月收入、工资流水、收入稳定性
# - 负债：当前负债、月还款额、负债收入比（DTI）
# - 现金流：月均流入、月均流出、账户余额、现金流波动
# - 资产：存款、房产、可变现资产
# 业务逻辑： 收入不稳定、负债过高、现金流持续恶化，通常意味着违约风险上升。

# 2. 历史信用表现变量（核心）
# 反映客户过去是否有履约意愿和履约能力。
# - 历史逾期次数、最大逾期天数
# - 历史 M1、M2、M3 逾期记录
# - 近 3/6/12 个月逾期次数
# - 历史结清情况、核销情况
# - 信用账户数量、信贷历史长度
# 业务逻辑： 近期发生的严重逾期，通常比很久以前的一次轻微逾期更值得关注。逾期的严重程度、频率和距今时间都很重要。

# 3. 多头借贷与负债压力变量
# 反映客户是否在多个机构借款，以及是否存在过度借贷。
# - 当前在贷机构数
# - 近 7/30/90 天贷款申请次数
# - 近 30 天新增借款次数
# - 总授信额度、已用额度、额度使用率
# - 多平台借款频率、短期借款增长率
# 业务逻辑： 短期内频繁申请贷款、多个平台同时新增借款，可能意味着资金压力增加。
# 注意：多头不一定等于高风险，要结合客户收入、存量负债和借款用途判断。

# 4. 申请行为与借款行为变量
# 反映客户在什么情况下申请贷款，以及借款行为是否异常。
# - 申请时间、申请频率
# - 申请金额、期限、选择的产品
# - 重复申请、短时间多次修改申请信息
# - 额度使用和支用行为
# - 借款后是否迅速再次借款
# 业务逻辑： 单次申请金额突然升高、短期反复申请，可能反映资金需求发生变化，但也可能是正常的消费或经营需求。

# 5. 身份真实性与反欺诈变量
# 反映客户是否真实、申请行为是否存在欺诈风险。
# - 设备关联账号数
# - IP、设备、手机号、身份证的关联关系
# - 注册到申请的时间间隔
# - 设备环境异常、模拟器或自动化行为
# - 身份信息与外部数据的一致性
# 业务逻辑： 同一设备关联大量无明显合理关系的申请人，可能存在团伙欺诈或批量注册风险。
# 这一类变量对反欺诈模型尤其重要，对纯粹的违约概率模型则要看数据和业务场景。

# 6. 账户与交易行为变量
# 反映客户近期财务状况、消费模式和行为变化。
# - 日均余额、余额波动率
# - 入账金额、出账金额、净现金流
# - 交易频次、交易金额分布
# - 工资入账规律、异常大额交易
# - 账户活跃度、交易对手集中度
# 业务逻辑： 比起单一时点的余额，持续多月的现金流趋势可能更有解释力。收入突然下降、余额持续减少等变化值得关注。

# 7. 产品与借款条件变量
# 反映客户承担的还款压力，以及不同产品本身的风险差异。
# - 借款金额、期限、利率
# - 月供金额、还款方式
# - 额度、首借或复借
# - 抵押物价值、贷款价值比（LTV）
# - 产品类型、渠道、资金方
# 业务逻辑： 相同的客户在不同贷款金额、期限和月供下，可能面临不同的违约风险。
# 特别注意：产品、渠道等变量可能捕捉到的是业务准入规则或历史客群差异，不一定是客户自身的风险特征。

# 8. 时间变化与趋势变量
# 这是很多建模人员容易低估的一类变量。
# - 近 7/30/90 天申请次数
# - 近 3 个月负债增长率
# - 最近一次逾期距今天数
# - 收入环比变化
# - 近期与历史平均行为的偏离程度
# 业务逻辑： 客户当前的风险状态可能比几年前的静态画像更重要。
# 例如，同样有 5 笔贷款，一个客户负债稳定且按时还款，另一个客户近一个月突然新增 4 笔贷款，两者的风险可能完全不同。

# %%
# =============================================================================
# 探索性数据分析（EDA）：只基于开发训练集 Train Dev
# =============================================================================
# 数据隔离原则（详见 00_stratified_split.py）：
# 1. 训练数据只读取 data/interim/application_train_dev.csv，即 00 脚本按
#    8:2 分层抽样得到的开发训练集；全量原始训练表 application_train.csv
#    和闭卷验证集 application_valid_blind.csv 均不参与 EDA，避免验证集信息
#    渗入缺失率统计、分箱拟合、IV 计算、相关性筛选等任何规则拟合环节。
# 2. 官方未标注测试集 application_test.csv 没有 TARGET 标签，仅作为跨客群
#    分布参照，用于计算开发集与官方测试集之间的 PSI 稳定性指标，
#    不参与任何有监督拟合。
# 3. 样本量、主键唯一性、开发集/验证集无重叠、坏账率一致性等切分校验
#    已由 00_stratified_split.py 完成并落盘到
#    outputs/validation/split_audit.csv，本脚本不再重复。
# =============================================================================

# 库
import gc
from pathlib import Path

import pandas as pd
import toad

from optbinning import BinningProcess

# 文件路径
DATA_PATH = Path.cwd() / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
INTERIM_DATA_PATH = DATA_PATH / "interim"
OUTPUTS_PATH = Path.cwd() / "outputs"
EDA_OUTPUT_PATH = OUTPUTS_PATH / "eda"
EDA_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 主申请表：开发训练集（00_stratified_split.py 的输出）与官方未标注测试集
train_dev = pd.read_csv(INTERIM_DATA_PATH / 'application_train_dev.csv')
application_test = pd.read_csv(RAW_DATA_PATH / 'application_test.csv')

cols_meaning = pd.read_csv(
    RAW_DATA_PATH / 'HomeCredit_columns_description.csv',
    usecols=['Table', 'Row', 'Description'],
    encoding='latin1'
)
cols_meaning = cols_meaning.loc[
    cols_meaning['Table'] == 'application_{train|test}.csv'
].reset_index(drop=True)

# %%
"一、开发样本规模与测试集字段对齐（Sample & Schema Check）"
# 样本量、主键唯一性、开发集/验证集无重叠、坏账率一致性等切分校验
# 已由 00_stratified_split.py 完成，审计结果见
# outputs/validation/split_audit.csv；本脚本不再重复校验。

# -- 数据规模 --
train_dev.shape
# (246008, 122)
# 开发训练集为原始训练表按 TARGET 分层抽取的 80%，共 246008 条、122 个字段

application_test.shape
# (48744, 121)
# application_test 样本量为48744，含有121个字段

# 检查测试集字段是否恰好等于开发集业务特征字段
set(application_test.columns) == (set(train_dev.columns) - {'TARGET'})
# True
# 字段一致，仅相差标签列 TARGET

# -- 官方测试集重复性 --
# 开发集的重复样本与主键校验随全量主键唯一性检查一并完成，此处不再重复
application_test.duplicated().any()
# False
" -> 官方测试集无重复样本"

# %%
"二、数据质量与单变量清洗（Univariate & Data Quality）"

# -- 数据类型与字段说明 --
cols_dev_types = train_dev.dtypes.rename('dtypes')
cols_test_types = application_test.dtypes.rename('dtypes')

cols_eda = cols_meaning.copy()
cols_eda['dtypes'] = cols_eda['Row'].map(cols_dev_types)

# 检查开发集与测试集业务特征字段类型是否一致
cols_dev_types.drop('TARGET').equals(cols_test_types)
# True

# 主键不作为业务特征，标签也不进入单变量特征质量筛查
feature_columns = [
    column for column in train_dev.columns
    if column not in {'SK_ID_CURR', 'TARGET'}
]
numeric_feature_columns = train_dev[feature_columns].select_dtypes(
    include='number'
).columns.tolist()

# -- 形成单变量质量汇总：缺失率、最大同值率、分位数 --
# 阈值只用于待审查清单：缺失率 >90%、最大同值率 >95% 优先复核；不能脱离业务含义直接删特征。
univariate_summary = pd.DataFrame(index=feature_columns)
univariate_summary.index.name = 'Row'
univariate_summary['missing_rate'] = train_dev[feature_columns].isna().mean()
univariate_summary['primary_value_ratio'] = pd.Series({
    column: train_dev[column].value_counts(dropna=False).iloc[0] / len(train_dev)
    for column in feature_columns
})

numeric_moments = train_dev[numeric_feature_columns].agg(
    ['count', 'mean', 'std']
).T
univariate_summary = univariate_summary.join(numeric_moments)

# -- 异常值识别 --
# 计算数值字段的0.01、0.05、0.5、0.95和0.99分位数
numeric_quantiles = train_dev[numeric_feature_columns].quantile(
    [0.01, 0.05, 0.50, 0.95, 0.99]
).T
numeric_quantiles.columns = ['p1', 'p5', 'p50', 'p95', 'p99']
univariate_summary = univariate_summary.join(numeric_quantiles)

cols_eda = cols_eda.merge(
    univariate_summary.reset_index(),
    on='Row',
    how='left'
)

# 清理字段类型说明与单变量质量汇总中间变量
del cols_meaning, cols_dev_types, cols_test_types
del feature_columns, numeric_feature_columns, numeric_moments, numeric_quantiles, univariate_summary
gc.collect()

# 在 DAYS_EMPLOYED 中发现 365243 特殊编码

# -- 缺失率分析 --
missing_summary = train_dev.isna().mean().reset_index()
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

# 缺失率相同的字段组（由当前开发集统计，按缺失率降序）
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
        temp_df = train_dev[temp_list]
        
        if len(temp_df.isna().T.drop_duplicates()) != 1:
            print(f"{temp_list}的缺失模式不同")

# 没有输出结果

" -> 相同缺失率的字段缺失模式都相同"
" 每个缺失率相同的字段组内部，字段的缺失标记在每条样本上都完全一致 "
" 也就是说，一条样本中组内字段要么全都有值，要么全缺失；不会出现组内某个字段缺失、另一个字段有值的情况 "

" -> 这提示这些字段的可用性可能受同一个数据采集或数据来源条件影响 "
" 对特征工程来说，组内的缺失指示变量会重复表达同一信息，通常一个组保留一个 is_missing 标记就够了；"
" 原字段的值仍可能各自有用，不应仅因缺失模式相同就删除 "

missing_summary_nonzero = missing_summary_nonzero.merge(
    cols_eda[['Row', 'Description']],
    on='Row',
    how='left')

''' -> 缺失率较高的字段大部分是与房屋相关
    -> 说明一部分申请人缺少相关房屋信息，而不是数据随机缺失 '''

cols_eda.to_csv(EDA_OUTPUT_PATH / 'cols_eda.csv', index=False)

# 清理缺失率统计与模式核验中间变量
del missing_summary, present_columns, missing_summary_nonzero, missing_columns
del rate, group, temp_list, temp_df
gc.collect()

# %%
"三、双变量分析与风险单调性检验（Bivariate & Risk Discrimination）"
# 先对 TARGET 以外的全部业务特征做统一初筛，再按 IV 降序查看，最后才附加业务类别
# 使用 optbinning 对数值和类别变量批量最优分箱，缺失与常见特殊编码单独处理
# 这是开发样本上的初筛 IV

# %%
"3.1 全量业务特征 IV 初筛与降序排名"
# 1. 区分特征列表与类别型变量
ignore_cols = ['SK_ID_CURR', 'TARGET']
feature_cols = [c for c in train_dev.columns if c not in ignore_cols]
cat_cols = train_dev[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

# 2. 配置特殊值编码（Home Credit 中 365243 代表退休/无业特殊编码，XNA 代表未知类别）
# 也可传入字典针对不同特征指定不同特殊值，如：{'DAYS_EMPLOYED': [365243], 'CODE_GENDER': ['XNA']}
special_codes = [365243, 'XNA']

# 3. 配置底层分箱规则（透传给每个特征的 OptimalBinning）
# EDA 阶段底层分箱配置：放开单调性限制，允许发现峰值(peak)/谷值(valley)
eda_fit_params = {
    col: {
        "monotonic_trend": "auto",          # 强制严格单调递增或递减（若允许U型可设为 "auto"）
                                            # EDA 摸底阶段设置为auto，发现真实的业务规律以及快速算出特征的理论最大 IV 上限
        #"min_n_bins":                      # 这里不设置最小分箱数量，防范高同值率特征导致求解器报错或强行切分，
                                            # 并直接利用 n_bins == 1 快速筛出无区分度的废特征
        "cat_cutoff": 0.05                  # 类别型变量中占比低于 5% 的长尾稀有类别自动合并为 "others"
    } 
    for col in feature_cols
}

# 4.实例化 EDA 专用 BinningProcess
binning_process = BinningProcess(
    variable_names=feature_cols,                 # 需要批量分箱的特征列名列表
    categorical_variables=cat_cols,              # 显式指定类别型或离散状态特征
    max_n_bins=8,                                # 最大分箱数（不含缺失箱和特殊值箱），评分卡通常限制在 5~6 箱以防过拟合
                                                 # EDA 阶段切 8~10 箱，观察更细腻的风险趋势
    min_bin_size=0.02,                           # 单个分箱的最小样本占比，一般设置5%，放宽至 2%，捕捉尾部极端风险
    special_codes=special_codes,                 # 特殊编码列表
                                                 # 命中这些值的样本会脱离常规分箱
    binning_fit_params=eda_fit_params,
    selection_criteria=None,                     # EDA 阶段不剔除任何变量，全量输出体检结果
    n_jobs=-1
)

binning_process.fit(
    train_dev[feature_cols],
    train_dev['TARGET']
)

# summary() 提供各变量的 IV、分箱数及求解状态
iv_summary = binning_process.summary().rename(columns={'name': 'feature'})

iv_summary = iv_summary.sort_values(by='iv', ascending=False).reset_index(drop=True)

# 结果加入cols_eda
cols_eda = cols_eda.join(iv_summary.set_index('feature'), on='Row')

cols_eda.to_csv(EDA_OUTPUT_PATH / 'cols_eda.csv', index=False)

# 清理本单元格的临时配置变量与中间表；保留 IV 结果及后续 PSI 所需对象。
del ignore_cols, cat_cols, special_codes, eda_fit_params, iv_summary
gc.collect()

# %%
"3.2 IV 达标特征清单"
high_iv_features = cols_eda.loc[
    cols_eda['iv'] >= 0.02, ['Row', 'iv']
].sort_values(by='iv', ascending=False)
# 按iv值大小降序排列
#                              Row      iv
# 43                  EXT_SOURCE_3  0.3363
# 42                  EXT_SOURCE_2  0.3225
# 41                  EXT_SOURCE_1  0.1472
# 18                 DAYS_EMPLOYED  0.1154
# 10               AMT_GOODS_PRICE  0.0924
# 17                    DAYS_BIRTH  0.0861
# 28               OCCUPATION_TYPE  0.0600
#  8                    AMT_CREDIT  0.0582
# 12              NAME_INCOME_TYPE  0.0562
# 13           NAME_EDUCATION_TYPE  0.0506
# 31   REGION_RATING_CLIENT_W_CITY  0.0499
# 95        DAYS_LAST_PHONE_CHANGE  0.0485
# 30          REGION_RATING_CLIENT  0.0465
# 40             ORGANIZATION_TYPE  0.0433
# 20               DAYS_ID_PUBLISH  0.0426
#  3                   CODE_GENDER  0.0396
# 51                 FLOORSMAX_AVG  0.0371
# 79                FLOORSMAX_MEDI  0.0370
# 88                TOTALAREA_MODE  0.0364
# 65                FLOORSMAX_MODE  0.0363
# 16    REGION_POPULATION_RELATIVE  0.0346
# 55                LIVINGAREA_AVG  0.0342
# 83               LIVINGAREA_MEDI  0.0341
# 23                FLAG_EMP_PHONE  0.0335
# 69               LIVINGAREA_MODE  0.0329
# 49                 ELEVATORS_AVG  0.0317
# 77                ELEVATORS_MEDI  0.0314
# 72               APARTMENTS_MEDI  0.0312
# 44                APARTMENTS_AVG  0.0311
#  9                   AMT_ANNUITY  0.0308
# 38        REG_CITY_NOT_WORK_CITY  0.0307
# 58               APARTMENTS_MODE  0.0306
# 63                ELEVATORS_MODE  0.0305
# 97               FLAG_DOCUMENT_3  0.0294
# 60  YEARS_BEGINEXPLUATATION_MODE  0.0286
# 19             DAYS_REGISTRATION  0.0285
# 74  YEARS_BEGINEXPLUATATION_MEDI  0.0284
# 46   YEARS_BEGINEXPLUATATION_AVG  0.0283
# 50                 ENTRANCES_AVG  0.0269
# 78                ENTRANCES_MEDI  0.0267
# 64                ENTRANCES_MODE  0.0258
# 89            WALLSMATERIAL_MODE  0.0242
# 90           EMERGENCYSTATE_MODE  0.0232
# 37        REG_CITY_NOT_LIVE_CITY  0.0231
# 21                   OWN_CAR_AGE  0.0226
# 14            NAME_FAMILY_STATUS  0.0221
# 57             NONLIVINGAREA_AVG  0.0219
# 87                HOUSETYPE_MODE  0.0218
# 85            NONLIVINGAREA_MEDI  0.0217
# 45              BASEMENTAREA_AVG  0.0215
# 71            NONLIVINGAREA_MODE  0.0215
# 73             BASEMENTAREA_MEDI  0.0215
# 59             BASEMENTAREA_MODE  0.0207

# 清理 IV 达标特征清单临时变量
del high_iv_features
gc.collect()

# %%
"四、跨时间稳定性与数据穿越排查（Stability & Leakage Check）"
# 由 02_adversarial_validation.py 的对抗验证结果，AUC=0.89 表明官方测试集与训练数据存在强烈的分布偏移
#                        feature     importance
# 6                   AMT_CREDIT  126529.919421
# 7                  AMT_ANNUITY   90629.472365
# 118  AMT_REQ_CREDIT_BUREAU_QRT   85252.653770
# 117  AMT_REQ_CREDIT_BUREAU_MON   70770.088168
# 8              AMT_GOODS_PRICE   54966.208739
# 0           NAME_CONTRACT_TYPE   36930.647128
# 25                  FLAG_EMAIL   32490.213146
# 39                EXT_SOURCE_1   24006.952228
# 38           ORGANIZATION_TYPE   23038.961704
# 93      DAYS_LAST_PHONE_CHANGE   20621.832702

# 补充单变量的PSI：用开发集拟合出的同一套分箱转换开发集和官方测试集
train_bins = pd.DataFrame(
    binning_process.transform(
        train_dev[feature_cols],
        metric="bins"
    ),
    columns=feature_cols,
    index=train_dev.index
)
test_bins = pd.DataFrame(
    binning_process.transform(
        application_test[feature_cols],
        metric="bins"
    ),
    columns=feature_cols,
    index=application_test.index
)


# 在相同分箱标签上直接计算 PSI（toad 参数顺序为 test、base）。
# 【注意 / Warning】: 此处计算的单特征 PSI 仅供宏观探索分析与数据分布体检报告（只读参照），
# 严禁将此处的测试集单特征 PSI 传递到下游特征工程与特征筛选环节作为剔除门槛，以防数据泄露。
psi_summary = toad.metrics.PSI(
    test_bins[feature_cols],
    train_bins[feature_cols]
).sort_values(ascending=False)

# 添加结果到cols_eda
cols_eda['psi'] = cols_eda['Row'].map(psi_summary)

cols_eda.to_csv(EDA_OUTPUT_PATH / 'cols_eda.csv', index=False)

# 清理 PSI 计算使用的临时分箱数据、已完成使命的分箱器、官方测试集与 PSI 汇总序列，释放内存
del train_bins, test_bins, binning_process, application_test, psi_summary
gc.collect()

# 清理开发集大表及所有剩余变量，释放内存（初筛与共线性分析已移至 03_feature_engineering.py）
del train_dev, cols_eda, feature_cols
gc.collect()

