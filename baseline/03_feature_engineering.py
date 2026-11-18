# %%
# =============================================================================
# 特征工程与特征筛选（Feature Engineering & Selection）
# =============================================================================
# 数据隔离原则（严格遵守 00_stratified_split.py 规范）：
# 1. 训练数据只读取 data/interim/application_train_dev.csv，即 00 脚本按 8:2 分层抽样
#    得到的开发训练集（246,008 条，占比 80%）；
# 2. 全量原始训练表 application_train.csv 和闭卷验证集 application_valid_blind.csv 
#    严禁参与本阶段的特征衍生规则拟合、分箱拟合（fit）、IV 计算与多变量相关性筛选，
#    彻底杜绝数据穿越（Data Leakage）；
# 3. 官方未标注测试集 application_test.csv 没有 TARGET 标签，仅作为跨客群分布参照，
#    用于计算开发集与官方测试集之间的 PSI 稳定性指标；
# 4. 逻辑回归评分卡通常只保留 10~20 个核心特征。塞入大量弱特征不仅无法提升 KS 和 AUC，
#    反而会引入噪声、增加共线性风险，导致权重失真甚至符号反转。
# =============================================================================

# %%
# 导入库
import gc
import os
import sys
from pathlib import Path

# Windows 环境下确保当前虚拟环境的 Library/bin 优先加载，防止 LAPACK/OpenBLAS DLL 符号解析冲突
dll_path = os.path.join(sys.prefix, "Library", "bin")
if os.path.exists(dll_path):
    os.add_dll_directory(dll_path)
    os.environ["PATH"] = dll_path + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import pandas as pd
import toad
from optbinning import BinningProcess

# 文件路径
DATA_PATH = Path.cwd() / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
INTERIM_DATA_PATH = DATA_PATH / "interim"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
OUTPUTS_PATH = Path.cwd() / "outputs"
EDA_OUTPUT_PATH = OUTPUTS_PATH / "eda"
EDA_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 主申请表：根据 00_stratified_split.py 约定，仅读取开发训练集与官方测试集
train_dev = pd.read_csv(INTERIM_DATA_PATH / 'application_train_dev.csv')
application_test = pd.read_csv(RAW_DATA_PATH / 'application_test.csv')

# 读取 EDA 字段说明表，提取标准化房屋信息相关列
cols_eda = pd.read_csv(EDA_OUTPUT_PATH / 'cols_eda.csv')
house_cols = cols_eda.loc[
    cols_eda['Description'].str.startswith('Normalized information'), 'Row'
].tolist()

# %%
# =============================================================================
# 一、特殊异常值替换与基础清洗
# =============================================================================
# -----------------------------------------------------------------------------
# 1. FLAG_DAYS_EMPLOYED_ANOMALOUS（异常工龄标志位/退休客群标记）
#    - 含义：EDA 发现 DAYS_EMPLOYED 中存在 365243 这一特定编码，代表退休人员或无正规就业记录者。
#    - 业务逻辑：将原字段的 365243 替换为 np.nan，避免扭曲连续型工龄分布与单调性；
#      同时提取二值标记位保留客群身份（退休人群通常坏账率较低且收入相对稳定）。
# -----------------------------------------------------------------------------
train_dev['FLAG_DAYS_EMPLOYED_ANOMALOUS'] = (train_dev['DAYS_EMPLOYED'] == 365243).astype(int)
train_dev['DAYS_EMPLOYED'] = train_dev['DAYS_EMPLOYED'].replace(365243, np.nan)

application_test['FLAG_DAYS_EMPLOYED_ANOMALOUS'] = (application_test['DAYS_EMPLOYED'] == 365243).astype(int)
application_test['DAYS_EMPLOYED'] = application_test['DAYS_EMPLOYED'].replace(365243, np.nan)

# CODE_GENDER：将 4 个 'XNA' 样本替换为众数 'F'
train_dev['CODE_GENDER'] = train_dev['CODE_GENDER'].replace('XNA', 'F')
application_test['CODE_GENDER'] = application_test['CODE_GENDER'].replace('XNA', 'F')

# OBS_30_CNT_SOCIAL_CIRCLE、OBS_60_CNT_SOCIAL_CIRCLE 将 30 以上的极端离群值替换为 np.nan
# 这两个字段 75% 分位数仅为 2，均值 1.42，300+ 属于极罕见的异常数据错误
train_dev.loc[train_dev['OBS_30_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
train_dev.loc[train_dev['OBS_60_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_60_CNT_SOCIAL_CIRCLE'] = np.nan

application_test.loc[application_test['OBS_30_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
application_test.loc[application_test['OBS_60_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_60_CNT_SOCIAL_CIRCLE'] = np.nan

# %%
# =============================================================================
# 二、高阶业务特征衍生（先在全量原始特征上衍生，参考 01_eda.py 结论）
# =============================================================================
def add_derived_features(df, house_cols_list):
    """
    全量特征衍生函数：
    在未筛选的原始完整数据上构造衍生特征，涵盖 01_eda.py 总结的八大类风控变量体系。
    基于实际数据 Schema 直接向量化计算，去除了冗余的空值/存在性条件判断。
    """
    # -------------------------------------------------------------------------
    # 1. 还款能力与负债压力变量（01_eda.py 核心类别 1 & 3）
    # -------------------------------------------------------------------------

    # [1.1] CREDIT_TO_INCOME_RATIO（信贷收入比 / 杠杆倍数）
    # 含义：贷款总金额与年收入的比值。反映申请人杠杆倍数，杠杆率越高，整体负债压力越大，抗风险能力越脆弱。
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    # [1.2] ANNUITY_TO_INCOME_RATIO（年金收入比 / DTI 负债收入比代理）
    # 含义：年还款额（年金）占借款人总年收入的比例。是风控中最基础的刚性偿债现金流负荷指标，比值过高极易引发违约。
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    # [1.3] FREE_INCOME_PROXY（扣除年金后的可用净现金流代理）
    # 含义：总收入扣除本笔借款年还款后的剩余流动资金规模。该值越大，借款人日常开销与突发意外的资金缓冲空间越充足。
    df['FREE_INCOME_PROXY'] = df['AMT_INCOME_TOTAL'] - df['AMT_ANNUITY']

    # [1.4] INCOME_PER_PERSON（家庭人均年收入）
    # 含义：总收入除以家庭成员人数。消除抚养人口规模对还款能力的稀释效应，衡量家庭实际人均富裕水平。
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    # [1.5] INCOME_PER_CHILD（抚养子女加权人均收入）
    # 含义：总收入除以 (1 + 子女数)。由于未成年子女属于纯支出、无收入群体，该比率直接刻画抚养负担下的边际储蓄能力。
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])

    # -------------------------------------------------------------------------
    # 2. 贷款条件与期限结构变量（01_eda.py 类别 4 & 7）
    # -------------------------------------------------------------------------

    # [2.1] ANNUITY_TO_CREDIT_RATIO（还款费率 / 期限倒数）
    # 含义：年金占信贷总额的比率。比率高反映贷款期限短、月供偿债节奏急促，或合同隐含较高的综合资金借贷利率。
    df['ANNUITY_TO_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # [2.2] ESTIMATED_TENURE（预估借款期数 / 还款期限代理）
    # 含义：信贷额度与年金之比，近似预估贷款年数。期限越长，宏观经济与借款人财务的不确定性敞口暴露周期越长。
    df['ESTIMATED_TENURE'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']

    # [2.3] GOODS_TO_CREDIT_RATIO（商品价值与信贷额度比 / 抵押覆盖率代理）
    # 含义：所购商品价格相对于借款额度的比例。比率大于1表示客户自付首付；比率小于1表示超额授信（信用贷套现风险更高）。
    df['GOODS_TO_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']

    # [2.4] DOWN_PAYMENT_PROXY（自筹首付金额代理）
    # 含义：商品价格减去信贷总额。差额为正越大，说明客户自掏腰包首付款越多，客户沉没成本高、违约意愿显著更低。
    df['DOWN_PAYMENT_PROXY'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']

    # [2.5] ANNUITY_TO_GOODS_RATIO（年还款额与商品价值比）
    # 含义：借款年金与商品真实市值的比重，综合反映借款人为该消费品所承担的实际现金流成本强度。
    df['ANNUITY_TO_GOODS_RATIO'] = df['AMT_ANNUITY'] / df['AMT_GOODS_PRICE']

    # -------------------------------------------------------------------------
    # 3. 外部征信评分深挖变量（01_eda.py 核心类别 2 & 6，全表最强预测特征）
    # -------------------------------------------------------------------------
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

    # [3.1] EXT_SOURCE_MEAN（外部评分综合均值）
    # 含义：三大外部征信评分的平均值。平滑单家机构评分的随机波动与孤立缺失，代表客户多方维度的基准信用水平。
    df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)

    # [3.2] EXT_SOURCE_STD（外部评分标准差 / 分歧度）
    # 含义：不同外部征信机构给分的离散程度。标准差大代表各机构评级产生冲突分歧，往往暗示借款人在特定机构存在隐形风险。
    df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)

    # [3.3] EXT_SOURCE_MIN（外部评分最低值 / 木桶短板效应）
    # 含义：外部评分中的最差表现。风控遵循木桶短板原则，只要在任一家机构出现极差评分，往往暴露了违约污点，具极强预警力。
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)

    # [3.4] EXT_SOURCE_MAX（外部评分最高值 / 信用天花板）
    # 含义：外部评分中的最高值，用于衡量借款人在最优场景下的潜在信用上限，有效区分中等资质与顶级优质客群。
    df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)

    # [3.5] EXT_SOURCE_PROD（外部评分联合乘积交互）
    # 含义：三大评分相乘，构造非线性高阶交互。放大多方评级同时优异（高乘积）与某方评级崩塌（乘积趋零）的极端分化。
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    # [3.6] EXT_SOURCE_WEIGHTED（基于 EDA IV 贡献权重的加权评分）
    # 含义：按 01_eda.py 中各特征 IV 大小（EXT3=0.336, EXT2=0.322, EXT1=0.147）经验加权（2:3:5），突出强特征贡献。
    df['EXT_SOURCE_WEIGHTED'] = (
        df['EXT_SOURCE_1'] * 2 + df['EXT_SOURCE_2'] * 3 + df['EXT_SOURCE_3'] * 5
    )

    # [3.7] EXT_SOURCE_NAN_CNT（外部征信评分缺失项个数）
    # 含义：统计三家外部评分中缺失的数量（0~3）。刻画征信“白户”程度，缺失评分越多代表正规信贷记录越匮乏，不确定性越高。
    df['EXT_SOURCE_NAN_CNT'] = df[ext_cols].isna().sum(axis=1)

    # [3.8] EXT_MEAN_TO_CREDIT_RATIO（综合信用与杠杆比率交叉）
    # 含义：综合信用评分均值与信贷收入比的比值。将“还款意愿/资信水平”与“还款压力/杠杆”深度交叉，比值极低者为高危违约户。
    df['EXT_MEAN_TO_CREDIT_RATIO'] = df['EXT_SOURCE_MEAN'] / df['CREDIT_TO_INCOME_RATIO']

    # -------------------------------------------------------------------------
    # 4. 时间变化、稳定性与生命周期交叉变量（01_eda.py 类别 8）
    # -------------------------------------------------------------------------

    # [4.1] EMPLOYED_TO_BIRTH_RATIO（工龄占年龄比例 / 就业稳定性比例）
    # 含义：工龄除以年龄（两者原始皆为负数，比值为正）。衡量借款人成年以来处于稳定在职状态的时间比重，比值大代表职业极度稳定。
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # [4.2] CAR_TO_BIRTH_RATIO（车龄占年龄比例）
    # 含义：名下车辆使用年限除以借款人实际年龄。若老旧车龄占借款人生命周期比重过大，反映其资产折旧严重且缺乏资产置换能力。
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (-df['DAYS_BIRTH'] / 365)

    # [4.3] ID_TO_BIRTH_RATIO（身份证更换天数占年龄比）
    # 含义：距最近一次更换身份证的天数占年龄比例。短期内近期频繁更换身份证常与身份信息变更、冒用欺诈或信用修复漂白相关。
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']

    # [4.4] REG_TO_BIRTH_RATIO（户籍登记天数占年龄比）
    # 含义：户口登记至今的天数占年龄比。长期未变更户口地址反映借款人定居环境极为稳定、社会关系紧密，违约跑路失联概率低。
    df['REG_TO_BIRTH_RATIO'] = df['DAYS_REGISTRATION'] / df['DAYS_BIRTH']

    # [4.5] PHONE_TO_BIRTH_RATIO（换手机天数占年龄比）
    # 含义：最近一次更换手机联系方式至今的天数占年龄比。手机号常年稳定不换是守信的核心特征，近期频繁换号常伴随失联逃债风险。
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']

    # [4.6] AGE_AT_EMPLOYMENT（初次参加工作估计年龄）
    # 含义：出生距今天数减去工龄天数换算为岁数。估算借款人进入职场的起始年龄，辅助识别过早辍学务工或极晚就业的异常客群。
    df['AGE_AT_EMPLOYMENT'] = (-df['DAYS_BIRTH'] + df['DAYS_EMPLOYED']) / 365

    # [4.7] PHONE_TO_EMPLOYED_RATIO（换手机时间与当前工龄之比）
    # 含义：联系方式变更距今天数与工龄之比。观察借款人通信稳定性与职业稳定性的相对匹配度。
    df['PHONE_TO_EMPLOYED_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED'].replace(0, np.nan)

    # -------------------------------------------------------------------------
    # 5. 缺失模式与数据采集指示变量（01_eda.py 缺失模式深度分析）
    # -------------------------------------------------------------------------

    # [5.1] HOUSE_INFO_MISSING_RATE（标准化房屋信息缺失率）
    # 含义：01_eda.py 发现房屋字段缺失率高达 50%~70% 且同组缺失 100% 同步，说明部分人群无正规房产登记。该缺失率直接刻画不动产资产缺失程度。
    avg_house_cols = [c for c in house_cols_list if c.endswith('AVG')]
    df['HOUSE_INFO_MISSING_RATE'] = df[avg_house_cols].isna().mean(axis=1)

    # [5.2] FLAG_OWN_CAR_MISSING（名下无车产记录二值标记）
    # 含义：OWN_CAR_AGE 缺失率约 66%，其缺失并非随机遗漏而是代表申请人名下无私家车。构建二值标记作为动产缺失判断。
    df['FLAG_OWN_CAR_MISSING'] = df['OWN_CAR_AGE'].isna().astype(int)

    # [5.3] TOTAL_NAN_COUNT（整行表申请资料总缺失项计数）
    # 含义：借款人在全表所有维度的缺失特征总数。反映客户信息完备度、配合度及在金融系统中的资料透明度，缺失多者风险通常更高。
    df['TOTAL_NAN_COUNT'] = df.isna().sum(axis=1)

    # -------------------------------------------------------------------------
    # 6. 身份真实性、反欺诈与社交圈风险变量（01_eda.py 类别 5）
    # -------------------------------------------------------------------------

    # [6.1] DOCUMENTS_PROVIDED_CNT（提交证明材料与文件总数量）
    # 含义：求和 FLAG_DOCUMENT_2~21。提交证明材料越齐全，借款人合规履约意愿越强、资质佐证越充分。
    doc_cols = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]
    df['DOCUMENTS_PROVIDED_CNT'] = df[doc_cols].sum(axis=1)

    # [6.2] ADDRESS_MISMATCH_MEAN（居住/工作/户籍地址多重错配均值）
    # 含义：统计 6 项户籍地、工作地、居住地不一致标记的均值。地址错配项越多，表明人户分离、异地务工或虚假包装工作嫌疑越重。
    addr_cols = [
        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION'
    ]
    df['ADDRESS_MISMATCH_MEAN'] = df[addr_cols].mean(axis=1)

    # [6.3] DEF_OBS_RATIO_30（近30天社交圈逾期人群占比）
    # 含义：在修复 30+ 极端离群值后，计算 30 天社交圈违约人数与可观察总人数之比。反映人际圈层的共性违约传染风险。
    df['DEF_OBS_RATIO_30'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)

    # [6.4] DEF_OBS_RATIO_60（近60天社交圈逾期人群占比）
    # 含义：中长期（60天）社交圈违约人数占比，衡量社交圈严重不良信用蔓延趋势。
    df['DEF_OBS_RATIO_60'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['OBS_60_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)

    # -------------------------------------------------------------------------
    # 7. 征信查询趋势与跨期稳定性改造变量（01_eda.py PSI 严重漂移特征改造）
    # -------------------------------------------------------------------------

    # [7.1] BUREAU_REQ_TOTAL（征信局历史查询总次数汇总）
    # 含义：求和各个周期的征信查询次数，综合反映借款人在人行/征信机构的整体被查询活跃度与融资饥渴程度。
    bureau_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    df['BUREAU_REQ_TOTAL'] = df[bureau_cols].sum(axis=1)

    # [7.2] FLAG_BUREAU_REQ（是否存在央行征信查询记录标记）
    # 含义：二值化区分征信白户与非白户，是否有过正规金融机构贷款查询历史。
    df['FLAG_BUREAU_REQ'] = (df['AMT_REQ_CREDIT_BUREAU_YEAR'] > 0).astype(int)

    # [7.3] BUREAU_REQ_SHORT_TO_LONG（短期征信查询与长期查询趋势比）
    # 含义：01_eda.py 发现单月和季度查询跨期 PSI > 0.31（严重漂移），按 EDA 结论改造成“短期激增趋势比”：
    # 近期查询（小时+天+周+月）占年查询的比率。短期查询激增往往表明借款人近期资金链紧张、正在密集借贷，是违约前兆。
    short_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON'
    ]
    df['BUREAU_REQ_SHORT_TO_LONG'] = df[short_cols].sum(axis=1) / (df['AMT_REQ_CREDIT_BUREAU_YEAR'] + 1)


# 对开发训练集与官方测试集执行特征衍生
add_derived_features(train_dev, house_cols)
add_derived_features(application_test, house_cols)

# %%
# =============================================================================
# 三、统一特征池漏斗初筛（Funnel Filtering on Unified Pool）
# =============================================================================
# 统一候选特征池（排除主键与标签，原始特征 + 衍生特征汇聚）
ignore_cols = ['SK_ID_CURR', 'TARGET']
candidate_features = [c for c in train_dev.columns if c not in ignore_cols]

# 1. 数据质量粗筛（同值率 > 0.95 或 缺失率 > 0.90）
missing_rates = train_dev[candidate_features].isna().mean()
primary_value_ratios = pd.Series({
    col: train_dev[col].value_counts(dropna=False).iloc[0] / len(train_dev)
    for col in candidate_features
})

high_pvr_cols = primary_value_ratios[primary_value_ratios > 0.95].index.tolist()
high_missing_cols = missing_rates[missing_rates > 0.90].index.tolist()
quality_drop_cols = sorted(list(set(high_pvr_cols + high_missing_cols)))

filtered_candidates = [c for c in candidate_features if c not in quality_drop_cols]

# 2. 批量最优分箱统一拟合（仅在开发集 train_dev 上 fit）
cat_cols = train_dev[filtered_candidates].select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in filtered_candidates if c not in cat_cols]

binning_fit_params = {
    col: {"monotonic_trend": "auto", "cat_cutoff": 0.05}
    for col in filtered_candidates
}

binning_process = BinningProcess(
    variable_names=filtered_candidates,
    categorical_variables=cat_cols,
    max_n_bins=8,
    min_bin_size=0.02,
    special_codes=[365243, 'XNA'],
    binning_fit_params=binning_fit_params,
    selection_criteria=None,
    n_jobs=-1
)
binning_process.fit(train_dev[filtered_candidates], train_dev['TARGET'])

iv_summary_df = binning_process.summary().rename(columns={'name': 'feature'})
iv_dict = dict(zip(iv_summary_df['feature'], iv_summary_df['iv']))

# 3. 跨客群稳定性指标 PSI 统一计算
train_bins = pd.DataFrame(
    binning_process.transform(train_dev[filtered_candidates], metric="bins"),
    columns=filtered_candidates,
    index=train_dev.index
)
test_bins = pd.DataFrame(
    binning_process.transform(application_test[filtered_candidates], metric="bins"),
    columns=filtered_candidates,
    index=application_test.index
)
psi_series = toad.metrics.PSI(test_bins[filtered_candidates], train_bins[filtered_candidates])
psi_dict = psi_series.to_dict()

del train_bins, test_bins, binning_process
gc.collect()

# 汇总全量特征体检表
feature_stats = pd.DataFrame({
    'Row': filtered_candidates,
    'iv': [iv_dict.get(c, np.nan) for c in filtered_candidates],
    'psi': [psi_dict.get(c, np.nan) for c in filtered_candidates],
    'missing_rate': [missing_rates.get(c, np.nan) for c in filtered_candidates],
    'primary_value_ratio': [primary_value_ratios.get(c, np.nan) for c in filtered_candidates],
    'dtypes': [train_dev[c].dtype.name for c in filtered_candidates],
    'is_derived': [c not in cols_eda['Row'].values for c in filtered_candidates]
})

# 4. 指标阈值统一初筛：保留 IV >= 0.02 且 PSI <= 0.25 的特征
low_iv_cols = feature_stats[feature_stats['iv'] < 0.02]['Row'].tolist()
high_psi_cols = feature_stats[feature_stats['psi'] > 0.25]['Row'].tolist()

metric_passed_stats = feature_stats[
    (feature_stats['iv'] >= 0.02) & (feature_stats['psi'] <= 0.25)
].sort_values(by='iv', ascending=False).reset_index(drop=True)

del low_iv_cols, high_psi_cols, quality_drop_cols
gc.collect()

# %%
# =============================================================================
# 四、统一多变量共线性与冗余筛选（Multivariate Collinearity & Redundancy）
# =============================================================================
passed_cols = metric_passed_stats['Row'].tolist()

# 1. 房屋标准化特征同源后缀去重与跨维度面积共线消除
passed_house_cols = [c for c in passed_cols if c in house_cols and train_dev[c].dtype.name == 'float64']
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
cross_dim_keep = cross_dim_iv.iloc[0]['Row']
cross_dim_drops = cross_dim_iv.iloc[1:]['Row'].tolist()
house_redundant_drops.extend(cross_dim_drops)

del house_df, house_keep, house_suffix_drops

cols_after_house = [c for c in passed_cols if c not in house_redundant_drops]

# 2. 全局贪心共线性消除算法 (Greedy Collinearity Elimination)
# 将初筛保留的数值特征按 IV 降序排列，计算 Spearman 秩相关系数
num_passed = [c for c in cols_after_house if train_dev[c].dtype.name != 'object']
cat_passed = [c for c in cols_after_house if train_dev[c].dtype.name == 'object']

num_stats = metric_passed_stats[metric_passed_stats['Row'].isin(num_passed)].sort_values(by='iv', ascending=False)
sorted_num_cols = num_stats['Row'].tolist()

spearman_corr = train_dev[sorted_num_cols].corr(method='spearman')

CORR_THRESHOLD = 0.80
kept_num_cols = []
dropped_corr_records = []

for col in sorted_num_cols:
    is_collinear = False
    for kept in kept_num_cols:
        r = spearman_corr.loc[col, kept]
        if abs(r) > CORR_THRESHOLD:
            is_collinear = True
            dropped_corr_records.append({
                'dropped_feature': col,
                'collinear_with': kept,
                'correlation': r,
                'dropped_iv': iv_dict[col],
                'kept_iv': iv_dict[kept]
            })
            break
    if not is_collinear:
        kept_num_cols.append(col)

# 合并数值与类别型特征，构建最终入模特征清单
final_features = kept_num_cols + cat_passed
final_summary = metric_passed_stats[metric_passed_stats['Row'].isin(final_features)].sort_values(by='iv', ascending=False).reset_index(drop=True)

del spearman_corr, dropped_corr_records, house_redundant_drops, num_passed, cat_passed, sorted_num_cols
gc.collect()

# %%
# =============================================================================
# 五、输出特征工程加工结果与体检账本
# =============================================================================
train_fe = train_dev[final_features + ['TARGET']].copy()
test_fe = application_test[final_features].copy()

# 落盘加工后宽表与入模特征体检账本
train_fe.to_csv(PROCESSED_DATA_PATH / 'train_fe.csv', index=False)
test_fe.to_csv(PROCESSED_DATA_PATH / 'test_fe.csv', index=False)
final_summary.to_csv(EDA_OUTPUT_PATH / 'model_features_summary.csv', index=False)

# 清理内存
del train_dev, application_test, train_fe, test_fe, cols_eda, house_cols, final_summary
gc.collect()
