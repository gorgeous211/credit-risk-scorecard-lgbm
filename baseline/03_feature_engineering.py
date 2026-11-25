# %%
# =============================================================================
# 全量风控特征工程与特征宽表构建
# (Feature Engineering Pipeline & Feature Table Construction)
# =============================================================================
# 模块功能：
# 完成原始特征的异常值清洗、业务特征衍生及行业分组统计，
# 输出全量候选特征宽表，供下游评分卡（LR）与提升树（LightGBM）统一读取。
#
# 数据隔离要求：
# 1. 仅使用 00 脚本切分的训练集（Train，80%）计算聚合统计量（如职业/行业收入均值）；
#    验证集（Valid，20%）与测试集（Test）仅执行映射（merge），避免数据穿越。
# 2. 测试集无 TARGET 标签，仅同步应用相同变换，用于后续 PSI 稳定性评估。
# 3. 生成的宽表保存至 data/processed/，各模型根据自身需求进一步筛选特征。
# =============================================================================

# %%
import gc
from pathlib import Path

import numpy as np
import pandas as pd

# 文件路径配置
DATA_PATH = Path.cwd() / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
INTERIM_DATA_PATH = DATA_PATH / "interim"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

OUTPUTS_PATH = Path.cwd() / "outputs"
EDA_OUTPUT_PATH = OUTPUTS_PATH / "eda"
EDA_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("正在读取原始与切分数据集...")
# 正在读取原始与切分数据集...
# 1. 训练集（唯一规则拟合来源，80%）
train = pd.read_csv(INTERIM_DATA_PATH / 'application_train_split.csv')

# 2. 验证集（20%）
valid = pd.read_csv(INTERIM_DATA_PATH / 'application_valid.csv')

# 3. 测试集（未知客群分布参照）
test = pd.read_csv(RAW_DATA_PATH / 'application_test.csv')

# 读取原始特征画像与说明表提取房屋相关字段与原始特征字段列表
raw_features_summary = pd.read_csv(EDA_OUTPUT_PATH / 'raw_features_summary.csv')
raw_feature_cols = [c for c in raw_features_summary['Row'].tolist() if c not in ['SK_ID_CURR', 'TARGET']]
house_cols = raw_features_summary.loc[
    raw_features_summary['Description'].str.startswith('Normalized information'), 'Row'
].tolist()

print(f"数据读取完成: 训练集 {train.shape}, 验证集 {valid.shape}, 测试集 {test.shape}")
# 数据读取完成: 训练集 (246008, 122), 验证集 (61503, 122), 测试集 (48744, 121)

# %%
# =============================================================================
# 一、特殊异常值替换与基础清洗
# =============================================================================
def clean_anomalies(df):
    """
    基础清洗与异常值修复：
    1. DAYS_EMPLOYED == 365243：该值约为 1000 年（代表退休或无正规就业），
       转为 NaN，并新增 FLAG_DAYS_EMPLOYED_ANOMALOUS 标识。
    2. CODE_GENDER：将 4 条未记录性别 'XNA' 替换为主导类别 'F'。
    3. 社交圈观察与逾期人数（OBS/DEF_30/60_CNT_SOCIAL_CIRCLE）：将异常大值 (>30) 置为 NaN，
       保持分子分母统计口径一致。
    """
    # [0.1] 特殊在职天数异常标记（1=原值为 365243，0=正常在职天数）：
    # 业务含义：Home Credit 中 365243 对应退休、无业或非正规就业客群；将其显式剥离为二值标识，
    # 既避免极端异常值扭曲后续数值/比率计算，又完整保留“非受薪/退休人群”的差异化违约风险信号。
    df['FLAG_DAYS_EMPLOYED_ANOMALOUS'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', 'F')
    anom_mask_30 = df['OBS_30_CNT_SOCIAL_CIRCLE'] > 30
    df.loc[anom_mask_30, ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE']] = np.nan
    anom_mask_60 = df['OBS_60_CNT_SOCIAL_CIRCLE'] > 30
    df.loc[anom_mask_60, ['OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']] = np.nan


# %%
# =============================================================================
# 二、全量业务特征衍生（覆盖 8 大风控维度）
# =============================================================================
def add_all_derived_features(df, house_cols_list, raw_feature_cols_list):
    """构建业务衍生特征（覆盖负债比率、还款期限、征信交互、时间稳定性等维度）。"""
    # -------------------------------------------------------------------------
    # 维度 1: 还款能力与负债压力 (Repayment Ability & Debt Burden)
    # 逻辑：将绝对收入与贷款额转化为相对比率（杠杆倍数、DTI 代理、人均可支配收入等），
    # 消除绝对金额受收入基数与地域差异的影响，衡量实际还款压力。
    # -------------------------------------------------------------------------
    # [1.1] 信贷收入比 (杠杆倍数)
    # 业务含义：衡量申请贷款总额相对于年总收入的放大倍数。比值越高表示举债杠杆越大、偿债周期越长，过度授信与违约风险越高。
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    # [1.2] 年金收入比 (DTI 代理: 月供占年总收入比率)
    # 业务含义：衡量单期还款额对年总收入的占用比例（类似 DTI 债务收入比）。比值越高说明刚性月供对日常现金流挤压越重，收入稍有波动即易逾期。
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # [1.3] 月均可用净现金流代理 (月均收入 - 单期月还款额)
    # 业务含义：估算扣除当期还款额后的月度剩余可支配资金。反映借款人还贷后的实际生活安全垫，结余越少或为负说明抗风险缓冲能力越弱。
    df['FREE_INCOME_PROXY'] = (df['AMT_INCOME_TOTAL'] / 12) - df['AMT_ANNUITY']
    # [1.4] 家庭人均年收入
    # 业务含义：按家庭总人数摊薄后的年收入，校正只看个人名义收入而忽视家庭赡养负担的偏差。人均越高通常代表家庭综合偿债余力越充足。
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    # [1.5] 抚养子女摊薄人均年收入
    # 业务含义：衡量承担子女抚养责任后的相对收入水平（分母 +1 防零除）。子女数越多，教育及刚性生活开销越大，对中低收入客群的实际还款能力削弱越明显。
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])

    # -------------------------------------------------------------------------
    # 维度 2: 贷款条件与抵押覆盖 (Loan Conditions & Collateral Coverage)
    # 逻辑：评估信贷额度与标的商品价值的匹配程度及客户自付成本；
    # 差额为负代表存在超额授信/加融（如附加手续费与保险），抵押覆盖越低违约风险通常越高。
    # -------------------------------------------------------------------------
    # [2.1] 还款费率 (年金信贷比 / 期限倒数代理)
    # 业务含义：单期月供占贷款总额的比例，综合反映贷款利率水平与期限长短（期数越短、利率越高则该值越大，常呈非单调风险结构）。
    df['ANNUITY_TO_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    # [2.2] 预估还款期数 (贷款期限代理)
    # 业务含义：贷款总额 / 单期月供，近似反映含息费名义还款期数。期数越长暴露在未来收入波动下的风险敞口越长，且高风险客群更倾向拉长分期压低月供。
    df['ESTIMATED_TENURE'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    # [2.3] 商品价值与信贷额度比 (抵押覆盖率代理: >1 为自付首付, <1 为超融)
    # 业务含义：衡量消费标的实际价值对放款额度的覆盖程度。>1 说明客户自付首付、违约沉没成本高；<1 代表超额授信或搭售保险/手续费加融，无抵押敞口大、违约率显著升高。
    df['GOODS_TO_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    # [2.4] 首付与超融差额代理 (商品标价 - 信贷总额: >0 为自付首付, <0 为超额授信)
    # 业务含义：以绝对金额刻画客户首付实力或超融规模。正差额越大表示自有资金充裕、还款意愿强；负差额代表客户流动性紧缺或高费率加融特征。
    df['DOWN_PAYMENT_PROXY'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    # [2.5] 单期月供与商品价值比
    # 业务含义：衡量单期还款负担占所购商品实际标价的比例，用于识别因高息、短期限或超额加融导致的过度月供压力。
    df['ANNUITY_TO_GOODS_RATIO'] = df['AMT_ANNUITY'] / df['AMT_GOODS_PRICE']

    # -------------------------------------------------------------------------
    # 维度 3: 外部征信评分多维交互 (External Credit Scores Multi-Interaction)
    # 逻辑：外部三方征信评级（EXT_SOURCE_1/2/3）具有较强判别力。
    # 构造均值、标准差、极值、经验加权及交互项，综合衡量借款人信用质量与机构间评级分歧。
    # -------------------------------------------------------------------------
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    # [3.1] 外部评分均值 (平滑单家机构波动)
    # 业务含义：综合三家外部征信机构的平均信用表现，降低单一机构覆盖盲区或短期波动噪声，均值越高代表综合信用资质越优、违约风险越低。
    df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    # [3.2] 外部评分标准差 (反映机构间评级分歧)
    # 业务含义：衡量不同外部征信源对同一客户评级的一致性。标准差越大表明客户在不同数据维度（如央行信贷 vs 行为征信）表现撕裂，隐含潜在渠道风险或信息不对称。
    df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
    # [3.3] 外部评分最低值 (衡量最差信用表现)
    # 业务含义：遵循风控“木桶短板效应”，捕捉借款人在各外部机构中的最差信用记录；只要任一机构给出极低分，往往预示实质性信用瑕疵。
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
    # [3.4] 外部评分最高值
    # 业务含义：刻画借款人在最优数据维度下达到的信用上限，可辅助识别因个别机构数据滞后或偏低而被误杀的潜在优质客户。
    df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
    # [3.5] 联合乘积交互 (三家机构均有评分时生效，缺失时为 NaN)
    # 业务含义：对低信用分具备强非线性惩罚的跨机构联合指标——只有三家机构均给出高分时乘积才显著偏高，用于强筛选跨渠道一致认可的优质客户。
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    # [3.6] 经验加权评分 (按 IV 经验 2:3:5 权衡，动态归一化非空项权重)
    # 业务含义：根据三家外部数据源自身的风险区分度（IV 经验权重 2:3:5）计算加权综合信用分，突出最具判别力的 EXT_SOURCE_3，并避免局部缺失导致分数偏低。
    ext_weights = pd.Series({'EXT_SOURCE_1': 2.0, 'EXT_SOURCE_2': 3.0, 'EXT_SOURCE_3': 5.0})
    ext_num = (df[ext_cols] * ext_weights).sum(axis=1, min_count=1)
    ext_den = (df[ext_cols].notna() * ext_weights).sum(axis=1).replace(0, np.nan)
    df['EXT_SOURCE_WEIGHTED'] = ext_num / ext_den
    # [3.7] 外部评分缺失项数量 (衡量征信信息完备度)
    # 业务含义：统计三方征信未覆盖的机构数量，衡量“信用白户”程度。缺失越少说明信用足迹越完整；全缺失意味着缺乏外部履约背书，信用不确定性高。
    df['EXT_SOURCE_NAN_CNT'] = df[ext_cols].isna().sum(axis=1)
    # [3.8] 外部评分均值与杠杆比率交叉
    # 业务含义：将“信用资质（分子）”与“借贷杠杆（分母）”联合，区分“高信用低杠杆”核心安全客群与“低信用高杠杆”高危客群。
    df['EXT_MEAN_TO_CREDIT_RATIO'] = df['EXT_SOURCE_MEAN'] / df['CREDIT_TO_INCOME_RATIO']
    # [3.9] 外部评分均值与年龄交叉
    # 业务含义：衡量单位年龄所积累的外部信用评分水平，用于校正年龄对征信历史沉淀长度的自然影响（同评分下年轻客群与高龄客群风险含义不同）。
    df['EXT_MEAN_TO_BIRTH'] = df['EXT_SOURCE_MEAN'] / (-df['DAYS_BIRTH'] / 365)

    # -------------------------------------------------------------------------
    # 维度 4: 时间稳定性与生命周期交叉 (Temporal Stability & Lifecycle Ratios)
    # 逻辑：将工龄、车龄、户籍登记、换证和换手机等负数天数除以年龄，
    # 消除绝对天数随年龄增长的自然关联，转化为相对生命周期占比。
    # -------------------------------------------------------------------------
    # [4.1] 就业稳定性 (工龄占年龄比)
    # 业务含义：消除年龄基数影响后的职业持续度。占比越高表明在当前单位受雇时间长、工作与收入来源高度稳定，违约概率越低。
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    # [4.2] 车辆使用年限占年龄比
    # 业务含义：在有车客群中反映车辆老化程度相对个人生命周期的比例。车龄过老（比值偏高）常对应资产折旧严重、消费更新能力偏弱或资产质量一般。
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (-df['DAYS_BIRTH'] / 365)
    # [4.3] 身份证持有时间占年龄比
    # 业务含义：反映最近一次身份核发/换证至今时长占年龄的比例。短期内刚换证有时伴随身份信息变动或集中申贷行为，长期未换证通常体现生活状态稳定。
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    # [4.4] 户口登记时间占年龄比
    # 业务含义：衡量当前户籍/居住登记持续时长占生命长度的比例。比值越高表示长期定居、本地根基稳固、失联风险低；近期刚变更登记者流动性较高。
    df['REG_TO_BIRTH_RATIO'] = df['DAYS_REGISTRATION'] / df['DAYS_BIRTH']
    # [4.5] 手机号更换时间占年龄比
    # 业务含义：反映当前手机号使用稳固度。长期不换号（高比值）代表社交与联络渠道高度稳定、贷后可触达性强；近期刚换号往往与躲避催收或生活动荡相关。
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    # [4.6] 换手机间隔与工龄比
    # 业务含义：对比联系方式稳定性与职业稳定性。若在职时间很长但手机号近期刚更换，或换号周期明显异于就业周期，可能提示联系信息存疑或规避催收风险。
    df['PHONE_TO_EMPLOYED_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED'].replace(0, np.nan)
    # [4.7] 初入职预估年龄
    # 业务含义：估算借款人进入当前工作的年龄（当前年龄 - 工龄）。壮年正常起步并长期留任通常更稳健；高龄近期才新入职可能暗示此前职业中断或收入波动。
    df['AGE_AT_EMPLOYMENT'] = (-df['DAYS_BIRTH'] + df['DAYS_EMPLOYED']) / 365
    # [4.8] 户口与身份证更换时间比
    # 业务含义：衡量户籍登记变动周期与证件换发周期的同步性，用于捕捉短期内同时集中变更居住登记与身份信息的异常迁徙/申贷信号。
    df['REG_TO_ID_RATIO'] = df['DAYS_REGISTRATION'] / df['DAYS_ID_PUBLISH'].replace(0, np.nan)

    # -------------------------------------------------------------------------
    # 维度 5: 缺失模式与资产完备度 (Missingness Patterns & Asset Completeness)
    # 逻辑：房屋和车辆相关字段属于条件填报（有资产才填）。
    # 提取房屋信息缺失率与车辆缺失标记，并统计原始填报字段的缺失总数。
    # -------------------------------------------------------------------------
    # [5.1] 标准化房屋信息同源缺失率
    # 业务含义：衡量居住房屋物理属性（面积、楼层、电梯等）的缺失程度。自有商品房或规范小区住户通常记录完整，高缺失率多对应租住、非标住宅或资产透明度低客群。
    avg_house_cols = [c for c in house_cols_list if c.endswith('AVG') and c in df.columns]
    df['HOUSE_INFO_MISSING_RATE'] = df[avg_house_cols].isna().mean(axis=1)
    # [5.2] 无私家车二值标记
    # 业务含义：车龄字段缺失即标识未持有自有车辆或未申报车辆资产（1=无车/缺失），直接区分有车与无车客群的动产缓冲垫差异。
    df['FLAG_OWN_CAR_MISSING'] = df['OWN_CAR_AGE'].isna().astype(int)
    # [5.3] 原始填报资料缺失总项数 (基于原始特征统计，度量申请资料完备度)
    # 业务含义：度量单笔进件原始信息的完整度与可核验性。缺失总数越多，说明客户资料盲区越大、可验证维度越少，逆向选择与信息不对称风险越高。
    matched_raw_cols = [c for c in raw_feature_cols_list if c in df.columns]
    df['TOTAL_NAN_COUNT'] = df[matched_raw_cols].isna().sum(axis=1)

    # -------------------------------------------------------------------------
    # 维度 6: 身份真实性与社交圈风险 (Identity Verification & Social Risk)
    # 逻辑：汇总提交的身份证明材料数量，统计居住地、户籍地与工作地的错配情况，
    # 并计算社交圈逾期人数占总观察人数的比例。
    # -------------------------------------------------------------------------
    # [6.1] 提交证明材料与资质文件总数量
    # 业务含义：反映客户进件时递交的身份与资质证明文件丰富度，衡量申请流程配合度与资料完备水平（如仅提交单份核心证件 vs 补充多项辅助证明）。
    doc_cols = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22) if f"FLAG_DOCUMENT_{i}" in df.columns]
    df['DOCUMENTS_PROVIDED_CNT'] = df[doc_cols].sum(axis=1)
    # [6.2] 地址错配均值 (户籍/工作/居住地一致性)
    # 业务含义：衡量户籍地、现居地与工作地在城市及省区维度的分离程度。错配度越高说明人户分离、跨城流动或信息填报不一致，贷后失联与违约风险越高。
    addr_cols = [
        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION'
    ]
    matched_addr_cols = [c for c in addr_cols if c in df.columns]
    df['ADDRESS_MISMATCH_MEAN'] = df[matched_addr_cols].mean(axis=1)
    # [6.3] 逾期 30 天社交圈违约人员占比 (逾期 30 天以上人数 / 观察总人数)
    # 业务含义：基于“物以类聚”的社交信用传导效应，衡量社交关系网中发生 30 天以上逾期的人口比例；比例越高提示所处圈层信用环境越差、共债或欺诈风险越高。
    df['DEF_OBS_RATIO_30'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)
    # [6.4] 逾期 60 天社交圈违约人员占比 (逾期 60 天以上人数 / 观察总人数)
    # 业务含义：社交圈中发生 60 天以上深度逾期（近坏账口径）的占比。相比 30 天口径更能过滤偶发忘还噪声，精准反映人际网络中的实质性恶意违约或无力偿债风险。
    df['DEF_OBS_RATIO_60'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['OBS_60_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)

    # -------------------------------------------------------------------------
    # 维度 7: 征信查询趋势与近期借贷 (Credit Inquiry Trends)
    # 逻辑：统计历史征信查询总量，并构造近 1 个月查询量与长期查询量的比值，
    # 观察近期是否存在借贷需求激增的情况。
    # -------------------------------------------------------------------------
    bureau_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    matched_bureau = [c for c in bureau_cols if c in df.columns]
    # [7.1] 征信局历史查询总次数 (保留全缺失语义)
    # 业务含义：反映过去 1 年内客户被外部金融机构查询征信的累计频次。总次数偏高代表长期多头借贷频密、资金链持续紧张，是强违约前兆。
    df['BUREAU_REQ_TOTAL'] = df[matched_bureau].sum(axis=1, min_count=1)
    # [7.2] 征信查询记录标记 (有历史查询为 1，无查询或缺失为 0)
    # 业务含义：二值区分“近期存在活跃外贷申请行为的客群”与“无查询记录/静默客群”，捕捉行为活跃度差异。
    df['FLAG_BUREAU_REQ'] = (df['BUREAU_REQ_TOTAL'] > 0).astype(int)
    # [7.3] 短期借贷激增比 (近 1 个月查询量 / 远期查询量加 1 平滑)
    # 业务含义：衡量近 1 个月内借贷申请热度相对于过去 1 年常规水平的突增程度。比值越高代表短时间内集中向多家机构“扫贷”求资，高度预警资金链断裂、拆东墙补西墙或集中骗贷。
    short_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON'
    ]
    matched_short = [c for c in short_cols if c in df.columns]
    df['BUREAU_REQ_SHORT_TO_LONG'] = (
        df[matched_short].sum(axis=1, min_count=1) / (df['AMT_REQ_CREDIT_BUREAU_YEAR'] + 1)
    )


print("正在执行全量数据集清洗与 7 大通用维度特征衍生...")
# 正在执行全量数据集清洗与 7 大通用维度特征衍生...
for dataset_name, df_obj in [("训练集", train), ("验证集", valid), ("测试集", test)]:
    clean_anomalies(df_obj)
    add_all_derived_features(df_obj, house_cols, raw_feature_cols)
    print(f"✓ {dataset_name} 特征衍生完成，当前特征数: {df_obj.shape[1]}")
# ✓ 训练集 特征衍生完成，当前特征数: 160
# ✓ 验证集 特征衍生完成，当前特征数: 160
# ✓ 测试集 特征衍生完成，当前特征数: 159

# %%
# =============================================================================
# 三、维度 8: 行业与职业分组基准统计量 (Industry & Occupation Benchmarks)
# =============================================================================
# 逻辑：不同行业和职业的收入基准存在客观差异。
# 计算同职业/同行业的收入均值，并构建个人收入相对群体均值的比率，
# 衡量借款人在其所属群体内的相对收入水平。
#
# 防数据穿越说明：
# 分组收入均值仅在训练集上计算，验证集和测试集通过 left join 匹配，避免统计量泄露。
# =============================================================================
print("正在拟合行业与职业分组基准统计量 (严格仅限训练集)...")
# 正在拟合行业与职业分组基准统计量 (严格仅限训练集)...
# [8.1] OCCUPATION_TYPE_INCOME_MEAN: 同一职业群体的平均收入基准
# 业务含义：刻画所属具体职业岗位的社会平均薪酬水位（如高管/IT vs 普通劳工），为评估个体收入高低提供客观岗位参照系。
occupation_stats = train.groupby(
    'OCCUPATION_TYPE', observed=True
)['AMT_INCOME_TOTAL'].mean().rename('OCCUPATION_TYPE_INCOME_MEAN').reset_index()
train = train.merge(occupation_stats, on='OCCUPATION_TYPE', how='left')
valid = valid.merge(occupation_stats, on='OCCUPATION_TYPE', how='left')
test = test.merge(occupation_stats, on='OCCUPATION_TYPE', how='left')

# [8.2] INCOME_TO_OCCUPATION_TYPE_RATIO: 个人收入相对职业群体平均收入的比率
# 业务含义：衡量借款人在同职业同行中的相对收入竞争力。>1 说明岗位收入领先（职级或能力更强、偿债能力更优）；过低反映同行内偏弱，极端高值则可辅助识别虚报收入。
train['INCOME_TO_OCCUPATION_TYPE_RATIO'] = (
    train['AMT_INCOME_TOTAL'] / train['OCCUPATION_TYPE_INCOME_MEAN']
)
valid['INCOME_TO_OCCUPATION_TYPE_RATIO'] = (
    valid['AMT_INCOME_TOTAL'] / valid['OCCUPATION_TYPE_INCOME_MEAN']
)
test['INCOME_TO_OCCUPATION_TYPE_RATIO'] = (
    test['AMT_INCOME_TOTAL'] / test['OCCUPATION_TYPE_INCOME_MEAN']
)
print("✓ 完成 OCCUPATION_TYPE 收入基准均值与相对比率构造")
# ✓ 完成 OCCUPATION_TYPE 收入基准均值与相对比率构造

# [8.3] ORGANIZATION_TYPE_INCOME_MEAN: 同一行业群体的平均收入基准
# 业务含义：刻画所属单位/行业类型（如金融、政府机关、建筑、贸易等）的整体薪资基准，反映行业景气度与客群整体收入层级。
organization_stats = train.groupby(
    'ORGANIZATION_TYPE', observed=True
)['AMT_INCOME_TOTAL'].mean().rename('ORGANIZATION_TYPE_INCOME_MEAN').reset_index()
train = train.merge(organization_stats, on='ORGANIZATION_TYPE', how='left')
valid = valid.merge(organization_stats, on='ORGANIZATION_TYPE', how='left')
test = test.merge(organization_stats, on='ORGANIZATION_TYPE', how='left')

# [8.4] INCOME_TO_ORGANIZATION_TYPE_RATIO: 个人收入相对行业群体平均收入的比率
# 业务含义：剥离行业间固有薪资差距，衡量个人在所属行业内部的相对收入地位与岗位稳定性，识别各行业内部的头部优质客群或边缘低收入群体。
train['INCOME_TO_ORGANIZATION_TYPE_RATIO'] = (
    train['AMT_INCOME_TOTAL'] / train['ORGANIZATION_TYPE_INCOME_MEAN']
)
valid['INCOME_TO_ORGANIZATION_TYPE_RATIO'] = (
    valid['AMT_INCOME_TOTAL'] / valid['ORGANIZATION_TYPE_INCOME_MEAN']
)
test['INCOME_TO_ORGANIZATION_TYPE_RATIO'] = (
    test['AMT_INCOME_TOTAL'] / test['ORGANIZATION_TYPE_INCOME_MEAN']
)
print("✓ 完成 ORGANIZATION_TYPE 收入基准均值与相对比率构造")
# ✓ 完成 ORGANIZATION_TYPE 收入基准均值与相对比率构造

del occupation_stats, organization_stats, house_cols, raw_feature_cols

# %%
# =============================================================================
# 四、输出全量特征宽表与数据字典账本
# =============================================================================
# 统一候选特征池统计
ignore_cols = ['SK_ID_CURR', 'TARGET']
feature_cols = [c for c in train.columns if c not in ignore_cols]
derived_cols = [c for c in feature_cols if c not in raw_features_summary['Row'].values]

print(f"\n全量候选特征池构建完毕！总特征数: {len(feature_cols)} 个 (含 {len(derived_cols)} 个新增衍生特征)")
# 全量候选特征池构建完毕！总特征数: 162 个 (含 42 个新增衍生特征)

# 汇总全量特征字典体检表 (仅基于训练集自身统计)
missing_rates = train[feature_cols].isna().mean()
primary_value_ratios = pd.Series({
    col: train[col].value_counts(dropna=False).iloc[0] / len(train)
    for col in feature_cols
})

full_features_summary = pd.DataFrame({
    'Row': feature_cols,
    'dtypes': [train[c].dtype.name for c in feature_cols],
    'missing_rate': [missing_rates[c] for c in feature_cols],
    'primary_value_ratio': [primary_value_ratios[c] for c in feature_cols],
    'is_derived': [c not in raw_features_summary['Row'].values for c in feature_cols]
}).sort_values(by=['is_derived', 'missing_rate'], ascending=[False, True]).reset_index(drop=True)

# 落盘全量特征宽表至 data/processed/
print("正在将全量特征宽表落盘至 data/processed/ ...")
# 正在将全量特征宽表落盘至 data/processed/ ...
train.to_csv(PROCESSED_DATA_PATH / 'train_features_full.csv', index=False)
valid.to_csv(PROCESSED_DATA_PATH / 'valid_features_full.csv', index=False)
test.to_csv(PROCESSED_DATA_PATH / 'test_features_full.csv', index=False)

# 保存体检报告
full_features_summary.to_csv(EDA_OUTPUT_PATH / 'full_features_summary.csv', index=False)

print("\n" + "=" * 70)
print("✓ [统一特征工程构建完成 - 审计清单]")
print(f"  1. 训练集宽表 : {PROCESSED_DATA_PATH / 'train_features_full.csv'} (样本数: {len(train):,}, 字段数: {train.shape[1]})")
print(f"  2. 验证集宽表 : {PROCESSED_DATA_PATH / 'valid_features_full.csv'} (样本数: {len(valid):,}, 字段数: {valid.shape[1]})")
print(f"  3. 测试集宽表 : {PROCESSED_DATA_PATH / 'test_features_full.csv'} (样本数: {len(test):,}, 字段数: {test.shape[1]})")
print(f"  4. 全量特征字典账本: {EDA_OUTPUT_PATH / 'full_features_summary.csv'}")
print("=" * 70)
# ======================================================================
# ✓ [统一特征工程构建完成 - 审计清单]
#   1. 训练集宽表 : D:\0.My-Projects\home-credit-default-risk\data\processed\train_features_full.csv (样本数: 246,008, 字段数: 164)
#   2. 验证集宽表 : D:\0.My-Projects\home-credit-default-risk\data\processed\valid_features_full.csv (样本数: 61,503, 字段数: 164)
#   3. 测试集宽表 : D:\0.My-Projects\home-credit-default-risk\data\processed\test_features_full.csv (样本数: 48,744, 字段数: 163)
#   4. 全量特征字典账本: D:\0.My-Projects\home-credit-default-risk\outputs\eda\full_features_summary.csv
# ======================================================================

# 清理内存
del train, valid, test, raw_features_summary, full_features_summary
del missing_rates, primary_value_ratios, feature_cols, ignore_cols, derived_cols
gc.collect()
