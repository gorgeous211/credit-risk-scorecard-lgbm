# %%
# =============================================================================
# 全量风控特征工程与特征宽表构建
# (Unified Feature Engineering Pipeline & Feature Table Construction)
# =============================================================================
# 核心架构定位：
# 本脚本作为全流程的“统一特征工程流水线”。
# 统一完成全量原始特征的异常值清洗、8大风控维度的业务特征衍生与行业分组统计，
# 并输出供下游“传统统计评分卡（LR）”与“机器学习提升树（LightGBM）”统一读取
# 的全量候选特征宽表，彻底消除重复代码与特征口径不一致的隐患。
#
# 数据隔离与防数据穿越原则（严格遵守 00_stratified_split.py 规范）：
# 1. 严格使用 00 脚本切分的训练集（Train, 80%）拟合所有带统计量的规则
#    （如行业/职业分组收入均值统计量），验证集（Valid, 20%）与测试集（Test）
#    仅执行纯映射（transform / merge），彻底杜绝数据穿越；
# 2. 测试集 application_test.csv 无 TARGET 标签，仅应用相同特征变换供最终
#    跨客群总分/概率 PSI 稳定性评估使用，严禁参与任何监督拟合或特征选择；
# 3. 本脚本输出全量特征宽表至 data/processed/，下游模型各自按需执行专属筛选，
#    实现“统一特征候选池 + 模型专属分流适配”。
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

# %%
# =============================================================================
# 一、特殊异常值替换与基础清洗（全表无参数变换）
# =============================================================================
def clean_anomalies(df):
    """
    基础清洗与异常值修复函数（前置清洗与异常衍生）：
    1. DAYS_EMPLOYED == 365243：该值约为 1000 年（业务编码标记为退休或无正规就业），
       直接作为连续数值会导致树模型产生严重失真分裂。将其重置为 NaN，同时提取独立
       二值风险特征 FLAG_DAYS_EMPLOYED_ANOMALOUS 显式保留该客群属性；
    2. CODE_GENDER：将极少数未知性别 'XNA' 替换为样本内主导众数 'F'，避免低频脏类别干扰编码；
    3. 社交圈观察与逾期人数（OBS/DEF_30/60_CNT_SOCIAL_CIRCLE）：极端离群值 (>30) 联动置为 NaN，
       确保圈层观察基底（分母）与逾期违约人数（分子）在后续构建比率时的口径严谨一致。
    """
    df['FLAG_DAYS_EMPLOYED_ANOMALOUS'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', 'F')
    anom_mask_30 = df['OBS_30_CNT_SOCIAL_CIRCLE'] > 30
    df.loc[anom_mask_30, ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE']] = np.nan
    anom_mask_60 = df['OBS_60_CNT_SOCIAL_CIRCLE'] > 30
    df.loc[anom_mask_60, ['OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']] = np.nan


# %%
# =============================================================================
# 二、全量业务特征衍生（覆盖 8 大风控维度体系）
# =============================================================================
def add_all_derived_features(df, house_cols_list, raw_feature_cols_list):
    """
    构建全量业务衍生特征池：
    基于信贷风控机理（Domain Knowledge）自发构造，摆脱对绝对数值与宏观周期的依赖。
    """
    # -------------------------------------------------------------------------
    # 维度 1: 还款能力与负债压力变量 (Repayment Ability & Debt Burden)
    # -------------------------------------------------------------------------
    # 【风控维度内涵与设计逻辑】：
    # 1. 业务含义：反映借款人“是否有足够的现金流来按期履约”，是信贷风险最基础的衡量支柱。
    # 2. 设计目的：原始的收入（AMT_INCOME_TOTAL）与贷款金额（AMT_CREDIT）为绝对金额，受地区物价、
    #    通胀与客户自身基数影响大，无法直接反映偿还压力。通过构造信贷收入比（杠杆倍数）、
    #    年金收入比（DTI 代理）、家庭人均可支配收入等无量纲比率，拉平不同客群的度量标尺。
    # 3. 风险机理：杠杆过高、月供吞噬大部分收入、月均可用净现金流接近或低于零，均预示着极高的流动性违约风险。
    # -------------------------------------------------------------------------
    # [1.1] 信贷收入比 (杠杆倍数)
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    # [1.2] 年金收入比 (DTI 负债收入比代理: 月供占年总收入比率，数值上等价于真实月度 DTI 的 1/12)
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # [1.3] 月均可用净现金流代理 (月均收入 - 单期月还款额，统一月度时间尺度)
    df['FREE_INCOME_PROXY'] = (df['AMT_INCOME_TOTAL'] / 12) - df['AMT_ANNUITY']
    # [1.4] 家庭人均年收入
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    # [1.5] 抚养子女摊薄人均年收入 (贷款人自身加计抚养子女)
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])

    # -------------------------------------------------------------------------
    # 维度 2: 贷款条件与抵押覆盖变量 (Loan Conditions & Collateral Coverage)
    # -------------------------------------------------------------------------
    # 【风控维度内涵与设计逻辑】：
    # 1. 业务含义：反映借款产品条款自身的结构特征、借款人自付成本（沉没成本）与机构的风险暴露敞口。
    # 2. 设计目的：评估信贷额度与标的商品价值（AMT_GOODS_PRICE）的匹配程度。自筹首付意味着客户拥有
    #    沉没资产与更强的还款意愿；而超融（贷款额超过商品原价，涵盖手续费、附加保险等）则代表极度资金匮乏。
    # 3. 风险机理：首付差额越负（超融加融越高）、抵押覆盖率越低，借款人弃贷/违约的沉没成本越低，坏账率显著上升。
    # -------------------------------------------------------------------------
    # [2.1] 还款费率 (年金信贷比 / 期限倒数代理)
    df['ANNUITY_TO_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    # [2.2] 预估还款期数 (贷款期限代理)
    df['ESTIMATED_TENURE'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    # [2.3] 商品价值与信贷额度比 (抵押覆盖率代理，>1 代表客户自付首付，<1 代表超额授信/加融)
    df['GOODS_TO_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    # [2.4] 首付与超融差额代理 (商品标价 - 信贷总额: >0 为自筹首付沉没成本; <0 为手续费/保险等超融加融敞口，差额越负违约率越高)
    df['DOWN_PAYMENT_PROXY'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    # [2.5] 单期月供与商品价值比
    df['ANNUITY_TO_GOODS_RATIO'] = df['AMT_ANNUITY'] / df['AMT_GOODS_PRICE']

    # -------------------------------------------------------------------------
    # 维度 3: 外部征信评分多维交互变量 (External Credit Scores Multi-Interaction - 全表判别力最强维度)
    # -------------------------------------------------------------------------
    # 【风控维度内涵与设计逻辑】：
    # 1. 业务含义：外部三方征信评级（EXT_SOURCE_1/2/3）是全表单一特征判别力（IV/AUC）最强的信息源，
    #    代表其他金融机构/征信局对借款人历史履约意愿和信用质量的综合评价。
    # 2. 设计目的：单家评级易受信息源覆盖面或统计偶发性影响。通过构建“均值（平滑波动）、标准差（捕捉评级分歧）、
    #    极值（木桶短板与天花板）、加权融合（按 IV 动态抗缺失）、非线性乘积（多机构高分共振）”，全方位提炼信用画像。
    # 3. 风险机理：任意机构亮红灯（极小值低）即触发木桶短板；三家机构评分离散度高反映客户信用存在争议；
    #    征信缺失数（白户度）则反映风控能见度。同时与杠杆和年龄交叉，可识别“高杠杆但高信用”与“虚假低杠杆”的客群分化。
    # -------------------------------------------------------------------------
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    # [3.1] 外部评分均值 (平滑单家机构波动)
    df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    # [3.2] 外部评分标准差 (离散度/评级冲突)
    df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
    # [3.3] 外部评分最低值 (木桶短板效应，任一机构亮红灯即预警)
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
    # [3.4] 外部评分最高值 (信用天花板)
    df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
    # [3.5] 联合乘积高阶非线性交互 (全量三方机构均有评分时生效，表征多机构共振；单一缺失时为 NaN 由树模型分支处理)
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    # [3.6] 经验加权评分 (按 IV 经验 2:3:5 权衡，动态按有效非空项权重和归一化至 [0, 1]，避免空值扩散)
    ext_weights = pd.Series({'EXT_SOURCE_1': 2.0, 'EXT_SOURCE_2': 3.0, 'EXT_SOURCE_3': 5.0})
    ext_num = (df[ext_cols] * ext_weights).sum(axis=1, min_count=1)
    ext_den = (df[ext_cols].notna() * ext_weights).sum(axis=1).replace(0, np.nan)
    df['EXT_SOURCE_WEIGHTED'] = ext_num / ext_den
    # [3.7] 外部评分缺失项个数 (征信白户度度量)
    df['EXT_SOURCE_NAN_CNT'] = df[ext_cols].isna().sum(axis=1)
    # [3.8] 综合资信与杠杆比率交叉 (资信 / 杠杆)
    df['EXT_MEAN_TO_CREDIT_RATIO'] = df['EXT_SOURCE_MEAN'] / df['CREDIT_TO_INCOME_RATIO']
    # [3.9] 综合资信与年龄交叉
    df['EXT_MEAN_TO_BIRTH'] = df['EXT_SOURCE_MEAN'] / (-df['DAYS_BIRTH'] / 365)

    # -------------------------------------------------------------------------
    # 维度 4: 时间稳定性与生命周期交叉变量 (Temporal Stability & Lifecycle Ratios)
    # -------------------------------------------------------------------------
    # 【风控维度内涵与设计逻辑】：
    # 1. 业务含义：衡量借款人的生活居住稳定性、职业扎根程度与生命周期成熟度。
    # 2. 设计目的：原始数据中的天数字段（DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION 等）均为负数，
    #    且绝对天数天然随年龄增长，容易造成树模型被年龄主导的伪相关。通过将工龄、户籍登记、换证、换手机
    #    等天数除以年龄或相互比值，转化为“无量纲的生命周期占比与相对稳定度”。
    # 3. 风险机理：工龄占年龄比过低、频繁变更手机号、户口迁徙频繁或短期内密集换证，通常表征生活漂泊、
    #    职业不稳定或失业换岗期，违约概率和失联欺诈概率显著偏高。
    # -------------------------------------------------------------------------
    # [4.1] 就业稳定性 (工龄占年龄比例)
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    # [4.2] 资产折旧度 (车龄占年龄比例)
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (-df['DAYS_BIRTH'] / 365)
    # [4.3] 身份证持有稳定度 (距上次换证时间占年龄比例，比值越大持证越久越稳定)
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    # [4.4] 户籍定居稳定度 (户口登记天数占年龄比)
    df['REG_TO_BIRTH_RATIO'] = df['DAYS_REGISTRATION'] / df['DAYS_BIRTH']
    # [4.5] 通信方式稳定性 (手机变更天数占年龄比)
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    # [4.6] 通信与工作稳定性匹配度 (换手机天数 / 工龄)
    df['PHONE_TO_EMPLOYED_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED'].replace(0, np.nan)
    # [4.7] 当前工作任职起始估计年龄
    df['AGE_AT_EMPLOYMENT'] = (-df['DAYS_BIRTH'] + df['DAYS_EMPLOYED']) / 365
    # [4.8] 户口与身份证更换时间比
    df['REG_TO_ID_RATIO'] = df['DAYS_REGISTRATION'] / df['DAYS_ID_PUBLISH'].replace(0, np.nan)

    # -------------------------------------------------------------------------
    # 维度 5: 缺失模式与资产完备度变量 (Missingness Patterns & Asset Completeness)
    # -------------------------------------------------------------------------
    # 【风控维度内涵与设计逻辑】：
    # 1. 业务含义：在信贷申请场景中，“缺失并非完全随机（MAR/NMAR），缺失本身即是高价值的风险信号”。
    # 2. 设计目的：针对房屋标准化信息（需有房资质才填报）、车辆信息（有车才填报车龄）提取专门的资产缺失标识；
    #    同时严格基于原始字段汇总全表缺失总数（TOTAL_NAN_COUNT），度量客户填报配合度与信息透明度。
    # 3. 风险机理：资产信息大面积缺失常对应无固定重资产的“无产/轻资产”客群；而申请资料全表缺失项畸高，
    #    往往反映借款人刻意隐匿负面信息、资质极差或通过黑中介粗制滥造申请，风险极高。
    # -------------------------------------------------------------------------
    # [5.1] 标准化房屋信息同源缺失率
    avg_house_cols = [c for c in house_cols_list if c.endswith('AVG') and c in df.columns]
    df['HOUSE_INFO_MISSING_RATE'] = df[avg_house_cols].isna().mean(axis=1)
    # [5.2] 名下无私家车二值标记
    df['FLAG_OWN_CAR_MISSING'] = df['OWN_CAR_AGE'].isna().astype(int)
    # [5.3] 原始填报资料缺失总项数 (严格基于原始特征字段统计，度量申请人资料完备度与透明度，杜绝衍生特征自引用污染)
    matched_raw_cols = [c for c in raw_feature_cols_list if c in df.columns]
    df['TOTAL_NAN_COUNT'] = df[matched_raw_cols].isna().sum(axis=1)

    # -------------------------------------------------------------------------
    # 维度 6: 身份真实性、反欺诈与社交圈共性风险变量 (Identity, Anti-Fraud & Social Risk)
    # -------------------------------------------------------------------------
    # 【风控维度内涵与设计逻辑】：
    # 1. 业务含义：第一道防线，核验借款人是否为真实合规个体、是否存在多地包装欺诈，以及其社交圈层的信用传染性。
    # 2. 设计目的：汇总证明材料提交量（资质可信度）；汇总户籍地、居住地、工作地等多重城市/区域错配均值；
    #    计算 30/60 DPD 社交圈违约人数占观察总人数的比率（圈层违约率）。
    # 3. 风险机理：证明文件提交越少、造假成本越低；地址错配越多，异地包装申贷与流动欺诈嫌疑越高；
    #    社交圈层逾期占比越高，说明其亲朋好友网络或共用联系人处于高违约恶化状态，信用传染风险极大。
    # -------------------------------------------------------------------------
    # [6.1] 提交证明材料与资质文件总数量
    doc_cols = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22) if f"FLAG_DOCUMENT_{i}" in df.columns]
    df['DOCUMENTS_PROVIDED_CNT'] = df[doc_cols].sum(axis=1)
    # [6.2] 地址多重错配均值 (户籍/工作/居住地一致性)
    addr_cols = [
        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION'
    ]
    matched_addr_cols = [c for c in addr_cols if c in df.columns]
    df['ADDRESS_MISMATCH_MEAN'] = df[matched_addr_cols].mean(axis=1)
    # [6.3] 逾期 30 DPD 社交圈违约人员占比 (逾期 30 天以上人数 / 观察总人数，度量信用圈层传染风险)
    df['DEF_OBS_RATIO_30'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)
    # [6.4] 逾期 60 DPD 社交圈重度违约人员占比 (逾期 60 天以上人数 / 观察总人数)
    df['DEF_OBS_RATIO_60'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['OBS_60_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)

    # -------------------------------------------------------------------------
    # 维度 7: 征信查询趋势与多头借贷激增变量 (Credit Inquiry Trends & Multi-Lending Spikes)
    # -------------------------------------------------------------------------
    # 【风控维度内涵与设计逻辑】：
    # 1. 业务含义：度量借款人的“资金饥渴度”以及近期是否发生“多头共债激增 / 集中借贷”。
    # 2. 设计目的：单纯查询总数可能受历史时间长度影响，无法反映当下的资金紧急状况。本维度汇总全量查询记录
    #    区分征信纯白户，并构造“近1个月短期查询 / 近1年长期查询”的环比趋势比率，消除绝对时间跨度干扰。
    # 3. 风险机理：短期内密集被银行或机构查询征信（激增比率高），通常代表借款人正在各平台疯狂借贷、
    #    甚至已被其他机构拒绝，正处于资金链濒临断裂的危机期，违约概率极高。
    # -------------------------------------------------------------------------
    bureau_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    matched_bureau = [c for c in bureau_cols if c in df.columns]
    # [7.1] 征信局历史查询总次数汇总 (全时段非重叠区间求和，min_count=1 保留全缺失语义，避免纯白户被隐式赋值为 0)
    df['BUREAU_REQ_TOTAL'] = df[matched_bureau].sum(axis=1, min_count=1)
    # [7.2] 央行征信记录标记 (有任一历史查询记录标记为 1，征信纯白户/查无征信标记为 0)
    df['FLAG_BUREAU_REQ'] = (df['BUREAU_REQ_TOTAL'] > 0).astype(int)
    # [7.3] 短期借贷激增比 (近1个月查询量 / 远期(3-12个月前)查询量加1平滑，捕捉短期密集借贷意图)
    short_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON'
    ]
    matched_short = [c for c in short_cols if c in df.columns]
    df['BUREAU_REQ_SHORT_TO_LONG'] = (
        df[matched_short].sum(axis=1, min_count=1) / (df['AMT_REQ_CREDIT_BUREAU_YEAR'] + 1)
    )


print("正在执行全量数据集清洗与 7 大通用维度特征衍生...")
for dataset_name, df_obj in [("训练集", train), ("验证集", valid), ("测试集", test)]:
    clean_anomalies(df_obj)
    add_all_derived_features(df_obj, house_cols, raw_feature_cols)
    print(f"✓ {dataset_name} 特征衍生完成，当前特征数: {df_obj.shape[1]}")

# %%
# =============================================================================
# 三、维度 8: 行业与职业分组基准统计量 (Industry & Occupation Benchmarks - 严格零数据穿越拟合)
# =============================================================================
# 【风控维度内涵与设计逻辑】：
# 1. 业务含义：度量借款人在所属职业群体（OCCUPATION_TYPE）和行业群体（ORGANIZATION_TYPE）内的
#    “相对收入位次与收入健康度”。
# 2. 设计目的：不同行业与职业天然存在薪酬壁垒（如 IT 研发人员与清洁工平均收入差距巨大）。直接比较绝对收入
#    会混淆行业固有差异；通过引入群体收入均值作为锚点基准，计算相对比率，能够剔除行业偏见，识别出
#    “即便在低薪行业中也属于高精尖稳健层”或“在高薪行业中却属于边缘低收入层”的真实抗风险分位。
# 3. 防数据穿越设计（Data Leakage Prevention）：
#    分组收入均值等统计量【严格仅在训练集 Train 上计算拟合】！
#    验证集与测试集仅执行 left join 映射匹配，杜绝统计量渗透与数据穿越。
# =============================================================================
print("正在拟合行业与职业分组基准统计量 (严格仅限训练集)...")
# [8.1] OCCUPATION_TYPE_INCOME_MEAN: 同一职业群体的平均收入基准
occupation_stats = train.groupby(
    'OCCUPATION_TYPE', observed=True
)['AMT_INCOME_TOTAL'].mean().rename('OCCUPATION_TYPE_INCOME_MEAN').reset_index()
train = train.merge(occupation_stats, on='OCCUPATION_TYPE', how='left')
valid = valid.merge(occupation_stats, on='OCCUPATION_TYPE', how='left')
test = test.merge(occupation_stats, on='OCCUPATION_TYPE', how='left')

# [8.2] INCOME_TO_OCCUPATION_TYPE_RATIO: 个人收入相对职业群体平均收入的比率
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

# [8.3] ORGANIZATION_TYPE_INCOME_MEAN: 同一行业群体的平均收入基准
organization_stats = train.groupby(
    'ORGANIZATION_TYPE', observed=True
)['AMT_INCOME_TOTAL'].mean().rename('ORGANIZATION_TYPE_INCOME_MEAN').reset_index()
train = train.merge(organization_stats, on='ORGANIZATION_TYPE', how='left')
valid = valid.merge(organization_stats, on='ORGANIZATION_TYPE', how='left')
test = test.merge(organization_stats, on='ORGANIZATION_TYPE', how='left')

# [8.4] INCOME_TO_ORGANIZATION_TYPE_RATIO: 个人收入相对行业群体平均收入的比率
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

# 清理内存
del train, valid, test, raw_features_summary, full_features_summary
del missing_rates, primary_value_ratios, feature_cols, ignore_cols, derived_cols
gc.collect()


# =============================================================================
# 架构附注：主申请表 8 大工程维度 与 现代信贷风控 8 大变量体系的映射关系
# =============================================================================
# 本脚本构造的 8 大衍生特征维度，是针对主申请表 application_{train|test}.csv
# 静态字段对银行业标准“风控 8 大核心变量体系”的落地剪裁与代理实现：
# - 工程维度 1 (还款能力与负债压力)   <--> 理论类别 1 (还款能力变量)
# - 工程维度 2 (贷款条件与抵押覆盖)   <--> 理论类别 7 (产品与借款条件变量)
# - 工程维度 3 (外部征信评分多维交互) <--> 理论类别 2 (历史信用表现变量 - 外部机构综合代理)
# - 工程维度 4 (时间稳定性与生命周期) <--> 理论类别 8 (时间变化与趋势变量 - 静态生命周期稳定性)
# - 工程维度 5 (缺失模式与资产完备度) <--> 理论类别 5 (身份真实性与反欺诈 - 填报配合度)
# - 工程维度 6 (身份真实性与社交圈)   <--> 理论类别 5 (身份真实性与反欺诈 - 圈层违约传染)
# - 工程维度 7 (征信查询趋势与多头)   <--> 理论类别 3 & 8 (多头借贷与短期激增趋势变量)
# - 工程维度 8 (行业与职业基准统计量) <--> 理论类别 1 (还款能力分群基准)
# 注：理论类别 2 (深度多期履约历史) 与 理论类别 6 (微观交易流水) 需在后续接入
#     bureau.csv、previous_application.csv 及 installments_payments.csv 补充强化。
# =============================================================================

# 风控建模中最重要的8类变量 (通用理论全景)

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