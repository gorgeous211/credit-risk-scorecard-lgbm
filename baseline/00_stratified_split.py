# %%
# =============================================================================
# 数据预处理：样本分层切分与验证集独立封存
# =============================================================================
# 一、背景与设计目的
# 为避免数据穿越（Data Leakage）与数据窥探（Data Snooping）：
# 若在切分训练集与验证集之前，对全量数据计算缺失率、同值率、异常值截断阈值，
# 或利用全量标签计算 IV/WOE 筛选特征，验证集信息就会提前渗入建模流程，
# 导致本地评估的 AUC 和 KS 产生乐观偏差，无法客观反映未见样本上的表现。
#
# 因此，在开展任何探索性分析（EDA）与特征工程前，先将原始数据做物理隔离。
#
# 二、为什么采用分层抽样（Stratified Sampling）
# 申请主表违约样本仅占约 8.07%，属于典型的不平衡分类场景。
# 普通随机抽样容易因抽样波动导致验证集的坏样本比例产生偏差，
# 进而影响基准好坏比（Base Odds）与违约概率的标定。
# 采用分层抽样能使开发集与验证集的先验违约率严格一致。
#
# 三、数据集划分与分工
# 1. 开发训练集（Train Dev, 80%）：
#    用于后续的 EDA 探索、缺失值与异常值规则定义、分箱拟合（fit）、
#    IV 计算与特征筛选、模型训练及内部交叉验证。
# 2. 闭卷验证集（Valid Blind, 20%）：
#    在建模完成前完全独立封存，不参与任何规则拟合；
#    模型训练完成后仅执行映射（transform）与推理，用于独立评估 AUC 与 KS。
# 3. 官方未标注测试集（Official Test, 4.8 万条）：
#    无 TARGET 标签，直接输入训练好的模型输出预测分值，
#    用于评估跨客群模型总分 PSI，检验模型面对未知分布时的稳定性。
# =============================================================================

# %%
# 基础库导入
import gc
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================================================
# 路径配置
# =============================================================================
DATA_PATH = Path.cwd() / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
INTERIM_DATA_PATH = DATA_PATH / "interim"

OUTPUTS_PATH = Path.cwd() / "outputs"
VALIDATION_OUTPUT_PATH = OUTPUTS_PATH / "validation"

# 数据文件路径
RAW_TRAIN_PATH = RAW_DATA_PATH / "application_train.csv"
TRAIN_DEV_PATH = INTERIM_DATA_PATH / "application_train_dev.csv"
VALID_BLIND_PATH = INTERIM_DATA_PATH / "application_valid_blind.csv"
SPLIT_AUDIT_PATH = VALIDATION_OUTPUT_PATH / "split_audit.csv"

# %%
# =============================================================================
# 1. 原始训练表读取与基础校验
# =============================================================================
raw_train = pd.read_csv(RAW_TRAIN_PATH)

# 数据规模核验
raw_train.shape
# (307511, 122)

# 主键唯一性检查
raw_train.duplicated(subset='SK_ID_CURR').any()
# False

# 标签列检查
'TARGET' in raw_train.columns
# True
# 标签列在训练集的列中

raw_train['TARGET'].isna().sum() == 0
# True
# 没有缺失值

unique_targets = set(raw_train['TARGET'].unique())
unique_targets.issubset({0, 1})
# True
# 只有0、1二元取值

overall_bad_count = int(raw_train['TARGET'].sum())
overall_good_count = len(raw_train) - overall_bad_count
overall_bad_rate = raw_train['TARGET'].mean()

print(f"标签分布: 正常(0) = {overall_good_count:,} | 违约(1) = {overall_bad_count:,} | 坏账率 = {overall_bad_rate:.4%}")
# 标签分布: 正常(0) = 282,686 | 违约(1) = 24,825 | 坏账率 = 8.0729%

# 清理步骤 1 校验中间变量
del unique_targets

# %%
# =============================================================================
# 2. 执行分层抽样切分
# =============================================================================
# 参数设计考量：
# - test_size = 0.20：按 8:2 划分，开发集保留约 24.6 万条样本以学习细分特征，
#   验证集保留约 4,965 个违约样本，保证 KS 与 AUC 评估的统计稳定性。
# - random_state = 123：固定随机种子，保证切分结果可复现。
# - shuffle = True：切分前打乱样本顺序，消除可能的物理存储序列关联。
# - stratify = raw_train['TARGET']：按标签比例分层抽样，保持两份数据坏账率一致。
# =============================================================================

train_dev, valid_blind = train_test_split(
    raw_train,
    test_size=0.20,
    random_state=123,
    shuffle=True,
    stratify=raw_train['TARGET']
)

train_dev = train_dev.reset_index(drop=True)
valid_blind = valid_blind.reset_index(drop=True)

# %%
# =============================================================================
# 3. 切分结果核验与质量审计
# =============================================================================
# 1. 验证两份数据无重叠主键
train_ids = set(train_dev['SK_ID_CURR'])
valid_ids = set(valid_blind['SK_ID_CURR'])
intersection_ids = train_ids.intersection(valid_ids)

len(intersection_ids) == 0
# 训练集与验证集无样本重叠

# 2. 核对坏账率是否一致
train_bad_rate = train_dev['TARGET'].mean()
valid_bad_rate = valid_blind['TARGET'].mean()

split_audit_df = pd.DataFrame({
    '数据集': ['原始训练表 (Raw)', '开发训练集 (Train Dev)', '闭卷验证集 (Valid Blind)'],
    '样本量': [len(raw_train), len(train_dev), len(valid_blind)],
    '样本占比': ['100.00%', f"{len(train_dev)/len(raw_train):.2%}", f"{len(valid_blind)/len(raw_train):.2%}"],
    '正常客户(0)': [overall_good_count, int((train_dev['TARGET'] == 0).sum()), int((valid_blind['TARGET'] == 0).sum())],
    '违约客户(1)': [overall_bad_count, int((train_dev['TARGET'] == 1).sum()), int((valid_blind['TARGET'] == 1).sum())],
    '坏账率': [f"{overall_bad_rate:.4%}", f"{train_bad_rate:.4%}", f"{valid_bad_rate:.4%}"]
})

abs(train_bad_rate - valid_bad_rate) < 1e-4

# 导出切分审计报告至 outputs/validation
split_audit_df.to_csv(SPLIT_AUDIT_PATH, index=False)

# 清理步骤 3 核验中间变量及已完成使命的原始数据
del train_ids, valid_ids, intersection_ids
del train_bad_rate, valid_bad_rate
del raw_train, overall_good_count, overall_bad_count, overall_bad_rate
del split_audit_df
gc.collect()

# %%
# =============================================================================
# 4. 数据文件落盘
# =============================================================================
# 输出约定：
# 1. application_train_dev.csv：
#    作为 01_eda.py、03_feature_engineering.py、04_scorecard_modeling.py
#    和 05_lightgbm_modeling.py 的唯一开发输入。
# 2. application_valid_blind.csv：
#    作为模型完成训练后的独立验证集，仅执行无监督映射与评分评估。
# =============================================================================
# 导出训练集与验证集
train_dev.to_csv(TRAIN_DEV_PATH, index=False)
valid_blind.to_csv(VALID_BLIND_PATH, index=False)

# 清理落盘中间变量与数据集，释放内存
del train_dev, valid_blind
gc.collect()
