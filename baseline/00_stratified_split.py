# %%
# =============================================================================
# 数据预处理：原始训练集分层切分（训练集与验证集）
# =============================================================================
# 一、背景与目的
# 避免数据穿越（Data Leakage）：
# 若在切分训练集与验证集前对全量数据统计缺失率、计算截断阈值或用标签计算 IV/WOE，
# 验证集信息会提前渗入，导致本地评估的 AUC/KS 虚高。
# 因此在开展 EDA 与特征工程前，先切分并隔离训练集与验证集。
#
# 二、采用分层抽样（Stratified Sampling）的原因
# 违约样本占比约 8.07%，样本分布不均衡。
# 普通随机抽样可能因抽样波动造成验证集坏账率偏移，影响 Base Odds 与概率标定。
# 分层抽样可保持切分前后两份数据的违约率一致。
#
# 三、数据集划分
# 1. 训练集（Train，80%）：
#    用于 EDA、数据清洗规则定义、分箱拟合、IV 计算、特征筛选及模型训练。
# 2. 验证集（Valid，20%）：
#    独立封存，不参与规则拟合；模型训练完成后仅执行映射与推理，用于评估 AUC 和 KS。
# 3. 测试集（Test，4.8 万条）：
#    对应 application_test.csv，无 TARGET 标签，用于评估跨客群预测分值的稳定性（PSI）。
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
INTERIM_DATA_PATH.mkdir(parents=True, exist_ok=True)

OUTPUTS_PATH = Path.cwd() / "outputs"
VALIDATION_OUTPUT_PATH = OUTPUTS_PATH / "validation"

# 数据文件路径
RAW_TRAIN_PATH = RAW_DATA_PATH / "application_train.csv"
TRAIN_PATH = INTERIM_DATA_PATH / "application_train_split.csv"
VALID_PATH = INTERIM_DATA_PATH / "application_valid.csv"
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

raw_train['TARGET'].isna().sum() == 0
# True (无缺失值)

unique_targets = set(raw_train['TARGET'].unique())
unique_targets.issubset({0, 1})
# True (二元标签 0 和 1)

overall_bad_count = int(raw_train['TARGET'].sum())
overall_good_count = len(raw_train) - overall_bad_count
overall_bad_rate = raw_train['TARGET'].mean()

print(f"标签分布: 正常(0) = {overall_good_count:,} | 违约(1) = {overall_bad_count:,} | 坏账率 = {overall_bad_rate:.4%}")
# 标签分布: 正常(0) = 282,686 | 违约(1) = 24,825 | 坏账率 = 8.0729%

# 清理核验临时变量
del unique_targets

# %%
# =============================================================================
# 2. 执行分层抽样切分
# =============================================================================
# 切分参数：
# - test_size = 0.20：按 8:2 切分，验证集含约 4,965 个违约样本，保证评估稳定性。
# - random_state = 123：固定随机种子，保证切分结果可复现。
# - shuffle = True：切分前打乱样本顺序。
# - stratify = raw_train['TARGET']：按标签比例分层抽样，保持两份数据坏账率一致。
# =============================================================================

train, valid = train_test_split(
    raw_train,
    test_size=0.20,
    random_state=123,
    shuffle=True,
    stratify=raw_train['TARGET']
)

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)

# %%
# =============================================================================
# 3. 切分结果核验与质量审计
# =============================================================================
# 1. 验证两份数据无重叠主键
train_ids = set(train['SK_ID_CURR'])
valid_ids = set(valid['SK_ID_CURR'])
intersection_ids = train_ids.intersection(valid_ids)

len(intersection_ids) == 0
# 训练集与验证集无样本重叠

# 2. 核对坏账率是否一致
train_bad_rate = train['TARGET'].mean()
valid_bad_rate = valid['TARGET'].mean()

split_audit_df = pd.DataFrame({
    '数据集': ['原始训练集 (Raw Train)', '训练集 (Train)', '验证集 (Valid)'],
    '样本量': [len(raw_train), len(train), len(valid)],
    '样本占比': ['100.00%', f"{len(train)/len(raw_train):.2%}", f"{len(valid)/len(raw_train):.2%}"],
    '正常客户(0)': [overall_good_count, int((train['TARGET'] == 0).sum()), int((valid['TARGET'] == 0).sum())],
    '违约客户(1)': [overall_bad_count, int((train['TARGET'] == 1).sum()), int((valid['TARGET'] == 1).sum())],
    '坏账率': [f"{overall_bad_rate:.4%}", f"{train_bad_rate:.4%}", f"{valid_bad_rate:.4%}"]
})

abs(train_bad_rate - valid_bad_rate) < 1e-4

# 导出切分审计报告至 outputs/validation
split_audit_df.to_csv(SPLIT_AUDIT_PATH, index=False)

# 清理核验临时变量与原始数据
del train_ids, valid_ids, intersection_ids
del train_bad_rate, valid_bad_rate
del raw_train, overall_good_count, overall_bad_count, overall_bad_rate
del split_audit_df
gc.collect()

# %%
# =============================================================================
# 4. 数据文件落盘
# =============================================================================
# 输出说明：
# 1. application_train_split.csv（80% 训练集）：
#    保存在 data/interim/，作为下游各建模脚本的训练输入。
# 2. application_valid.csv（20% 验证集）：
#    保存在 data/interim/，作为独立验证集，仅用于评估阶段的映射与推理。
# =============================================================================
# 导出训练集与验证集
train.to_csv(TRAIN_PATH, index=False)
valid.to_csv(VALID_PATH, index=False)

# 清理落盘中间变量并释放内存
del train, valid
gc.collect()
