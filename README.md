# Home Credit Default Risk: 信贷违约预测与双轨建模
# Home Credit Default Risk: Dual-Track Modeling Pipeline (Scorecard & LightGBM)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/) [![Scorecard](https://img.shields.io/badge/Model-Scorecard%20(OptBinning)-orange.svg)](https://gnpalencia.org/optbinning/) [![Validation](https://img.shields.io/badge/Validation-Zero%20Data%20Leakage-red.svg)](#) [![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

[**中文说明 (Chinese)**](#中文说明) | [**English Documentation**](#english-documentation)

---

## 中文说明

### 1. 项目背景与核心设计

本项目基于 Kaggle 经典信贷风控赛题 [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)，参考信贷风控实践与合规审计要求构建。

针对申请主表（`application_train.csv`）正负样本不平衡（坏账率约 8.07%）、高维稀疏特征及多机构外部征信差异，项目建立了从分层抽样物理隔离、对抗性验证、特征工程漏斗初筛，到标准信用评分卡与 LightGBM 双轨建模的完整流程。

#### 核心设计点：

1. **分层抽样与物理隔离（防数据穿越）**：
   - 在进行探索性分析（EDA）、异常值截断、分箱拟合及 IV 筛选前，先执行 8:2 分层抽样，将 20% 的**闭卷验证集（Valid Blind，61,503 条）独立封存**。
   - 验证集仅套用开发集拟合好的规则执行转换（transform），避免全局统计量渗透导致的乐观评估偏差。
2. **训练集与测试集对抗性验证（Adversarial Validation）**：
   - 使用浅层、强正则化的 LightGBM 检验开发集与官方未标注测试集的宏观分布差异（AUC 为 0.89），识别出 `AMT_CREDIT`、`AMT_ANNUITY` 等分布变化较明显的变量，供特征筛选和稳定性评估参考。
3. **覆盖 8 个风控维度的特征工程**：
   - 衍生变量涵盖还款能力与负债杠杆（`ANNUITY_TO_INCOME_RATIO`、`INCOME_PER_CHILD`）、贷款期限与抵押覆盖（`GOODS_TO_CREDIT_RATIO`）、外部征信深度交互（`EXT_SOURCE` 均值/极小值/加权/乘积）、职业与联系方式稳定性比率、房产同源缺失模式，以及征信短期激增趋势比（`BUREAU_REQ_SHORT_TO_LONG`）。
4. **标准评分卡与 LightGBM 双轨建模**：
   - **轨道一（标准信用评分卡）**：基于 `OptBinning` 单调分箱、WOE 编码、相关性贪心剔除、方差膨胀因子（VIF <= 5）多重共线性处理、逐步回归（AIC 准则）以及逻辑回归系数符号保序校验，按商业银行常用的 PDO 刻度（基准分 600 分，Odds 翻倍 PDO = 20）输出整数分值评分卡表。
   - **轨道二（LightGBM 梯度提升决策树）**：结合分层 5 折交叉验证 OOF、累计增益（Gain Importance 前 95%）精筛与 Optuna TPE 贝叶斯调参，完成违约概率预测并输出评估指标。

---

### 2. 实验结果与评估指标

模型评估均在独立封存的**闭卷验证集（Valid Blind）**与**官方未标注测试集（Official Test）**上完成：

#### 2.1 模型综合性能指标对比

| 模型架构 | 数据集划分 | 样本量 | 区分度 (ROC-AUC) | 鉴别力 (KS) | 跨客群稳定性 (PSI) | 核心特性说明 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **LightGBM (5折CV + Optuna)** | 开发训练集 (Train Dev OOF) | 246,008 | **0.7601** | **0.3841** | 基准 (0.00000) | 5折交叉验证集成，综合非线性捕获能力强 |
| **LightGBM (5折CV + Optuna)** | 闭卷验证集 (Valid Blind) | 61,503 | **0.7572** | **0.3879** | — | 独立封存无偏差盲测评估 |
| **LightGBM (5折CV + Optuna)** | 官方测试集 (Official Test) | 48,744 | — | — | **0.00421** | 跨客群表现稳定 (PSI < 0.10) |
| **标准信用评分卡 (OptBinning)** | 开发训练集 (Train Dev) | 246,008 | 0.7517 | 0.3749 | 基准 (0.00000) | 单调WOE + VIF + 逐步回归，纯整数分值 |
| **标准信用评分卡 (OptBinning)** | 闭卷验证集 (Valid Blind) | 61,503 | 0.7470 | 0.3681 | — | 独立封存无偏差盲测评估 |
| **标准信用评分卡 (OptBinning)** | 官方测试集 (Official Test) | 48,744 | — | — | **0.00157** | 跨客群表现稳定 (PSI < 0.10) |

> **群体稳定性指标（PSI）**：常用标准中 PSI < 0.10 表示客群分布基本无变化。评分卡总分与 LightGBM 预测概率在测试集上的 PSI 分别为 0.00157 和 0.00421，未见明显客群漂移。

#### 2.2 样本分层审计账本 (`split_audit.csv`)

| 数据集 | 样本量 | 样本占比 | 正常客户 (0) | 违约客户 (1) | 坏账率 (Bad Rate) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 原始训练集 (Raw Train) | 307,511 | 100.00% | 282,686 | 24,825 | 8.0729% |
| **开发训练集 (Train Dev)** | 246,008 | 80.00% | 226,148 | 19,860 | 8.0729% |
| **闭卷验证集 (Valid Blind)** | 61,503 | 20.00% | 56,538 | 4,965 | 8.0728% |

---

### 3. 流水线执行架构

代码位于 `baseline/` 目录，按数字编号顺序执行：

```
                    [ 原始数据: data/raw/application_train.csv ]
                                       │
                ┌──────────────────────┴──────────────────────┐
                │ 00_stratified_split.py (8:2 分层切分与物理隔离) │
                └──────────────────────┬──────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
 [ 开发训练集: Train Dev (80%) ]                       [ 闭卷验证集: Valid Blind (20%) ]
  (仅用于拟合规则与参数学习)                               (完全封存，不参与任何规则 fit)
            │                                                     │
   ┌────────┴────────┐                                            │
   ▼                 ▼                                            │
01_eda.py    02_adversarial_validation.py                         │
(特征体检账本)  (LightGBM 探测跨期宏观漂移)                           │
   │                 │                                            │
   └────────┬────────┘                                            │
            ▼                                                     │
03_feature_engineering.py                                         │
(8大维度特征衍生 + IV/PSI 初筛 + 贪心去共线)                          │
            │                                                     │
            ├─────────────────────────────────┐                   │
            ▼                                 ▼                   │
04_scorecard_modeling.py             05_lightgbm_modeling.py      │
- 单调分箱 & WOE 编码                 - 5折交叉验证 OOF              │
- VIF <= 5 多重共线性消除             - 累计 Gain 精筛 (Top 95%)   │
- 逐步回归 (AIC) & 负符号约束          - Optuna TPE 贝叶斯调参        │
- PDO=20, 600 基准分刻度映射          - 浅树强正则抗噪约束          │
            │                                 │                   │
            └────────────────┬────────────────┘                   │
                             ▼                                    │
                    [ 统一闭卷映射与盲测验证 ] ◄────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
[ 盲测 AUC / KS 独立评估 ]         [ 官方测试集总分 / 概率 PSI 跨客群稳定性 ]
```

---

### 4. 核心脚本解析

- **`baseline/00_stratified_split.py`**：
  按标签分层抽样（`random_state=123`），将 30.7 万条原始训练数据划分为 80% 开发集与 20% 验证集。校验主键唯一性、样本交集（为 0）与坏账率偏差（< 0.01%），输出审计文件 `split_audit.csv`。
- **`baseline/01_eda.py`**：
  分析开发集 120 余个字段的统计量、缺失率、主值占比、单变量分箱、IV、JS 散度与 Gini 系数；处理房屋信息群聚缺失与 `DAYS_EMPLOYED=365243` 异常值，输出特征体检表 `cols_eda.csv`。
- **`baseline/02_adversarial_validation.py`**：
  以“是否属于测试集”（`is_test`）为标签训练 LightGBM，通过浅树与限制叶节点样本（`max_depth=5, num_leaves=16, min_child_samples=50`）检查训练集与测试集的分布差异，保存特征重要性表至 `adversarial_validation_feature_importance.csv`。
- **`baseline/03_feature_engineering.py`**：
  衍生 8 个维度的业务特征；仅在开发集上拟合 OptBinning，结合质量指标、IV >= 0.02、PSI <= 0.25、房屋特征去重以及 Spearman 秩相关系数贪心剔除（|r| > 0.80），筛选出 40 个候选特征。
- **`baseline/04_scorecard_modeling.py`**：
  构建标准信用评分卡。对学历做单调有序映射，通过 VIF 检验消除多重共线性（控制在 5 以内），使用双向逐步回归（AIC）并检查系数符号（确保均为负号）。按基准好坏比设定 PDO = 20、基准分 600 分，导出分箱得分表 `final_scorecard_table.csv`。
- **`baseline/05_lightgbm_modeling.py`**：
  构建单表衍生特征与行业/职业分组统计均值（统计量仅在开发集上计算）；通过 5 折交叉验证的累计增益筛选保留前 95% 特征；使用 Optuna 运行 30 次 TPE 贝叶斯搜索优化树结构与正则化参数；完成 5 折模型集成并评估验证集和官方测试集。

---

### 5. 核心特征分析

结合评分卡 IV 与 LightGBM 增益重要性（Gain Importance），违约区分度较高的特征主要集中在以下维度：

1. **外部征信评分**：
   `EXT_SOURCE_MEAN`、`EXT_SOURCE_MIN`、`EXT_SOURCE_3`、`EXT_SOURCE_2`。单变量 IV 超过 0.33，在 LightGBM 中贡献了近 50% 的分裂增益。均值有助于平滑单家机构波动，最小值反映较差维度的信用表现。
2. **负债杠杆与偿付能力**：
   `GOODS_TO_CREDIT_RATIO`（商品价值与贷款额度比，反映首付自筹比例）、`AMT_ANNUITY`（年还款金额）、`ANNUITY_TO_CREDIT_RATIO`（还款金额与贷款额度比，反映还款期限与成本）。
3. **稳定性与基本信息**：
   `DAYS_BIRTH`（年龄）、`DAYS_EMPLOYED`（工龄）、`EMPLOYED_TO_BIRTH_RATIO`（工龄占年龄比例，反映就业稳定程度）、`ORGANIZATION_TYPE` / `OCCUPATION_TYPE`（单位类型与职业）。
4. **行为变化与征信频次**：
   `DAYS_LAST_PHONE_CHANGE`（更换手机天数）、`TOTAL_NAN_COUNT`（整表信息缺失总项数，反映资料完整度）、`BUREAU_REQ_SHORT_TO_LONG`（短期与中长期征信查询次数比）。

---

### 6. 项目文件组织结构

```
home-credit-default-risk/
├── baseline/                               # 核心流水线脚本
│   ├── 00_stratified_split.py              # 1. 样本 8:2 分层抽样与物理隔离
│   ├── 01_eda.py                           # 2. 探索性数据分析与特征体检表构建
│   ├── 02_adversarial_validation.py        # 3. LightGBM 对抗性验证 (检验分布差异)
│   ├── 03_feature_engineering.py           # 4. 业务特征衍生与漏斗初筛
│   ├── 04_scorecard_modeling.py            # 5. 标准信用评分卡建模 (WOE/VIF/逐步回归/PDO)
│   └── 05_lightgbm_modeling.py             # 6. LightGBM 建模 (5折CV/Optuna调优/测试推理)
│
├── data/                                   # 数据目录 (.gitignore 已忽略大文件)
│   ├── raw/                                # 原始数据集
│   │   ├── application_train.csv           # 申请主表 (带标签, 30.7万条)
│   │   ├── application_test.csv            # 官方测试集 (未标注, 4.8万条)
│   │   └── HomeCredit_columns_description.csv # 字段描述说明表
│   ├── interim/                            # 物理隔离切分后的中间数据
│   │   ├── application_train_dev.csv       # 开发训练集 (80%, 24.6万条)
│   │   └── application_valid_blind.csv     # 独立闭卷验证集 (20%, 6.1万条)
│   └── processed/                          # 特征工程加工后宽表
│       ├── train_fe.csv                    # 开发集精选候选特征
│       └── test_fe.csv                     # 官方测试集特征表
│
├── outputs/                                # 实验输出、模型评估与审计报告
│   ├── validation/                         # 数据切分与验证检验
│   │   ├── split_audit.csv                 # 开发集与验证集分层切分审计报告
│   │   └── adversarial_validation_feature_importance.csv # 对抗验证特征重要性表
│   ├── eda/                                # 探索性分析全量字段体检账本
│   │   ├── cols_eda.csv                    # 原始 120+ 字段 IV/PSI/缺失/分箱详细报告
│   │   └── model_features_summary.csv      # 入模候选特征质量与筛选报告
│   └── models/                             # 模型文件与评估汇总
│       ├── final_scorecard_table.csv       # 标准评分卡各分箱得分明细表 (含PDO加减分)
│       ├── scorecard_eval_summary.csv      # 评分卡 AUC、KS 与测试集 PSI 汇总表
│       ├── lgb_feature_importance.csv      # LightGBM 增益特征重要性与累计 Gain 表
│       └── lgb_eval_summary.csv            # LightGBM 5折 OOF、闭卷验证与测试集评估汇总
│
├── .gitignore                              # Git 忽略配置
└── README.md                               # 项目说明文档 (中英双语)
```

---

### 7. 快速开始与运行指南

#### 7.1 环境配置

推荐使用 Python 3.10 或 3.11。安装依赖：

```bash
# 克隆代码仓库
git clone https://github.com/gorgeous211/home-credit-default-risk.git
cd home-credit-default-risk

# 安装核心依赖包
pip install numpy pandas scipy scikit-learn lightgbm optuna toad optbinning statsmodels
```

> **Windows 运行说明**：
> 如遇 OpenBLAS / LAPACK 动态链接库冲突，代码会自动优先加载虚拟环境的 `Library/bin` 路径；使用 Conda 环境通常更平稳。

#### 7.2 数据准备

1. 从 Kaggle 下载赛题数据：[Home Credit Default Risk Data](https://www.kaggle.com/competitions/home-credit-default-risk/data)
2. 解压后将 CSV 文件放置在 `data/raw/` 目录（需包含 `application_train.csv`、`application_test.csv` 与 `HomeCredit_columns_description.csv`）。

#### 7.3 运行步骤

脚本支持独立运行，请按编号顺序执行：

```bash
# 步骤 0: 执行分层样本切分与物理隔离 (生成 data/interim/* 与 split_audit.csv)
python baseline/00_stratified_split.py

# 步骤 1: 执行开发集特征分析 (输出 outputs/eda/cols_eda.csv)
python baseline/01_eda.py

# 步骤 2: 运行训练集与测试集对抗性验证 (输出 adversarial_validation_feature_importance.csv)
python baseline/02_adversarial_validation.py

# 步骤 3: 运行特征衍生与漏斗筛选 (生成 data/processed/* 与 model_features_summary.csv)
python baseline/03_feature_engineering.py

# 步骤 4: 训练标准信用评分卡模型并计算 PDO (输出 final_scorecard_table.csv 与评估指标)
python baseline/04_scorecard_modeling.py

# 步骤 5: 训练 LightGBM 决策树、执行 Optuna 调参并评估盲测表现
python baseline/05_lightgbm_modeling.py
```

---

### 8. 相关工具与参考

- **赛题来源**：[Kaggle - Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
- **评分卡工具**：[OptBinning: Optimal Binning for Credit Risk](https://github.com/guillermo-navas-palencia/optbinning)
- **特征工程工具**：[Toad: Industrial Credit Risk Scorecard Development](https://github.com/amphibian-dev/toad)
- **梯度提升树**：[LightGBM](https://github.com/microsoft/LightGBM)
- **超参数优化**：[Optuna](https://optuna.org/)

---

## English Documentation

### 1. Project Background and Key Design

This project builds a credit risk modeling pipeline based on the Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition, following standard credit underwriting and validation practices.

The primary applicant table (`application_train.csv`) has an 8.07% default rate, high missingness across multiple credit bureau fields, and significant variance across reporting agencies. The pipeline addresses these challenges through stratified data isolation, adversarial validation, funnel feature selection, and two parallel modeling tracks: a standard logistic regression scorecard and a LightGBM gradient boosted tree.

#### Key design points:

1. **Stratified data isolation (zero leakage)**:
   - Before running exploratory data analysis, outlier handling, binning, or feature selection, an 80/20 stratified split sets aside 20% of the training records as an independent validation set (61,503 rows).
   - The validation set applies only the transformation rules and parameters fitted on the development set, preventing data snooping and optimistic evaluation bias.
2. **Adversarial validation**:
   - A regularized LightGBM model tests for distribution differences between the development set and the unlabelled official test set (AUC = 0.89), identifying drifting variables such as `AMT_CREDIT`, `AMT_ANNUITY`, and bureau inquiry volumes.
3. **Feature engineering across eight risk dimensions**:
   - Derived features cover debt-to-income and payment burdens (`ANNUITY_TO_INCOME_RATIO`, `INCOME_PER_CHILD`), loan-to-goods ratios (`GOODS_TO_CREDIT_RATIO`), cross-bureau aggregations (`EXT_SOURCE` mean, min, product, and weighted score), employment stability ratios, housing missingness flags, and short-to-long term credit inquiry ratios (`BUREAU_REQ_SHORT_TO_LONG`).
4. **Dual-track modeling**:
   - **Track 1: Standard scorecard**: Uses `OptBinning` for monotonic optimal binning and WOE encoding, removes collinear variables using VIF (threshold 5), runs stepwise logistic regression by AIC, verifies all coefficient signs are negative, and scales points with PDO = 20 and a 600-point base score.
   - **Track 2: LightGBM**: Uses 5-fold stratified cross-validation with out-of-fold (OOF) predictions, selects features covering the top 95% of cumulative gain, and tunes hyperparameters with Optuna TPE.

---

### 2. Benchmark Evaluation and Results

Model performance was evaluated on the holdout validation set and the unlabelled official test set:

#### 2.1 Model performance comparison

| Model Framework | Dataset Split | Sample Size | ROC-AUC | KS Statistic | Population Stability (PSI) | Key Highlights |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **LightGBM (5-Fold CV + Optuna)** | Train Dev (5-Fold OOF) | 246,008 | **0.7601** | **0.3841** | Baseline (0.00000) | 5-fold CV ensemble capturing non-linear interactions |
| **LightGBM (5-Fold CV + Optuna)** | Valid Blind | 61,503 | **0.7572** | **0.3879** | — | Unbiased holdout validation |
| **LightGBM (5-Fold CV + Optuna)** | Official Test | 48,744 | — | — | **0.00421** | Stable score distribution (PSI < 0.10) |
| **Standard Scorecard (OptBinning)** | Train Dev | 246,008 | 0.7517 | 0.3749 | Baseline (0.00000) | Monotonic WOE, VIF <= 5, stepwise logistic regression |
| **Standard Scorecard (OptBinning)** | Valid Blind | 61,503 | 0.7470 | 0.3681 | — | Unbiased holdout validation |
| **Standard Scorecard (OptBinning)** | Official Test | 48,744 | — | — | **0.00157** | Stable score distribution (PSI < 0.10) |

> **Population Stability Index (PSI)**: By conventional standards, PSI < 0.10 indicates negligible population shift. Test set PSI was 0.00157 for scorecard points and 0.00421 for LightGBM default probabilities, showing consistent score distributions between train and test.

#### 2.2 Stratified split audit ledger (`split_audit.csv`)

| Dataset Partition | Sample Count | Percentage | Non-Default (0) | Default (1) | Bad Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Raw Training Table | 307,511 | 100.00% | 282,686 | 24,825 | 8.0729% |
| **Train Dev (Development Set)** | 246,008 | 80.00% | 226,148 | 19,860 | 8.0729% |
| **Valid Blind (Hold-Out Set)** | 61,503 | 20.00% | 56,538 | 4,965 | 8.0728% |

---

### 3. Pipeline Architecture

```
                    [ Raw Data: data/raw/application_train.csv ]
                                       │
                ┌──────────────────────┴──────────────────────┐
                │ 00_stratified_split.py (80/20 Stratified Split) │
                └──────────────────────┬──────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
 [ Train Dev: 80% (246k rows) ]                        [ Valid Blind: 20% (61k rows) ]
  (Fitting rules & model parameter learning)             (Strictly frozen, zero fitting)
            │                                                     │
   ┌────────┴────────┐                                            │
   ▼                 ▼                                            │
01_eda.py    02_adversarial_validation.py                         │
(Feature Ledger) (LightGBM Shift Detection)                       │
   │                 │                                            │
   └────────┬────────┘                                            │
            ▼                                                     │
03_feature_engineering.py                                         │
(8 Risk Dimensions + IV/PSI Funnel + Greedy Corr Elimination)     │
            │                                                     │
            ├─────────────────────────────────┐                   │
            ▼                                 ▼                   │
04_scorecard_modeling.py             05_lightgbm_modeling.py      │
- Monotonic Optimal Binning           - 5-Fold Stratified CV OOF  │
- VIF <= 5 Collinearity Cleanup       - Cumulative Gain (Top 95%) │
- Stepwise LR (AIC) + Negative Sign   - Optuna TPE Bayesian Tuning│
- PDO=20, Base 600 Points Mapping     - Regularized Shallow Trees │
            │                                 │                   │
            └────────────────┬────────────────┘                   │
                             ▼                                    │
               [ Unsupervised Blind Transform & Inference ] ◄─────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
[ Independent Valid AUC / KS ]    [ Official Test PSI Stability Evaluation ]
```

---

### 4. Pipeline Scripts

- **`baseline/00_stratified_split.py`**:
  Splits the 307,511 raw records into an 80% development set and a 20% validation set using stratified sampling (`random_state=123`). Verifies ID uniqueness, zero overlap, and matched bad rates (< 0.01% difference), saving `split_audit.csv`.
- **`baseline/01_eda.py`**:
  Profiles all 120+ fields in the development set for missingness, primary value concentrations, optimal binning, IV, JS divergence, and Gini scores. Accounts for correlated housing missingness and the `DAYS_EMPLOYED=365243` retiree flag, outputting `cols_eda.csv`.
- **`baseline/02_adversarial_validation.py`**:
  Trains a LightGBM model to predict whether a sample belongs to the test set (`is_test`). Constrained hyperparameters (`max_depth=5`, `num_leaves=16`, `min_child_samples=50`) measure macro-level drift (AUC = 0.89) and save rankings to `adversarial_validation_feature_importance.csv`.
- **`baseline/03_feature_engineering.py`**:
  Derives risk features across eight domains. Fits OptBinning on the development set only, filters by data quality, IV >= 0.02, and PSI <= 0.25, removes redundant housing fields, and applies greedy Spearman correlation pruning (|r| > 0.80) to select 40 candidate features.
- **`baseline/04_scorecard_modeling.py`**:
  Builds a standard scorecard. Maps education to monotonic integer tiers, eliminates collinearity with iterative VIF pruning (VIF <= 5), runs bidirectional stepwise logistic regression (AIC), and verifies that all coefficients remain negative. Applies PDO = 20 and Base = 600 scaling to write `final_scorecard_table.csv`.
- **`baseline/05_lightgbm_modeling.py`**:
  Computes interaction features and occupation group mean income (calculated strictly on the development set). Uses 5-fold cross-validation to select features covering the top 95% of cumulative gain, tunes tree structure and regularization over 30 Optuna TPE trials, and evaluates both the validation set and official test set.

---

### 5. Important Features

Features with the strongest predictive power across both the scorecard and LightGBM fall into four groups:

1. **External credit bureau scores**:
   `EXT_SOURCE_MEAN`, `EXT_SOURCE_MIN`, `EXT_SOURCE_3`, and `EXT_SOURCE_2`. Individual scores have IV above 0.33, and together they account for nearly half of LightGBM's total split gain. The mean smooths reporting noise across bureaus, while the minimum captures the worst recorded credit history.
2. **Debt burden and collateral coverage**:
   `GOODS_TO_CREDIT_RATIO` (goods price divided by credit, indicating down payment contribution), `AMT_ANNUITY` (annual debt service amount), and `ANNUITY_TO_CREDIT_RATIO` (annuity relative to credit, reflecting interest rate and loan term).
3. **Demographics and employment stability**:
   `DAYS_BIRTH` (age), `DAYS_EMPLOYED` (length of current employment), `EMPLOYED_TO_BIRTH_RATIO` (share of adult life spent employed), and `ORGANIZATION_TYPE` / `OCCUPATION_TYPE` (employer sector and job type).
4. **Application behavior and inquiry trends**:
   `DAYS_LAST_PHONE_CHANGE` (recency of phone update), `TOTAL_NAN_COUNT` (total missing fields across the application), and `BUREAU_REQ_SHORT_TO_LONG` (ratio of recent bureau inquiries to annual inquiries).

---

### 6. Directory Structure

```
home-credit-default-risk/
├── baseline/                               # Core pipeline scripts
│   ├── 00_stratified_split.py              # 1. 80/20 Stratified split & physical isolation
│   ├── 01_eda.py                           # 2. EDA & comprehensive feature quality ledger
│   ├── 02_adversarial_validation.py        # 3. LightGBM adversarial validation
│   ├── 03_feature_engineering.py           # 4. Feature engineering & funnel screening
│   ├── 04_scorecard_modeling.py            # 5. Scorecard modeling (WOE/VIF/Stepwise/PDO)
│   └── 05_lightgbm_modeling.py             # 6. LightGBM modeling (5-Fold CV/Optuna/Inference)
│
├── data/                                   # Data directory (.gitignored)
│   ├── raw/                                # Kaggle competition raw files
│   │   ├── application_train.csv           # Main application table with labels (307k)
│   │   ├── application_test.csv            # Official test set without labels (48k)
│   │   └── HomeCredit_columns_description.csv # Official column descriptions
│   ├── interim/                            # Physically isolated partitions
│   │   ├── application_train_dev.csv       # Development train set (80%, 246k)
│   │   └── application_valid_blind.csv     # Frozen blind validation set (20%, 61k)
│   └── processed/                          # Processed wide tables for modeling
│       ├── train_fe.csv                    # Screened development feature table
│       └── test_fe.csv                     # Processed test table
│
├── outputs/                                # Artifacts, logs, and evaluation reports
│   ├── validation/                         # Validation and split audits
│   │   ├── split_audit.csv                 # Train/Valid stratification audit report
│   │   └── adversarial_validation_feature_importance.csv # Drift feature importance
│   ├── eda/                                # Full feature profiling ledgers
│   │   ├── cols_eda.csv                    # 120+ features with IV, PSI, and bin details
│   │   └── model_features_summary.csv      # Pre-screened feature summary ledger
│   └── models/                             # Model weights, scorecards & metrics
│       ├── final_scorecard_table.csv       # Production-ready point allocation scorecard
│       ├── scorecard_eval_summary.csv      # Scorecard AUC, KS, and PSI evaluation
│       ├── lgb_feature_importance.csv      # LightGBM gain importance and cumulative gain
│       └── lgb_eval_summary.csv            # LightGBM 5-Fold OOF, blind, and test summary
│
├── .gitignore                              # Git exclusion rules
└── README.md                               # Bilingual project documentation
```

---

### 7. Quick Start and Setup

#### 7.1 Environment Setup

Python 3.10 or 3.11 is recommended. Install dependencies:

```bash
# Clone the repository
git clone https://github.com/gorgeous211/home-credit-default-risk.git
cd home-credit-default-risk

# Install dependencies
pip install numpy pandas scipy scikit-learn lightgbm optuna toad optbinning statsmodels
```

> **Windows Platform Note**:
> If LAPACK/OpenBLAS DLL conflict warnings occur on Windows, the scripts automatically prepend `Library/bin` from the active environment to `PATH`. Using a Conda environment avoids these conflicts.

#### 7.2 Data Preparation

1. Download dataset files from [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data).
2. Place unzipped CSV files into `data/raw/` (`application_train.csv`, `application_test.csv`, and `HomeCredit_columns_description.csv`).

#### 7.3 Running the Pipeline

Run each script in numeric order:

```bash
# Step 0: Perform stratified split & zero-leakage physical isolation
python baseline/00_stratified_split.py

# Step 1: Execute EDA and generate feature audit ledger
python baseline/01_eda.py

# Step 2: Run adversarial validation to detect distribution shift
python baseline/02_adversarial_validation.py

# Step 3: Run advanced feature engineering and funnel selection
python baseline/03_feature_engineering.py

# Step 4: Fit standard credit scorecard with PDO point scaling
python baseline/04_scorecard_modeling.py

# Step 5: Train LightGBM with Optuna TPE tuning and evaluate blind test
python baseline/05_lightgbm_modeling.py
```

---

### 8. References

- **Competition**: [Kaggle - Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
- **Scorecard Tool**: [OptBinning: Optimal Binning for Credit Risk](https://github.com/guillermo-navas-palencia/optbinning)
- **Feature Selection**: [Toad: Industrial Credit Risk Scorecard Development](https://github.com/amphibian-dev/toad)
- **Gradient Boosting**: [LightGBM](https://github.com/microsoft/LightGBM)
- **Hyperparameter Search**: [Optuna](https://optuna.org/)
