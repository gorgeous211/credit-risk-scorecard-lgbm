# Home Credit Default Risk: 传统统计评分卡（LR）与机器学习提升树（LightGBM）信贷违约预测

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/) [![Package Manager](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv) [![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/) [![Scorecard](https://img.shields.io/badge/Model-Scorecard%20(OptBinning)-orange.svg)](https://gnpalencia.org/optbinning/) [![Validation](https://img.shields.io/badge/Validation-Zero%20Data%20Leakage-red.svg)](#) [![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 1. 项目背景与核心设计

本项目基于 Kaggle 经典信贷风控赛题 [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)，结合信贷风控实践与合规审计要求构建。

申请主表（`application_train.csv`）存在正负样本不平衡（违约率约 8.07%）、高维稀疏特征及多机构征信差异等特点。本项目搭建了分层抽样、对抗性验证、特征工程以及双轨建模（LR 评分卡与 LightGBM）的完整流程。

### 核心设计点：

1. **分层抽样与物理隔离（防数据穿越）**：
   - 在探索性数据分析（EDA）、异常值处理、分箱和特征筛选前，先按 8:2 分层切分，将 20% 的**验证集（Valid，61,503 条）单独封存**。
   - 验证集仅套用训练集拟合的规则进行转换（transform），避免全局统计量渗透造成的评估偏差。
2. **训练集与测试集对抗性验证（Adversarial Validation）**：
   - 使用浅层、强正则化的 LightGBM 检验训练集与测试集的分布差异（AUC 为 0.89），识别出 `AMT_CREDIT`、`AMT_ANNUITY` 等分布变化较明显的变量，供特征筛选和稳定性评估参考。
3. **8 个维度的特征工程（共 42 个衍生特征）**：
   - **异常清洗与状态衍生**：将 `DAYS_EMPLOYED == 365243`（约1000年业务编码）重置为 NaN，并提炼独立二值特征 `FLAG_DAYS_EMPLOYED_ANOMALOUS` 标识退休或无正规就业客群；将未知性别替换为众数，并将社交圈人数极端离群值（>30）联动置空。
   - **维度 1: 还款能力与负债压力变量（5 个）**：信贷收入比（杠杆倍数）、年金收入比（`ANNUITY_TO_INCOME_RATIO`，即月供占年收入比例，相当于月度 DTI 的 1/12）、月均可用净现金流代理（`FREE_INCOME_PROXY`，月均收入减去当期月供）、家庭人均年收入及抚养子女摊薄人均年收入（`INCOME_PER_CHILD`），通过相对比率降低绝对收入和地区物价差异的影响；
   - **维度 2: 贷款条件与抵押覆盖变量（5 个）**：还款费率（`ANNUITY_TO_CREDIT_RATIO`）、预估还款期数（`ESTIMATED_TENURE`）、商品价值与信贷额度比（`GOODS_TO_CREDIT_RATIO`，抵押覆盖率代理）、首付与超融差额代理（`DOWN_PAYMENT_PROXY`，正值反映自筹首付沉没成本，负值反映手续费与保险加融的超额授信敞口，差额越负违约率越高）及单期月供与商品价值比（`ANNUITY_TO_GOODS_RATIO`）；
   - **维度 3: 外部征信评分多维交互变量（9 个）**：基于三大外部评分（`EXT_SOURCE_1/2/3`）构建均值、极小值、极大值、标准差、联合乘积、动态归一化的经验加权评分（`EXT_SOURCE_WEIGHTED`，按 IV 2:3:5 权重折算）、缺失项数，以及评分与信贷收入比、年龄的交叉特征；
   - **维度 4: 时间稳定性与生命周期交叉变量（8 个）**：将原始负数天数转为生命周期相对比例，避免年龄自然增长带来的伪相关。包括工龄占年龄比（`EMPLOYED_TO_BIRTH_RATIO`）、车龄占年龄比（`CAR_TO_BIRTH_RATIO`）、身份证持有稳定度、户籍定居稳定度、通信方式稳定度（`PHONE_TO_BIRTH_RATIO`）、换手机天数相对工龄匹配度（`PHONE_TO_EMPLOYED_RATIO`）、入职起始估计年龄（`AGE_AT_EMPLOYMENT`）及户籍与身份证变更时间比；
   - **维度 5: 缺失模式与资产完备度变量（3 个）**：记录缺失模式：包括房屋信息集中缺失率（`HOUSE_INFO_MISSING_RATE`，反映无房产资产情况）、无私家车标记（`FLAG_OWN_CAR_MISSING`），以及仅统计原始字段的缺失总数（`TOTAL_NAN_COUNT`，反映填报配合度，避免将衍生列的空值重复计入）；
   - **维度 6: 身份真实性、反欺诈与社交圈风险变量（4 个）**：提交证明材料总数量（`DOCUMENTS_PROVIDED_CNT`，衡量造假成本）、户籍/工作/居住地多重错配均值（`ADDRESS_MISMATCH_MEAN`，度量跨地域异常或异地包装嫌疑），以及逾期 30/60 DPD 社交圈违约人员传染占比（`DEF_OBS_RATIO_30/60`）；
   - **维度 7: 征信查询趋势与多头借贷激增变量（3 个）**：央行征信局历史查询总次数汇总（`BUREAU_REQ_TOTAL`，`min_count=1` 保留未查征信的空值语义）、央行征信记录标记（`FLAG_BUREAU_REQ`），以及近 1 个月查询量相对远期（3~12 个月前）的短期借贷激增比（`BUREAU_REQ_SHORT_TO_LONG`），捕捉短时间内集中借贷的资金压力；
   - **维度 8: 行业与职业分组基准统计量（4 个，严格零穿越设计）**：仅在训练集上按职业和行业聚合计算收入均值，再通过 `left join` 映射计算借款人在同业中的相对收入比率（`INCOME_TO_OCCUPATION_TYPE_RATIO`、`INCOME_TO_ORGANIZATION_TYPE_RATIO`），消除行业间基准差异；
   - **全量特征宽表构建**：由 `03_feature_engineering.py` 统一生成并保存候选特征宽表（120 个原始特征 + 1 个异常标记 + 41 个业务衍生 = 162 个候选特征，宽表共 164 列含 ID 与 TARGET），供评分卡与 LightGBM 统一读取与分流适配。
4. **传统统计评分卡（LR）与机器学习提升树（LightGBM）建模**：
   - **传统统计评分卡（Logistic Regression）**：对接特征宽表，经数据质量粗筛、批量 IV 初筛（$\ge 0.02$）、房屋同源去重、Spearman 贪心去共线（$|r| \ge 0.80$）、`OptBinning` 约束分箱（单箱样本 $\ge 5\%$、坏样本 $\ge 30$、数值单调与比率 U 型放宽）、方差膨胀因子（VIF <= 5）多重共线性处理、双向逐步回归（AIC 准则）以及逻辑回归系数负符号严格校验，按常用的 PDO 刻度（基准分 600 分，Odds 翻倍 PDO = 20）输出整数评分卡。
   - **机器学习提升树（LightGBM）**：对接特征宽表，保留连续分布与缺失分支，结合分层 5 折交叉验证 OOF、累计增益（Gain Importance 前 95%）精选与 Optuna TPE 贝叶斯搜索浅树与强正则超参数，完成违约概率预测并输出评估指标。

---

## 2. 实验结果与评估指标

模型评估均在独立封存的**验证集（Valid）**与**测试集（Test）**上完成：

### 2.1 模型综合性能指标对比

| 模型架构                         | 数据集划分                 | 样本量     | 区分度 (ROC-AUC) | 鉴别力 (KS)   | 跨客群稳定性 (PSI) | 核心特性说明                   |
|:---------------------------- |:--------------------- |:-------:|:-------------:|:----------:|:------------:|:------------------------ |
| **LightGBM (5折CV + Optuna)** | 训练集 (Train 5-Fold OOF) | 246,008 | **0.7676**    | **0.3991** | 基准 (0.00000) | 5折交叉验证集成，具备非线性建模能力      |
| **LightGBM (5折CV + Optuna)** | 验证集 (Valid)         | 61,503  | **0.7658**    | **0.3982** | —            | 独立验证集评估                    |
| **LightGBM (5折CV + Optuna)** | 测试集 (Test)          | 48,744  | —             | —          | **0.00141**  | 跨客群表现稳定 (PSI < 0.10)      |
| **标准信用评分卡 (OptBinning)**     | 训练集 (Train)         | 246,008 | 0.7523        | 0.3747     | 基准 (0.00000) | 单调WOE + VIF + 逐步回归，纯整数分值 |
| **标准信用评分卡 (OptBinning)**     | 验证集 (Valid)         | 61,503  | 0.7466        | 0.3665     | —            | 独立验证集评估                    |
| **标准信用评分卡 (OptBinning)**     | 测试集 (Test)          | 48,744  | —             | —          | **0.00131**  | 跨客群表现稳定 (PSI < 0.10)      |

> **群体稳定性指标（PSI）**：通常以 PSI < 0.10 作为客群分布基本稳定的参考阈值。评分卡总分与 LightGBM 预测概率在测试集上的 PSI 分别为 0.00131 和 0.00141，分布未见明显偏移。为防止数据穿越，特征筛选阶段仅使用训练集自身的质量与 IV 指标，不引入测试集进行 PSI 过滤；测试集 PSI 仅用于最终模型的稳定性验证。

### 2.2 样本分层审计账本 (`split_audit.csv`)

| 数据集                     | 样本量     | 样本占比    | 正常客户 (0) | 违约客户 (1) | 坏账率 (Bad Rate) |
|:----------------------- |:-------:|:-------:|:--------:|:--------:|:--------------:|
| 原始训练集 (Raw Train)       | 307,511 | 100.00% | 282,686  | 24,825   | 8.0729%        |
| **训练集 (Train)**         | 246,008 | 80.00%  | 226,148  | 19,860   | 8.0729%        |
| **验证集 (Valid)**         | 61,503  | 20.00%  | 56,538   | 4,965    | 8.0728%        |

---

## 3. 流水线执行架构

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
    [ 训练集: Train (80%) ]                               [ 验证集: Valid (20%) ]
    (仅用于拟合规则与参数学习)                               (独立封存，不参与任何规则 fit)
            │                                                     │
   ┌────────┴────────┐                                            │
   ▼                 ▼                                            │
01_eda.py    02_adversarial_validation.py                         │
(特征体检账本)  (LightGBM 探测跨期分布差异)                            │
   │                 │                                            │
   └────────┬────────┘                                            │
            ▼                                                     │
03_feature_engineering.py (统一特征工程) ◄────────────────────────┘
- 异常值修复与退休客群标记
- 覆盖 8 大风控维度业务衍生 (41 个业务衍生 + 1 个状态标记)
- 行业/职业分组基准均值与相对水平比率 (严格仅在训练集计算)
- 输出全量特征宽表: train_features_full, valid_features_full, test_features_full
            │
            ├─────────────────────────────────┐
            ▼                                 ▼
04_scorecard_modeling.py             05_lightgbm_modeling.py
(传统统计评分卡：LR)                 (机器学习提升树：LightGBM)
- 数据质量与 IV >= 0.02 漏斗粗筛      - 原生 category 类别特征适配
- 房屋同源去重与相关性贪心过滤         - 剔除同值率 > 95% 特征
- OptBinning 约束分箱 & WOE 编码     - Spearman 宽阈值去共线 (|r| >= 0.95)
- VIF <= 5 多重共线性消除             - 5折 CV 累计分裂增益精选 (Top 95%)
- 逐步回归 (AIC) & 系数负符号校验     - Optuna TPE 贝叶斯搜索浅树与强正则
- PDO=20, 600 基准分刻度映射          - 5折模型集成推理
            │                                 │
            └────────────────┬────────────────┘
                             ▼
                    [ 统一特征映射与模型验证 ]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
 [ 验证集 AUC / KS 独立评估 ]       [ 测试集总分 / 概率 PSI 跨客群稳定性 ]
```

---

## 4. 核心脚本解析

- **`baseline/00_stratified_split.py`**：
  按标签分层抽样（`random_state=123`），将 30.7 万条原始训练数据划分为 80% 训练集与 20% 验证集。校验主键唯一性、样本交集（为 0）与坏账率偏差（< 0.01%），输出审计文件 `split_audit.csv`。
- **`baseline/01_eda.py`**：
  分析训练集 120 余个字段的统计量、缺失率、主要值占比、单变量分箱、IV、JS 散度与 Gini 系数；处理房屋信息集中缺失与 `DAYS_EMPLOYED=365243` 异常值，输出原始特征画像表 `raw_features_summary.csv`。
- **`baseline/02_adversarial_validation.py`**：
  以“是否属于测试集”（`is_test`）为标签训练 LightGBM，通过浅树与限制叶节点样本量（`max_depth=5, num_leaves=16, min_child_samples=50`）检查两组样本的分布差异，保存特征重要性表至 `adversarial_validation_feature_importance.csv`。
- **`baseline/03_feature_engineering.py`（统一特征工程）**：
  处理异常值与退休客群标记（`FLAG_DAYS_EMPLOYED_ANOMALOUS`）；构造覆盖 8 大风控维度的 41 个业务衍生特征（共计 42 个新增衍生特征）；仅在训练集上聚合计算同行业与同职业的收入均值基准，并分别以无监督映射形式（`left join`）合并至验证集与测试集；生成并保存全量特征宽表（162 个候选特征，总宽表 164 列含 ID 与 TARGET）至 `data/processed/*_features_full.csv`，输出数据字典账本 `full_features_summary.csv`。
- **`baseline/04_scorecard_modeling.py`（标准信用评分卡）**：
  读取全量特征宽表。执行学历有序单调映射，通过数据质量粗筛、批量 IV 初筛（$\ge 0.02$）、房屋同源去重与全局 Spearman 秩相关贪心过滤；拟合 `OptBinning` 精细约束分箱（单箱 $\ge 5\%$、坏样本 $\ge 30$、数值单调与比率 U 型放宽）；通过迭代 VIF 诊断（$\le 5.0$）消除多重共线性，运行双向逐步回归（AIC 准则）并严格校验逻辑回归系数符号（剔除 6 个符号反转变量）；设定 PDO = 20、基准分 600 分，导出各分箱得分明细表 `final_scorecard_table.csv`（最终入模 25 个特征）并评估验证集指标与测试集 PSI。
- **`baseline/05_lightgbm_modeling.py`（LightGBM 集成建模）**：
  读取全量特征宽表。统一对齐类别变量原生 `category` 格式；剔除同值率 $> 95\%$ 字段与极端共线（$|r| \ge 0.95$）字段；运行 5 折交叉验证基准模型，按累计分裂增益（Gain Importance 前 95%）精选核心特征（入模 53 个特征）；使用 Optuna 运行 30 次 TPE 贝叶斯搜索优化浅树深度、叶子数与 L1/L2 正则化超参数；完成 5 折模型集成并评估验证集和测试集。

---

## 5. 核心特征分析

结合评分卡 IV 与 LightGBM 增益重要性（Gain Importance），违约区分度较高的特征主要集中在以下维度：

1. **外部征信评分**：
   `EXT_SOURCE_MEAN`、`EXT_SOURCE_MIN`、`EXT_SOURCE_3`、`EXT_SOURCE_2`。单变量 IV 均在 0.33 以上，在 LightGBM 中合计贡献了近 50% 的分裂增益。其中均值有助于平滑单家机构波动，最小值反映短板维度的信用记录。
2. **负债杠杆与偿付能力**：
   `GOODS_TO_CREDIT_RATIO`（商品价值与贷款额度比，反映抵押覆盖与超融情况）、`AMT_ANNUITY`（单期月供金额）、`ANNUITY_TO_CREDIT_RATIO`（月供与贷款额度比，反映还款期数与费率倒数）。
3. **稳定性与基本信息**：
   `DAYS_BIRTH`（年龄）、`DAYS_EMPLOYED`（工龄）、`EMPLOYED_TO_BIRTH_RATIO`（工龄占年龄比例，反映就业稳定程度）、`ORGANIZATION_TYPE` / `OCCUPATION_TYPE`（单位类型与职业）。
4. **行为变化与征信频次**：
   `DAYS_LAST_PHONE_CHANGE`（更换手机天数）、`TOTAL_NAN_COUNT`（原始信息缺失总项数，反映填报完整度）、`BUREAU_REQ_SHORT_TO_LONG`（短期与中长期征信查询次数比）。

---

## 6. 项目文件组织结构

```
home-credit-default-risk/
├── baseline/                               # 核心流水线脚本
│   ├── 00_stratified_split.py              # 1. 样本 8:2 分层抽样与物理隔离
│   ├── 01_eda.py                           # 2. 探索性数据分析与特征体检表构建
│   ├── 02_adversarial_validation.py        # 3. LightGBM 对抗性验证 (检验分布差异)
│   ├── 03_feature_engineering.py           # 4. 统一风控特征工程与全量特征宽表构建
│   ├── 04_scorecard_modeling.py            # 5. 标准信用评分卡建模 (IV/分箱/VIF/逐步回归/PDO)
│   └── 05_lightgbm_modeling.py             # 6. LightGBM 建模 (Gain精选/Optuna调优/测试推理)
│
├── data/                                   # 数据目录 (.gitignore 已忽略大文件)
│   ├── raw/                                # 原始数据集
│   │   ├── application_train.csv           # 原始训练集 (带标签, 30.7万条)
│   │   ├── application_test.csv            # 测试集 (未标注, 4.8万条)
│   │   └── HomeCredit_columns_description.csv # 字段描述说明表
│   ├── interim/                            # 物理隔离切分后的中间数据
│   │   ├── application_train_split.csv     # 训练集 (80%, 24.6万条)
│   │   └── application_valid.csv           # 验证集 (20%, 6.1万条)
│   └── processed/                          # 统一特征工程特征宽表
│       ├── train_features_full.csv         # 训练集全量特征宽表 (164 列)
│       ├── valid_features_full.csv         # 验证集全量特征宽表 (164 列)
│       └── test_features_full.csv          # 测试集全量特征宽表 (163 列)
│
├── outputs/                                # 实验输出、模型评估与审计报告
│   ├── validation/                         # 数据切分与验证检验
│   │   ├── split_audit.csv                 # 训练集与验证集分层切分审计报告
│   │   └── adversarial_validation_feature_importance.csv # 对抗验证特征重要性表
│   ├── eda/                                # 探索性分析全量字段体检账本
│   │   ├── raw_features_summary.csv        # 原始 120+ 字段 IV/缺失/分箱详细报告
│   │   └── full_features_summary.csv       # 全量候选特征字典与质量体检账本
│   └── models/                             # 模型文件与评估汇总
│       ├── final_scorecard_table.csv       # 标准评分卡各分箱得分明细表 (含PDO加减分)
│       ├── scorecard_eval_summary.csv      # 评分卡 AUC、KS 与测试集 PSI 汇总表
│       ├── lgb_feature_importance.csv      # LightGBM 增益特征重要性与累计 Gain 表
│       └── lgb_eval_summary.csv            # LightGBM 5折 OOF、验证集与测试集评估汇总
│
├── pyproject.toml                          # 项目依赖与配置 (uv 规范)
├── uv.lock                                 # 依赖版本锁定文件
├── .python-version                         # Python 解释器版本 (3.12)
├── .gitignore                              # Git 忽略配置
└── README.md                               # 项目说明文档
```

---

## 7. 快速开始与运行指南

### 7.1 环境配置（基于 uv）

本项目使用 [uv](https://github.com/astral-sh/uv) 管理虚拟环境与依赖。Python 版本（3.12）及第三方库均已记录在 `pyproject.toml` 与 `uv.lock` 中。

#### 1. 安装 uv
根据操作系统安装 `uv`（已安装可跳过）：
- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **macOS / Linux**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **通过 pip 安装（备选）**:
  ```bash
  pip install uv
  ```

#### 2. 同步并安装依赖
克隆仓库后，在项目根目录下执行 `uv sync`。`uv` 会自动读取 `.python-version` 创建 `.venv`，并安装锁定的全部依赖：

```bash
# 克隆代码仓库
git clone https://github.com/gorgeous211/home-credit-default-risk.git
cd home-credit-default-risk

# 同步环境并安装依赖
uv sync
```

> **环境与硬件加速提示**：
> 1. **执行方式**：使用 `uv run python <script.py>` 会直接在项目虚拟环境中运行脚本，无需手动激活环境。
> 2. **NVIDIA GPU (CUDA) 加速**：LightGBM 默认使用 CPU 多线程运算。如需使用 CUDA 加速，可根据本地 CUDA 环境安装支持 GPU 的 LightGBM 构建版本。
> 3. **Windows 动态链接库说明**：如遇 OpenBLAS / LAPACK 动态库冲突，脚本会自动优先读取虚拟环境下的 `Library/bin` 路径。

---

### 7.2 数据准备

1. 从 Kaggle 下载赛题数据：[Home Credit Default Risk Data](https://www.kaggle.com/competitions/home-credit-default-risk/data)
2. 解压后将 CSV 文件放置在 `data/raw/` 目录（包含 `application_train.csv`、`application_test.csv` 与 `HomeCredit_columns_description.csv`）。

---

### 7.3 流水线运行步骤

各脚本解耦设计，建议按编号顺序依次执行：

```bash
# 步骤 0: 样本分层切分与物理隔离 (生成 data/interim/* 与 split_audit.csv)
uv run python baseline/00_stratified_split.py

# 步骤 1: 特征分析与指标报告生成 (输出 outputs/eda/raw_features_summary.csv)
uv run python baseline/01_eda.py

# 步骤 2: 对抗性验证检验分布差异 (输出 adversarial_validation_feature_importance.csv)
uv run python baseline/02_adversarial_validation.py

# 步骤 3: 全量风控特征工程与特征宽表构建 (生成 data/processed/* 与 outputs/eda/full_features_summary.csv)
uv run python baseline/03_feature_engineering.py

# 步骤 4: 拟合标准信用评分卡与 PDO 刻度计算 (输出 final_scorecard_table.csv 与评估指标)
uv run python baseline/04_scorecard_modeling.py

# 步骤 5: 训练 LightGBM 模型与 Optuna 贝叶斯调参
uv run python baseline/05_lightgbm_modeling.py
```

---

## 8. 相关工具与参考

- **环境与依赖管理**：[uv](https://github.com/astral-sh/uv)
- **赛题来源**：[Kaggle - Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
- **评分卡工具**：[OptBinning: Optimal Binning for Credit Risk](https://github.com/guillermo-navas-palencia/optbinning)
- **特征工程工具**：[Toad: Industrial Credit Risk Scorecard Development](https://github.com/amphibian-dev/toad)
- **梯度提升决策树**：[LightGBM](https://github.com/microsoft/LightGBM)
- **超参数优化框架**：[Optuna](https://optuna.org/)
