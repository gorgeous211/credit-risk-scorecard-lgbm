# Home Credit Default Risk: 传统统计评分卡（LR）与机器学习提升树（LightGBM）工业级信贷风控基准系统

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/) [![Package Manager](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv) [![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/) [![Scorecard](https://img.shields.io/badge/Model-Scorecard%20(OptBinning)-orange.svg)](https://gnpalencia.org/optbinning/) [![Validation](https://img.shields.io/badge/Validation-Zero%20Data%20Leakage-red.svg)](#) [![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 1. 方案定位与核心价值

在信贷风控与零售金融场景中，核心挑战在于平衡**极端样本不平衡（违约率仅 8.07%）**、**高维稀疏异构数据**与**严苛的监管合规审计要求（要求规则完全透明、逻辑单调且零数据穿越）**。

本项目基于真实海量零售信贷场景（Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)），打破过程式堆砌与黑盒刷分范式，构建了一套严格遵循现代商业银行合规审计标准与先进机器学习工程实践的**工业级信贷风控基准系统**。

本项目确立了**三大不可替代的核心工程价值**：

### 核心价值一：严密完备的物理零穿越合规防线（Zero Data Leakage）
- **根除事前信息渗透**：在探索性分析、特征清洗、单变量分箱及 IV 筛选前，率先按 8:2 完成物理隔离分层抽样，将 20% 的**独立验证集（Valid，61,503 条）完全闭卷封存**。
- **纯单向映射规范**：所有异常清洗规则、缺失模式识别、OptBinning 最优分箱、VIF 多重共线性消除及行业均值基准，**严格 100% 仅在训练集上学习与拟合**，验证集与未知测试集仅执行单向转换（Transform / Merge）。
- **对抗性验证量化分布漂移**：部署浅层强正则 LightGBM 对抗性验证，提前探测并量化训练集与测试集的跨期时序漂移特征，确立全流程稳定性监控基准。

### 核心价值二：驱动业务实质的 8 维机理级特征工程体系（42 个衍生特征）
- **摆脱绝对数值与周期依赖**：围绕信贷风控核心机理（Domain Knowledge），自发构造 42 个高判别力衍生特征，涵盖**还款能力与负债压力**、**贷款条件与抵押覆盖**、**外部征信高阶交互**、**生命周期稳定性**、**资产完备度**、**身份真实性与圈层传染**、**短期多头借贷激增**以及**行业/职业分群基准**。
- **卓越的判别主导地位**：衍生特征直接确立了后续模型的判别力基石。在 LightGBM 特征分裂增益榜单中，衍生变量在前 10 名中占据 7 席（如 `EXT_SOURCE_MEAN` 单项贡献达 27.5%、`EXT_SOURCE_MIN` 贡献 7.2%、`ESTIMATED_TENURE` 贡献 3.4%）。

### 核心价值三：“白盒可解释”与“算法性能上限”双轨协同架构（Dual-Track Benchmark Framework）
- **合规生产轨（标准信用评分卡 Logistic Regression）**：经数据质量漏斗、单调性 WOE 约束分箱、VIF $\le 5.0$ 诊断、逐步回归（AIC 准则）以及严格的**系数符号反转校验（严防风险倒挂）**，采用商业银行标杆的 PDO=20（基准分 600 分）输出纯整数加减分明细表。在保证 100% 逻辑透明与可解释性的前提下，实现验证集 AUC **0.7466**、KS **0.3665**，测试集总分跨客群 PSI 仅为 **0.00131**，完全满足巴塞尔协议强监管要求。
- **算法天花板轨（LightGBM 5-Fold 集成提升树）**：保留连续数值形态与高缺失分支，通过 5 折分层交叉验证 OOF、累计增益（Gain Importance Top 95%）精选与 Optuna TPE 贝叶斯搜索深度与强正则超参数，实现验证集 AUC **0.7658**、KS **0.3982**，测试集概率 PSI 仅为 **0.00141**，确立了当前场景下的算法判别力上限。

---

## 2. 实验结果与评估指标

模型评估严格基于独立物理封存的**验证集（Valid）**与**未标注测试集（Test）**展开，全方位验证判别精度与跨分布泛化稳定性：

### 2.1 工业双轨基准性能对比

| 模型架构 | 业务定位与核心机制 | 评估数据集 | 样本量 | 区分度 (ROC-AUC) | 鉴别力 (KS) | 跨客群稳定性 (PSI) | 核心特性说明 |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **标准信用评分卡**<br>*(OptBinning + LR)* | **白盒可解释生产轨**<br>单调WOE + VIF诊断 + 逐步回归 + 符号反转校验 + PDO=20整数映射 | 训练集 (Train)<br>验证集 (Valid)<br>测试集 (Test) | 246,008<br>61,503<br>48,744 | 0.7523<br>**0.7466**<br>— | 0.3747<br>**0.3665**<br>— | 基准 (0.00000)<br>—<br>**0.00131** | 100% 白盒可追溯<br>满足巴塞尔协议强监管要求<br>无业务倒挂与数据穿越 |
| **LightGBM 集成提升树**<br>*(5折CV + Optuna)* | **算法性能上限探索轨**<br>Gain精选 (Top 95%) + TPE贝叶斯调优 + 5折模型集成推理 | 训练集 (5-Fold OOF)<br>验证集 (Valid)<br>测试集 (Test) | 246,008<br>61,503<br>48,744 | 0.7676<br>**0.7658**<br>— | 0.3991<br>**0.3982**<br>— | 基准 (0.00000)<br>—<br>**0.00141** | 强大非线性交互捕获能力<br>5折集成平滑预测方差<br>确立风控判别力上限 |

### 2.2 核心成果与泛化稳定性深度剖析

1. **白盒可解释轨与算法上限轨的高效协同**：
   - 传统评分卡以 25 个精选特征实现了 **0.7466** 的验证集 AUC 和 **0.3665** 的 KS，在维持极高业务可读性与单调性约束的同时，仅以极小性能代价换取了完备的合规穿透能力；
   - LightGBM 则在 53 个高增益核心特征上达到 **0.7658** 的 AUC 和 **0.3982** 的 KS，为风控反欺诈、额度授信与差异化定价提供了极具区分度的概率参考基准。
2. **极度稳健的跨客群抗漂移能力（PSI 远优于行业基线）**：
   - 商业银行与消费金融通常以 $\text{PSI} < 0.10$ 作为客群基本稳定的判定标准。本项目评分卡总分测试集 PSI 为 **0.00131**，LightGBM 预测概率测试集 PSI 为 **0.00141**，展现出卓越的抗周期分布漂移能力。
   - 这一结果直接佐证了：本项目的特征衍生（无量纲比率、动态加权、分群相对水平）成功剔除了原始金额对宏观周期的依赖，实现了模型在未知时序客群上的高保真泛化。

### 2.3 样本分层审计账本 (`split_audit.csv`)

| 数据集 | 样本量 | 样本占比 | 正常客户 (0) | 违约客户 (1) | 坏账率 (Bad Rate) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 原始训练集 (Raw Train) | 307,511 | 100.00% | 282,686 | 24,825 | 8.0729% |
| **训练集 (Train)** | 246,008 | 80.00% | 226,148 | 19,860 | 8.0729% |
| **验证集 (Valid)** | 61,503 | 20.00% | 56,538 | 4,965 | 8.0728% |

---

## 3. 流水线执行架构

代码位于 `baseline/` 目录，按流水线编号严格解耦执行：

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
   (规则拟合与参数学习唯一来源)                              (独立封存，严禁参与任何规则 fit)
            │                                                     │
   ┌────────┴────────┐                                            │
   ▼                 ▼                                            │
01_eda.py    02_adversarial_validation.py                         │
(特征体检账本)  (LightGBM 探测跨期宏观漂移)                            │
   │                 │                                            │
   └────────┬────────┘                                            │
            ▼                                                     │
03_feature_engineering.py (统一特征工程) ◄────────────────────────┘
- 统一异常值修复与退休客群标记
- 覆盖 8 大风控维度全量业务衍生 (41 个业务衍生 + 1 个状态标记)
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

## 4. 模块化工程实现解析

- **`baseline/00_stratified_split.py`（物理隔离与数据纯度保障）**：
  基于目标变量执行严格分层抽样（`random_state=123`），锁定 80% 训练集与 20% 验证集。自动化校验样本主键唯一性、集合无交集（样本交叠为 0）与违约率一致性（偏差 < 0.0001%），生成审计凭证 `split_audit.csv`。
- **`baseline/01_eda.py`（全量字段体检账本）**：
  构建 120 余个字段的多维质量画像（缺失率、最大同值率、分位数、最优单变量分箱、IV、JS 散度及 Gini 系数）；锁定房屋信息同源缺失机制与 `DAYS_EMPLOYED=365243` 异常编码，输出特征画像表 `raw_features_summary.csv`。
- **`baseline/02_adversarial_validation.py`（对抗性验证与时序漂移探测）**：
  构建以“是否属于测试集”（`is_test`）为目标的对抗分类器，通过强正则浅树（`max_depth=5, num_leaves=16`）量化两组样本分布差异，生成特征漂移重要性表 `adversarial_validation_feature_importance.csv`，指导后续特征工程与抗漂移设计。
- **`baseline/03_feature_engineering.py`（统一特征工程与特征宽表构建）**：
  异常值联动清洗与退休客群显式标记；向量化构造 8 大风控维度的 41 个业务衍生特征（合计 42 个新增衍生特征）；严格仅在训练集上聚合计算行业/职业基准均值，以单向映射（`left join`）合并至各数据集；构建并落盘全量特征宽表（162 个候选特征，总表 164 列含 ID 与 TARGET）至 `data/processed/*_features_full.csv`，输出数据字典账本 `full_features_summary.csv`。
- **`baseline/04_scorecard_modeling.py`（标准信用评分卡建模）**：
  直接对接全量特征宽表。执行学历有序单调映射，实施数据质量初筛、IV 初筛（$\ge 0.02$）、房屋同源去重与全局 Spearman 秩相关贪心过滤；通过 `OptBinning` 拟合精细约束分箱（单箱 $\ge 5\%$、坏样本 $\ge 30$、严格单调与比率 U 型放宽）；通过迭代 VIF 诊断（$\le 5.0$）消除多重共线性，运行双向逐步回归（AIC 准则）并严格校验逻辑回归系数符号（剔除 6 个符号反转变量以防风险倒挂）；设定基准分 600 分、PDO = 20，导出各分箱整数得分明细表 `final_scorecard_table.csv`（最终入模 25 个特征）并评估验证集指标与测试集 PSI。
- **`baseline/05_lightgbm_modeling.py`（LightGBM 集成建模与超参调优）**：
  直接对接全量特征宽表。统一对齐类别变量原生 `category` 维度；剔除同值率 $> 95\%$ 字段与极端共线（$|r| \ge 0.95$）字段；运行 5 折交叉验证基准模型，按累计分裂增益（Gain Importance 前 95%）精选核心特征（入模 53 个特征）；使用 Optuna 运行 30 次 TPE 贝叶斯搜索优化浅树深度、叶子数与 L1/L2 正则化超参数；完成 5 折模型集成并评估验证集和测试集。

---

## 5. 风控机理与特征增益洞察

结合评分卡单变量 IV 矩阵与 LightGBM 增益贡献排行榜（Gain Importance），本系统的判别力集中体现于四大金融风控机理：

1. **外部多机构征信综合体系（全表最强判别力）**：
   `EXT_SOURCE_MEAN`、`EXT_SOURCE_MIN`、`EXT_SOURCE_3`、`EXT_SOURCE_2`。单变量 IV 均突破 0.33，在 LightGBM 中单项贡献超 50% 分裂增益。均值有效平滑单家机构波动，最小值精准捕捉“木桶短板”失信效应，动态加权则稳固了非空客群的信用画像。
2. **负债压力与授信敞口代理（还款能力核心）**：
   `GOODS_TO_CREDIT_RATIO`（商品价值与贷款额度比，反映抵押覆盖度与超融敞口）、`AMT_ANNUITY`（单期月供金额）、`ANNUITY_TO_CREDIT_RATIO`（单期月供与贷款额度比，表征隐性融资成本与期限倒数）、`DOWN_PAYMENT_PROXY`（超融差额越负违约率显著上升）。
3. **生命周期稳定性与行为错配（履约意愿基石）**：
   `DAYS_BIRTH`（年龄）、`DAYS_EMPLOYED`（工龄）、`EMPLOYED_TO_BIRTH_RATIO`（工龄占年龄比，反映终身职业连续性）、`AGE_AT_EMPLOYMENT`（当前任职入职年龄）、`ORGANIZATION_TYPE` / `OCCUPATION_TYPE`（单位性质与职业群体分位）。
4. **信息透明度与短周期多头激增（流动性危机预警）**：
   `TOTAL_NAN_COUNT`（原始资料填报缺失总项数，真实反映客户透明度与配合度，入选评分卡核心特征）、`DAYS_LAST_PHONE_CHANGE`（更换手机天数）、`BUREAU_REQ_SHORT_TO_LONG`（近 1 个月短期激增比，灵敏捕捉资金链断裂前夕多头借贷先兆）。

---

## 6. 项目工程结构

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

## 7. 快速复现与运行指南

### 7.1 环境配置（基于 uv）

本项目使用现代高性能 Python 包管理器 [uv](https://github.com/astral-sh/uv) 管理虚拟环境与依赖。Python 版本（3.12）及第三方库均已精确锁定在 `pyproject.toml` 与 `uv.lock` 中。

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
克隆仓库后，在项目根目录下执行 `uv sync`。`uv` 会自动读取 `.python-version` 创建 `.venv`，并秒级安装锁定的全部依赖：

```bash
# 克隆代码仓库
git clone https://github.com/gorgeous211/home-credit-default-risk.git
cd home-credit-default-risk

# 极速同步环境并安装依赖
uv sync
```

> **运行提示**：
> 1. **执行方式**：通过 `uv run python <script.py>` 会自动绑定项目专属虚拟环境执行，无需手动切换或激活虚拟环境。
> 2. **硬件加速**：LightGBM 默认采用 CPU 高效并行多线程计算；如需启用 GPU 加速，可根据本地驱动安装对应构建版本。

---

### 7.2 数据准备

1. 从 Kaggle 下载赛题官方数据：[Home Credit Default Risk Data](https://www.kaggle.com/competitions/home-credit-default-risk/data)
2. 解压后将 CSV 文件放置在 `data/raw/` 目录（包含 `application_train.csv`、`application_test.csv` 与 `HomeCredit_columns_description.csv`）。

---

### 7.3 流水线复现步骤

流水线脚本按模块解耦设计，严格按编号顺序依次执行：

```bash
# 步骤 0: 样本分层切分与物理隔离 (生成 data/interim/* 与 split_audit.csv)
uv run python baseline/00_stratified_split.py

# 步骤 1: 全量字段探索性分析与体检表构建 (输出 outputs/eda/raw_features_summary.csv)
uv run python baseline/01_eda.py

# 步骤 2: 对抗性验证探测时序客群漂移 (输出 adversarial_validation_feature_importance.csv)
uv run python baseline/02_adversarial_validation.py

# 步骤 3: 8 维风控特征工程与全量宽表构建 (生成 data/processed/* 与 outputs/eda/full_features_summary.csv)
uv run python baseline/03_feature_engineering.py

# 步骤 4: 标准信用评分卡建模与 PDO 刻度映射 (输出 final_scorecard_table.csv 与 scorecard_eval_summary.csv)
uv run python baseline/04_scorecard_modeling.py

# 步骤 5: LightGBM 5折集成训练与 Optuna 贝叶斯搜索 (输出 lgb_feature_importance.csv 与 lgb_eval_summary.csv)
uv run python baseline/05_lightgbm_modeling.py
```

---

## 8. 技术栈与参考资料

- **环境与依赖调度**：[uv - Astral](https://github.com/astral-sh/uv)
- **数据源与竞赛主页**：[Kaggle - Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
- **风控最优分箱与评分卡框架**：[OptBinning: Optimal Binning for Credit Risk](https://github.com/guillermo-navas-palencia/optbinning)
- **工业风控建模工具链**：[Toad: Industrial Credit Risk Scorecard Development](https://github.com/amphibian-dev/toad)
- **梯度提升决策树引擎**：[LightGBM](https://github.com/microsoft/LightGBM)
- **贝叶斯超参数优化框架**：[Optuna](https://optuna.org/)
