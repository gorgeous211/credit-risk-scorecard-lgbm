# 训练集与测试集分布检验：基于 LightGBM 的对抗验证

# %%
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# 文件路径
DATA_PATH = Path.cwd() / "data"
OUTPUTS_PATH = Path.cwd() / "outputs"
VALIDATION_OUTPUT_PATH = OUTPUTS_PATH / "validation"
VALIDATION_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 主申请表
train = pd.read_csv(DATA_PATH / 'raw' / 'application_train.csv').drop(
    columns=['SK_ID_CURR', 'TARGET']
)
test = pd.read_csv(DATA_PATH / 'raw' / 'application_test.csv').drop(columns='SK_ID_CURR')

# %%
# 标记训练集和测试集
train['is_test'] = 0
test['is_test'] = 1

# 合并训练集和测试集
application = pd.concat([train, test], ignore_index=True)

# 将类别特征转换为 category 类型，适配 LightGBM 原生接口
obj_cols = application.select_dtypes(include='object').columns
for col in obj_cols:
    application[col] = application[col].astype('category')

# 切分特征 X 与标签 y
X = application.drop(columns='is_test')
y = application['is_test']

# 设置分层 5 折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
oof_preds = np.zeros(len(X))
feature_importance = np.zeros(X.shape[1])

# 对抗验证参数设计：
# 核心是捕捉宏观分布差异并防止过拟合，无需网格调参。
# - min_child_samples：总样本量约 35 万，取 50 避免叶节点拟合局部噪声。
# - max_depth / num_leaves：浅树结构（depth=5, leaves=16），控制树复杂度。
# - subsample / colsample_bytree：设为 0.8，避免少数特征主导分裂。
# - class_weight：保持 None，使预测概率反映实际样本密度比。
params = { 
    # 复杂度控制
    'max_depth': 5,             # 树最大深度
    'num_leaves': 16,           # 叶子节点数（远小于 2^5=32，控制树复杂度）
    'min_child_samples': 50,    # 单叶子节点最小样本数，防止局部过拟合
                                
    # 采样与扰动
    'subsample': 0.8,           # 样本行采样比例
    'subsample_freq': 1,        # 采样频率
    'colsample_bytree': 0.8,    # 特征列采样比例，促进探索次要特征
    
    # 正则化与迭代
    'reg_alpha': 0.5,           # L1 正则化
    'reg_lambda': 1.0,          # L2 正则化
    'learning_rate': 0.05,
    'n_estimators': 1000,       # 配合早停使用

    # 目标与评估
    'objective': 'binary',
    'metric': 'auc',
    'class_weight': None,       # 测试集占比 13.7%，保持原始比例可反映实际密度比
    'random_state': 123,
    'importance_type': 'gain',  # 按信息增益计算特征重要性
    'n_jobs': -1,
}

callbacks = [lgb.early_stopping(
    stopping_rounds=50,         # 连续 50 轮未提升则早停
    verbose=True)]

for train_idx, val_idx in skf.split(X, y):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['tran', 'valid'],
        eval_metric='auc',
        callbacks=callbacks
    )
    
    oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
    feature_importance += clf.feature_importances_ / skf.n_splits

overall_auc = round(roc_auc_score(y, oof_preds), 2)
# 0.89

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

importance_df.head(10)
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

# 结论说明：
# 对抗验证 AUC 约为 0.89，表明训练集与测试集存在可识别的特征分布差异。
# AMT_CREDIT、AMT_ANNUITY 等金额特征排名靠前，说明金额类变量在训练集与测试集间有较明显分布偏离。

# %%
# 保存对抗验证特征重要性表至 outputs/validation
importance_df.to_csv(VALIDATION_OUTPUT_PATH / 'adversarial_validation_feature_importance.csv', index=False)
