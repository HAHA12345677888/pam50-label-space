# Best Model — Clinical + mRNA + Methylation (LogReg C=0.01)

这个文件夹保存当前验证集上表现最好的模型：

- **模态组合**: Clinical (69d) + mRNA (412d) + Methylation (232d) = **713 维**
- **模型**: Logistic Regression, C=0.01, max_iter=4000, 多分类 (6 classes)
- **评估结果** (data/val.npz, 205 样本):
  - **Val Accuracy: 79.0%**
  - **Test Accuracy: 73.8%** (test 集从未参与调参)
  - 5-fold CV (train 集): **74.3% ± 2.8%**
- **对照**: 原深度 Transformer (Tier 1) 只有 60.0%

## 为什么选这三个模态？

5-fold 交叉验证扫描结果（LogReg C=0.01）：

| 模态组合 | CV acc |
|---|---|
| mRNA 单模态 | 73.8% |
| Clinical + mRNA | 74.5% |
| **Clinical + mRNA + Methyl** | **75.3%** |
| Clinical + mRNA + Mutat | 72.6% |
| 全部 5 模态 | 72.4% |

CNA 和 Mutation 模态加入后反而降低准确率（在 958 训练样本的小数据上引入噪声），因此被排除。

## 文件说明

- `train_best.py` — 完整训练脚本（数据加载 → 模态选择 → 标准化 → 训练 → 评估）
- `predict.py` — 预测脚本（加载模型，对 .npz 数据预测亚型）
- `best_model.joblib` — 训练好的模型 + 标准化器（pickle 格式）
- `evaluation_report.txt` — 评估报告（val 集）

## 使用方法

```bash
# 训练（重新训练并评估 val + test）
python train_best.py

# 预测（用已保存的模型预测新数据）
python predict.py --data ../data/test.npz --model best_model.joblib
```

依赖：`scikit-learn`, `numpy`, `pandas`

## 下一步建议（往 80%+）

1. **模型集成**: LogReg + GBDT + RF 软投票（此前 val 达 76.6%，若组合 Clinical+mRNA+Methyl 可能更高）
2. **特征工程**: mRNA 基因模块 eigengene（soft_module_mask.npz 里的 20 个模块）作为额外特征
3. **阈值/校准**: 用验证集校准概率，找最优决策阈值
4. **深度模型重构**: 若仍需深度模型发论文，用 LogReg 作为 baseline，深度模型改为小参数量 + 交叉验证评估
