# 模型组 `model/`

负责网络结构、训练流程、评估与指标产出。

## 文件

| 文件 | 职责 |
|------|------|
| `model.py` | ResNet-50（IMAGENET1K_V2 预训练，替换 FC 层）；`get_model(num_classes, pretrained)` |
| `train.py` | 训练循环，保存最优检查点 |
| `eval.py` | 测试集评估，产出报告与图表 |

## 用法（从项目根目录运行）

```bash
# 训练 → 保存 best_model.pth + training_history.json（按验证集最优保存）
python model/train.py

# 评估 → classification_report.txt + confusion_matrix.png + training_curves.png
python model/eval.py
```

- 超参：Adam，lr=1e-4，10 epochs，batch 32（见根目录 `config.py`）
- 训练时长：~26 min（GPU）
- 数据来自数据组：`from data.data import get_dataloaders`

## 当前指标

| 指标 | 数值 |
|------|------|
| 最佳验证准确率 | 99.9%（epoch 8） |
| 测试准确率 | 99.90%（1000 张仅 1 张错：Forza→Apex） |

## 注意点

- **产物路径相对项目根**（`best_model.pth` / `training_history.json` 等），必须从根目录运行
- 后台跑 `python model/train.py 2>&1 | Tee-Object` 时 Python stdout 块缓冲，日志要进程退出才刷新；想实时看加 `$env:PYTHONUNBUFFERED=1`；判断是否真在跑用 `nvidia-smi` 看 GPU 占用，别看空日志
- GPU 是 RTX 50 系（Blackwell, sm_120），torch 须 cu128，否则 `cuda.is_available()` 为 True 但 kernel 报错
- 改类别数需同步 `config.NUM_CLASSES` + `config.CLASS_NAMES`，且必须重训（FC 层维度变）

## 依赖

`torch` `torchvision` `scikit-learn` `matplotlib` `seaborn` `tqdm`
