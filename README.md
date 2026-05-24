# Screenshot to Game Classifier

识别一张截图属于 10 款热门游戏中的哪一款 —— 既支持直接上传游戏截图，也支持拍一张别人手机/显示器屏幕的照片，自动检测屏幕边框、矫正透视后再分类。

Machine Learning 2 Final Project。

## Features

- **直接截图分类** — 上传游戏截图，返回 Top-3 预测及置信度
- **拍屏模式** — 拍实拍手机/显示器照片，自动识别屏幕、矫正透视后分类
- **屏幕识别流程**：Mobile SAM（语义分割，主策略）→ 亮度百分位 → Otsu → Canny 边缘（兜底）
- **数据采集工具** — 从 YouTube 下载游戏视频、按时间区间抽帧、感知哈希去重

## 当前本地数据集类别（17 类合并集）

经典 HF 10 类 + 自采 YouTube 10 类，按重叠游戏去重合并为 **17 类**：

ARCRaiders · Among Us · Apex Legends · CounterStrike2 · Fortnite · Forza Horizon · Free Fire · Genshin Impact · God of War · LeagueOfLegends · MarvelRivals · Minecraft · Roblox · RocketLeague · Subnautica2 · Terraria · Valorant

> Fortnite / Minecraft / Genshin Impact 两套数据源都有，已合并去重。

当前默认本地数据目录是 `dataset_combined/`（由 `data/build_combined_dataset.py` 从 `dataset_hf/` + `dataset_youtube_hq/` 硬链接合并而成，共 20,000 张；训练时每类均衡到 1,000 张）。

## 模型效果（17 类合并集）

| 项目 | 数值 |
|------|------|
| 架构 | ResNet-50（微调，替换 FC 层 → 17 类） |
| 预训练权重 | IMAGENET1K_V2 |
| 训练 / 验证 / 测试 | 13,600 / 1,700 / 1,700 张（17 类各均衡 1,000，分层 80/10/10） |
| 最佳验证准确率 | **99.65%**（epoch 4，提前停止） |
| 测试准确率 | **99.53%**（1,700 张） |
| 训练 | 4 epoch 即饱和（RTX 5090，约 1 min/epoch） |

> 仅 4 个 epoch 验证准确率即达 99.65%，遂提前停止。每类测试准确率 98–100%。

### 几何处理（关键设计）

输入统一走 **Letterbox**（等比缩放 + 居中补黑边到 224×224），**训练 / 评估 / 演示推理三方共用同一套几何**。
相比把 16:9 硬压成正方形（失真）或 Resize+CenterCrop（丢边缘 HUD/小地图），letterbox 不失真也不丢信息，几何一致后测试准确率从 99.40% 提升到 99.90%。

### 数据增强（仅训练集）

`LetterboxResize` → `RandomAffine(scale 0.85–1.0, translate 0.05)` → `RandomHorizontalFlip` · `RandomRotation(±15°)` · `RandomPerspective(0.4)` · `ColorJitter` · `RandomGrayscale` · `GaussianBlur` · `RandomErasing`

透视/旋转/模糊等增强专门针对"拍手机屏幕"场景，使模型对倾斜、变形、失焦的游戏画面更鲁棒。

## 项目结构

项目按小组分工拆为三个子包，`config.py` 为三组共用契约，留在根目录：

```
├── config.py              # 【共享】超参数、类别名称、路径常量；默认 DATASET_DIR=dataset_combined
├── data/                  # 【数据组】
│   ├── data.py            #   HF / 本地数据加载、分层切分、平衡采样、数据增强
│   ├── export_hf_dataset.py # HF baseline 导出成本地 ImageFolder
│   ├── collect_data.py    #   YouTube 视频下载 + 时间区间抽帧
│   ├── clean_data.py      #   坏图 / 低质量 / 重复帧清洗
│   ├── report_data.py     #   类别分布、尺寸分布、切分计划统计
│   └── HANDOFF.md         #   数据组给模型组的交接说明
├── model/                 # 【模型组】
│   ├── model.py           #   ResNet-50（替换 FC 层）
│   ├── train.py           #   训练循环，保存最优检查点
│   └── eval.py            #   测试评估、混淆矩阵、训练曲线
├── demo/                  # 【Demo 组】
│   ├── app.py             #   Gradio 演示界面（前端）
│   └── screen_crop.py     #   手机屏幕检测 + 透视矫正
├── requirements.txt
└── best_model.pth         # 训练后生成（不纳入 git）
```

> **运行约定**：所有脚本一律**从项目根目录**执行（如 `python model/train.py`）。
> 脚本会自动把根目录加进 `sys.path` 解析 `import config`；`best_model.pth` /
> `examples/` / `mobile_sam.pt` / `dataset/` 等均为相对根目录的路径，换目录运行会找不到。

### 小组分工

| 子包 | 负责人 | 职责范围 |
|------|--------|----------|
| `data/` | 数据组 | YouTube 采集、质量清洗、数据报告、本地/HF DataLoader、增强策略 |
| `model/` | 模型组 | 网络结构、训练流程、评估与指标产出 |
| `demo/` | Demo 组 | Gradio 前端、拍屏检测与透视矫正 |
| `config.py` | 三组共管 | 改动需协商（超参/类别名/路径，三组都依赖） |

## 安装

```bash
pip install -r requirements.txt

# 拍屏模式推荐：启用 Mobile SAM 屏幕分割
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

> **GPU 说明**：若用 RTX 50 系（Blackwell, sm_120），torch 须装 cu128 版本（nightly），cu121/cu124 编译产物不含该架构 kernel，`cuda.is_available()` 返回 True 但实际 kernel 会报错。

---

## 启动前端（演示界面）

```bash
python demo/app.py
```

- 启动后本地访问 **http://localhost:7860**
- `app.py` 中 `demo.launch(share=True)` 会同时打印一个公网 `*.gradio.live` 链接，可分享给他人临时访问
- 界面提供两种模式（radio 切换）：**直接截图分类** 与 **拍屏模式**
- 首次进入拍屏模式时自动下载 Mobile SAM 权重（~40 MB）；国内若直连 huggingface.co 卡住，可先手动下载：

```bash
curl.exe -L https://hf-mirror.com/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt -o mobile_sam.pt
```

> 注意：`app.py` 推理预处理复用 `data.py` 的 `LetterboxResize`，与训练几何严格一致 —— 改动任一处预处理须三处（`train_tf` / `eval_tf` / `app._transform`）同步。

---

## 下载数据（采集训练/测试数据）

### Hugging Face baseline

如需使用之前的公开 baseline 数据集，可直接下载/导出到本地：

```bash
python data/export_hf_dataset.py --output-dir dataset_hf
python data/report_data.py --dataset-dir dataset_hf --output dataset_hf_report.json
```

本地导出后结构为 `dataset_hf/<GameName>/hf_XXXXX.png`，共 10 类、每类 1,000 张。当前的 HF manifest 已归档到 `tobeclean/metadata/dataset_hf_metadata/hf_manifest.csv`。注意它不包含 Valorant、League of Legends、Rocket League；最终训练默认使用更新后的自采 YouTube 数据集 `dataset_youtube_hq/`。

`collect_data.py` 从 YouTube 下载游戏视频，按时间区间抽帧、感知哈希去重，输出为
`dataset/<GameName>/frame_XXXXX.png`，与 `torchvision.datasets.ImageFolder` 格式兼容。
同时会追加 `dataset/collection_manifest.csv`，记录图片路径、类别、来源 URL、视频时间点和 pHash。
理想最终数据应来自真实视频帧。当前 `collect_data.py` 会优先使用 YouTube Android/Android VR client 拉真实 MP4/video stream；若 YouTube 被 PO token / SABR 限制挡住，会跳过该视频，不会默认使用低清 storyboard。Storyboard fallback 仅用于临时 debug，需要显式加 `--allow-storyboard`。

### 前置：YouTube 解析依赖（必需）

YouTube 现在用 JS "n-challenge" 门控视频格式，yt-dlp 需要**两样东西**才能下载，缺一即
`n challenge solving failed` / `No supported JavaScript runtime`：

1. **JS runtime**（deno）：

   ```bash
   winget install denoland.deno
   ```

   装完即可 —— `collect_data.py` 会**自动探测** winget/scoop 安装的 deno 并注入 PATH（`ensure_deno_on_path()`），无需手动设环境变量。若未找到会打印安装提示。

2. **EJS 远程挑战求解脚本**：已固化在 `collect_data.py`（`--remote-components ejs:github`），首次运行自动从 GitHub 拉取，无需手动操作。

### 用法

```bash
# 指定单个 YouTube 链接
python data/collect_data.py --game "Minecraft" --url "https://youtu.be/XXX"

# 只截取视频的某个时间区间（支持 SS / MM:SS / HH:MM:SS）
python data/collect_data.py --game "Minecraft" --url "https://youtu.be/XXX" --start-time 1:30 --end-time 5:00

# YouTube 搜索（自动下载前 N 个结果）
python data/collect_data.py --game "Fortnite" --search "Fortnite gameplay 2024 no commentary" --max-videos 3

# 从 URL 文件批量下载（每行一个链接）
python data/collect_data.py --game "Genshin Impact" --url-file urls.txt

# 实例：下载这个视频（视频较长，需调大 --max-duration 否则被时长过滤）
python data/collect_data.py --game "LeagueOfLegends" --url "https://www.youtube.com/watch?v=zKaRQUzTtvM" --max-duration 1000 --fps 1 --max-frames 300
```

> 命令均写成单行 —— 本项目在 PowerShell 下运行，PowerShell 不识别 bash 的 `\` 续行符（续行用反引号 `` ` ``）。单行可在 bash / PowerShell 直接粘贴。

### 常用参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--start-time` / `--end-time` | 抽帧时间区间，`SS` / `MM:SS` / `HH:MM:SS` | 全片 |
| `--fps` | 每秒采样帧数 | 1.0 |
| `--max-frames` | 每个视频最多保存帧数 | 500 |
| `--hash-dist` | 感知哈希去重阈值（越大保留越多） | 8 |
| `--max-duration` | 跳过时长超过此值（秒）的视频 | 600 |
| `--output-dir` | 输出根目录 | `dataset` |

> `--max-duration` 默认 600 秒（10 分钟），更长的视频会在**下载前被静默过滤**，采集长视频需手动调大。

### 清洗与统计本地数据

采集后先 dry run，检查坏图、低分辨率图、低信息量帧、过暗/过亮帧和近重复帧：

```bash
python data/clean_data.py --dataset-dir dataset
```

确认报告后移动 rejected 图片到 `dataset_rejected/`：

```bash
python data/clean_data.py --dataset-dir dataset --apply
```

生成数据集统计报告：

```bash
python data/report_data.py --dataset-dir dataset
```

会输出 `dataset_cleaning_report.csv` 和 `dataset_report.json`，可用于 final report 的 data collection / data cleaning 章节。

当前 YouTube 新数据已经整理在：

- `dataset_youtube_hq/`: 10 类 ImageFolder 数据，10,000 张，全部 640×360
- `tobeclean/reports/dataset_youtube_hq_summary.json`: 数据总览、split plan、caveat
- `tobeclean/reports/dataset_youtube_hq_class_counts.csv`: 每类数量总表
- `tobeclean/reports/dataset_youtube_hq_report.json`: 类别数量、尺寸分布、80/10/10 split 统计
- `tobeclean/reports/dataset_youtube_hq_cleaning_report_final.csv`: 早期清洗记录，扩展后的最终数量以 summary/report JSON 为准
- `tobeclean/metadata/youtube_1000_expansion_manifest.csv`: YouTube source URL / frame provenance

---

## 训练与评估

```bash
# 训练 —— 保存 best_model.pth 和 training_history.json
python model/train.py

# 使用本地 dataset/<GameName>/ 数据训练
DATA_SOURCE=local python model/train.py

# 评估 —— 打印分类报告，保存 confusion_matrix.png / training_curves.png
python model/eval.py
```

> 后台运行这些脚本时 Python stdout 块缓冲，日志要进程退出才一次性刷新；想实时看日志加环境变量 `PYTHONUNBUFFERED=1`，判断训练是否真在跑可用 `nvidia-smi` 看 GPU 占用。

## 拍屏模式说明

上传一张实拍手机/显示器运行游戏的照片：

1. 优先用 **Mobile SAM** 对屏幕区域语义分割
2. SAM 不可用时依次尝试：亮度百分位阈值 → Otsu → Canny 边缘检测
3. 检测到的四边形经单应性变换做透视矫正
4. 矫正后的图像送入分类器（同样走 LetterboxResize）

预览图上的绿色角点表示屏幕检测成功。

## 数据集

**17 类合并集** = 两个来源去重合并，由 `data/build_combined_dataset.py` 构建到 `dataset_combined/`：

1. **经典 HF baseline** — [Bingsu/Gameplay_Images](https://huggingface.co/datasets/Bingsu/Gameplay_Images)，10 类各 1,000 张，导出在 `dataset_hf/`。
2. **自采 YouTube 高清** — 10 类各 1,000 张真实视频抽帧，在 `dataset_youtube_hq/`。

两套合并后 Fortnite / Minecraft / Genshin Impact 去重，共 17 类 20,000 张，640×360 PNG。训练时每类均衡到 1,000，经 `data.py` 分层切分 80 / 10 / 10（`seed=42` 可复现）。

## 分类结果（测试集，17 类各 100 张，共 1,700）

整体测试准确率 **99.53%**，每类 98–100%。完整报告见 `classification_report.txt`：

```
                 precision    recall  f1-score   support

     ARCRaiders       1.00      1.00      1.00       100
       Among Us       1.00      1.00      1.00       100
   Apex Legends       0.99      0.98      0.98       100
 CounterStrike2       1.00      1.00      1.00       100
       Fortnite       1.00      0.99      0.99       100
  Forza Horizon       1.00      1.00      1.00       100
      Free Fire       1.00      0.99      0.99       100
 Genshin Impact       0.99      0.99      0.99       100
     God of War       0.99      1.00      1.00       100
LeagueOfLegends       1.00      1.00      1.00       100
   MarvelRivals       0.99      1.00      1.00       100
      Minecraft       1.00      0.98      0.99       100
         Roblox       0.98      0.99      0.99       100
   RocketLeague       1.00      1.00      1.00       100
    Subnautica2       1.00      1.00      1.00       100
       Terraria       0.99      1.00      1.00       100
       Valorant       0.99      1.00      1.00       100

       accuracy                           1.00      1700
      macro avg       1.00      1.00      1.00      1700
   weighted avg       1.00      1.00      1.00      1700
```

误分类均为个位数，集中在写实画风/同类型游戏之间。混淆矩阵见 `confusion_matrix.png`。
