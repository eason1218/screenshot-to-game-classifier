# 数据组 `data/`

本目录负责项目的数据部分：从公开视频采集 gameplay frame、清洗低质量/重复图片、统计数据平衡性，并把本地数据接入 PyTorch 训练流程。

## 文件

| 文件 | 职责 |
|------|------|
| `collect_data.py` | YouTube / 本地视频抽帧；按时间区间抽帧；pHash 去重 + provenance manifest |
| `export_hf_dataset.py` | 把 Hugging Face `Bingsu/Gameplay_Images` baseline 下载/导出成本地 ImageFolder |
| `clean_data.py` | 本地数据质量控制：坏图、低分辨率、过暗/过亮、低信息量、近重复图 |
| `report_data.py` | 生成数据集统计报告：类别数量、类别平衡、图片尺寸、计划切分 |
| `data.py` | 训练/验证/测试 DataLoader，支持 HuggingFace baseline 和本地 `dataset/` |
| `HANDOFF.md` | 给模型组的最终交接说明 |

## 推荐数据流程

### 0. 下载 Hugging Face baseline 数据集

如果需要把之前提到的 `Bingsu/Gameplay_Images` 下载成本地文件夹：

```bash
python data/export_hf_dataset.py --output-dir dataset_hf
python data/report_data.py --dataset-dir dataset_hf --output dataset_hf_report.json
```

这个数据集是 10 类 baseline：Among Us, Apex Legends, Fortnite, Forza Horizon, Free Fire, Genshin Impact, God of War, Minecraft, Roblox, Terraria。它可以作为可复现的公开数据来源，但不包含 Valorant、League of Legends、Rocket League，所以如果最终项目坚持这几个新游戏，仍然需要用自采视频补齐。

### 1. 采集每个游戏的视频帧

输出格式是 `dataset/<GameName>/frame_XXXXX.png`，兼容 `torchvision.datasets.ImageFolder`。
采集脚本同时追加 `dataset/collection_manifest.csv`，记录每张图的来源 URL、视频时间点和 pHash。

理想最终数据应来自真实视频帧，而不是 YouTube storyboard 缩略图。当前脚本默认禁用 storyboard fallback；只有临时 debug 时才加 `--allow-storyboard`。脚本会优先尝试 YouTube Android/Android VR client；被 PO token / SABR 限制挡住的视频会被跳过。

```bash
python data/collect_data.py --game "LeagueOfLegends" --url "https://www.youtube.com/watch?v=..." --start-time 1:30 --end-time 6:30 --fps 1 --max-frames 300 --max-duration 1200
```

也可以用 YouTube 搜索自动抓前几个结果：

```bash
python data/collect_data.py --game "Valorant" --search "Valorant gameplay 2024 no commentary" --max-videos 3 --fps 1 --max-frames 300
```

如果已经手动下载了高清 gameplay 视频，推荐直接从本地视频抽帧：

```bash
python data/collect_data.py --game "RocketLeague" --video-dir raw_videos/RocketLeague --fps 1 --max-frames 500 --start-time 0:30 --end-time 12:00
```

本地视频是最稳定的高质量路径：不会受 YouTube 403/SABR 限制影响，也能保证抽出的帧来自真实 720p/1080p 视频。

采集建议：

- 每类至少 500-1000 张清洗后的高清真实视频帧，类别之间尽量接近。
- 优先选纯 gameplay、少人脸、少菜单、少开场动画的视频。
- 每个游戏尽量来自多个视频/地图/角色/场景，避免模型只记住某一个主播或地图。
- 文件夹名就是类别名，最终训练时要和 `config.CLASS_NAMES` 顺序/内容同步。

### 2. 清洗低质量和重复帧

先 dry run 看报告，不会移动文件：

```bash
python data/clean_data.py --dataset-dir dataset
```

确认后把 rejected 图片移动到 `dataset_rejected/`：

```bash
python data/clean_data.py --dataset-dir dataset --apply
```

清洗会检查：

- 图片是否损坏或无法读取
- 分辨率是否太小
- 是否几乎全黑、全白、低纹理
- 同一类别内是否和已保留图片过于相似

输出 `dataset_cleaning_report.csv`，可以作为 final report 的数据清洗证据。当前 10 类 YouTube 数据的最终数量统计在 `tobeclean/reports/dataset_youtube_hq_summary.json` 和 `tobeclean/reports/dataset_youtube_hq_report.json`。

### 3. 生成数据报告

```bash
python data/report_data.py --dataset-dir dataset
```

输出：

- 每个类别图片数
- 最大/最小类别比例
- 图片尺寸分布
- 计划中的 80/10/10 train/val/test split
- `dataset_report.json`

### 4. 训练时使用本地数据

`data.py` 默认仍使用 HuggingFace baseline，保证原模型代码不被破坏。要切换到本地数据，有两种方式：

```python
from data.data import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(source="local")
```

或运行训练前设置环境变量：

```bash
DATA_SOURCE=local python model/train.py
```

本地数据默认会做 class balancing：每一类最多取到最小类别的数量，避免某个游戏图片太多导致模型偏向它。也可以在代码里设置 `balance_local=False` 或 `max_per_class=800`。

## 对外接口

```python
from data.data import get_dataloaders
from data.data import get_local_dataloaders
from data.data import get_transforms
from data.data import LetterboxResize
```

- `get_dataloaders(source="hf")`: 使用 HuggingFace `Bingsu/Gameplay_Images` baseline。
- `get_dataloaders(source="local")`: 使用 `config.DATASET_DIR` 下你们采集的数据，当前为 `dataset_youtube_hq/`。
- `get_dataloaders(source="auto")`: 优先本地数据；本地类别不足时 fallback 到 HuggingFace。
- `LetterboxResize`: 等比缩放 + 居中补边到 224x224，训练、评估、demo 必须保持一致。

## 和其他组的接口约定

- 如果新增/替换游戏类别，需要同步修改根目录 `config.py` 的 `CLASS_NAMES` 和 `NUM_CLASSES`。
- 本地 `ImageFolder` 标签默认按文件夹名字母序排列。为了避免 label mismatch，最终训练前应让 `config.CLASS_NAMES` 与打印出的 `Class mapping` 一致。
- `demo/app.py` 推理预处理复用 `LetterboxResize`，不要单独改 demo 预处理。
- 当前根目录已有 `best_model.pth` 是旧 10 类模型；如果本地类别改成 League/Valorant/Rocket League 等，需要重新训练模型。

## 可写进 final report 的 data 部分

Our data pipeline collects gameplay screenshots from publicly available gameplay videos. For each target game, we use high-resolution gameplay videos whenever possible, extract frames at a controlled sampling rate, resize frames to 640x360, and remove near-duplicate frames using perceptual hashing. We then run a quality-control script to filter invalid images, low-resolution frames, low-information frames, overly dark/bright frames, and remaining near-duplicates. The cleaned dataset is organized in ImageFolder format with one folder per game class. We generate a dataset report containing class counts, image size distribution, class-balance ratio, and the planned stratified 80/10/10 train/validation/test split. During training, images are loaded through a reproducible PyTorch DataLoader with stratified splitting, optional class balancing, and augmentations designed for real-world screenshots and photos of screens.
