# 数据采集方法 (Data Collection)

本项目最终的 **17 类**训练集由**两个来源合并**而成。所有脚本在 `data/` 下，一律**从项目根目录运行**。

## 总览

| 来源 | 类数 | 每类 | 脚本 | 落地目录 |
|------|------|------|------|----------|
| 经典公开数据集 (HuggingFace) | 10 | 1000 | `export_hf_dataset.py` | `dataset_hf/` |
| 自采 YouTube 高清帧 | 10 | 1000 | `collect_youtube_to_1000.py` | `dataset_youtube_hq/` |
| 合并去重 → **最终训练集** | **17** | 1000 | `build_combined_dataset.py` | `dataset_combined/` |

> 重叠的 Fortnite / Minecraft / Genshin Impact 两源都有，合并后物理上是 2000 张；训练时 `data.py` 再把每类**均衡到 1000**、分层切分 80/10/10（`seed=42` 可复现）。

---

## 来源一：经典 HuggingFace baseline

[`Bingsu/Gameplay_Images`](https://huggingface.co/datasets/Bingsu/Gameplay_Images) —— 10 款游戏各 1000 张 640×360。一行下载并导出成本地 ImageFolder：

```bash
python data/export_hf_dataset.py --output-dir dataset_hf
python data/report_data.py --dataset-dir dataset_hf --output dataset_hf_report.json
```

类别：Among Us · Apex Legends · Fortnite · Forza Horizon · Free Fire · Genshin Impact · God of War · Minecraft · Roblox · Terraria。作为可复现的公开基线。

---

## 来源二：自采 YouTube 高清帧（核心）

公开数据集里没有的、更新的 10 个游戏（ARC Raiders、CS2、League of Legends、Marvel Rivals、Rocket League、Valorant、Subnautica 2 等），从 YouTube 真实 gameplay 视频抽帧得到。

### 怎么采的（方法论）

核心原则：**只用真实视频帧，保证来源多样性，去掉重复和低质帧。** 分五步：

**① 选源 —— 用搜索词而不是随便找**
`collect_youtube_to_1000.py` 给每个游戏预设了 3–4 条搜索词，刻意挑「无解说 / 新版本 / 实战」的视频：

```
"Minecraft survival gameplay no commentary 2026"
"Valorant ranked gameplay 2026 no commentary"
"ARC Raiders extraction gameplay no commentary"
...
```

- **no commentary**：避开摄像头小窗、解说脸、直播间装饰，画面只剩游戏本身
- **多条查询 × 每条取前 8 个视频**：同一游戏来自不同主播 / 地图 / 角色 / 场景，让模型学「游戏」而不是「某个视频」
- 自动对 URL 去重，避免同一视频被重复抽帧

**② 下载 —— 绕过 YouTube 限制**（底层 `collect_data.py`，用 `yt-dlp`）
- `player_client=android_vr,android`：YouTube 近年对网页端只给 SABR 流，Android VR/testsuite 客户端能拿到正常 HTTPS 的 MP4，无需 ffmpeg 合流
- 优先 **≤480p mp4**：224px 模型输入足够，比多 GB 的 720p+ 下载更快更稳
- 需要 **`deno`**（JS 运行时）解 YouTube 的 n-challenge，否则报 `n challenge solving failed`；脚本会自动探测 winget/scoop 装的 deno 并注入 PATH

```bash
winget install denoland.deno   # 一次性安装 JS runtime
```

**③ 抽帧 —— 控制采样率与区间**
- 默认 **2 fps** 采样，只取视频的 **0:30–12:00** 区间（跳过片头/片尾/加载）
- 每帧 resize 到 **640×360**
- 每条视频/查询有帧数上限，防止单一来源占比过高

**④ 去重 —— 感知哈希 (pHash)**
- 每帧算 pHash，与已保留帧的汉明距离 **< 8** 即判为近重复并丢弃
- 游戏里大量静止/重复镜头，这步显著提升有效数据比例

**⑤ 配额与溯源**
- 每类采到 **1000 张**即停，超出的自动移到 `tobeclean/rejected/`
- 每张图的来源都写进 manifest（`tobeclean/metadata/youtube_1000_expansion_manifest.csv`）：**源 URL、视频时间点、pHash、尺寸** —— 完全可追溯、可复现

### 一键复现自采集

```bash
# 把所有 10 个 YouTube 类各采到 1000 张
python data/collect_youtube_to_1000.py

# 只采某几个类
python data/collect_youtube_to_1000.py --classes Valorant ARCRaiders
```

### 单独采某个游戏（底层引擎 collect_data.py）

```bash
# 指定视频 + 时间区间
python data/collect_data.py --game "Minecraft" --url "https://youtu.be/XXX" --start-time 1:30 --end-time 5:00

# 用搜索自动下载前 N 个结果
python data/collect_data.py --game "Valorant" --search "Valorant gameplay 2024 no commentary" --max-videos 3

# 从本地已下载的高清视频抽帧（最稳，不受 YouTube 限制）
python data/collect_data.py --game "RocketLeague" --video-dir raw_videos/RocketLeague --fps 1 --max-frames 500
```

> 默认**只用真实视频帧**。被 YouTube 403/SABR 挡下的视频直接跳过；只有显式加 `--allow-storyboard` 才会退用低清 storyboard 缩略图（仅供临时 debug，不进最终数据）。输出 `dataset/<Game>/frame_XXXXX.png` + `collection_manifest.csv`（同样逐张记录 provenance）。

---

## 清洗与统计

```bash
# 质检：坏图 / 低分辨率 / 过暗过亮 / 低信息量 / 近重复（先 dry-run 看报告）
python data/clean_data.py --dataset-dir dataset_youtube_hq
python data/clean_data.py --dataset-dir dataset_youtube_hq --apply    # 确认后移走 rejected

# 统计：类别数量、尺寸分布、80/10/10 切分计划
python data/report_data.py --dataset-dir dataset_youtube_hq
```

---

## 合并成 17 类训练集

```bash
python data/build_combined_dataset.py
```

把 `dataset_hf/`（经典 10）+ `dataset_youtube_hq/`（自采 10）**硬链接**合并到 `dataset_combined/`，重叠的 Fortnite / Minecraft / Genshin Impact 去重 → 17 类共 20000 张，并生成 `combined_manifest.csv` 记录每张图来自哪个源。

---

## 完整复现（从零到训练好的模型）

```bash
# 1. 经典 10 类
python data/export_hf_dataset.py --output-dir dataset_hf

# 2. 自采 10 类（需先: winget install denoland.deno）
python data/collect_youtube_to_1000.py
python data/clean_data.py --dataset-dir dataset_youtube_hq --apply
python data/report_data.py --dataset-dir dataset_youtube_hq

# 3. 合并成 17 类
python data/build_combined_dataset.py
```
```powershell
# 4. 训练 / 评估（PowerShell 设环境变量；用合并后的本地数据）
$env:DATA_SOURCE='local'; python model/train.py
$env:DATA_SOURCE='local'; python model/eval.py
```

> 仓库已附带训练好的 `best_model.pth`（17 类，test 99.53%），想直接看效果可跳过 1–4 步，直接 `python demo/app.py`。
