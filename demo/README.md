# Demo 组 `demo/`

负责 Gradio 前端，以及拍屏模式的屏幕检测与透视矫正。

## 文件

| 文件 | 职责 |
|------|------|
| `app.py` | Gradio Blocks 演示界面，两种模式（radio 切换） |
| `screen_crop.py` | 手机/显示器屏幕检测 + 透视矫正；`crop_screen()` / `load_sam()` |

## 用法（从项目根目录运行）

```bash
python demo/app.py
```

- 本地：http://localhost:7860
- `demo.launch(share=True)` 同时打印公网 `*.gradio.live` 链接（1 周有效）
- 两种模式：**直接截图分类** / **拍屏模式**（拍实拍照片自动矫正后分类）

## 拍屏检测流程（`crop_screen`）

四策略依次尝试，返回第一个成功的：
**Mobile SAM 语义分割**（主）→ 亮度百分位阈值 → Otsu → Canny 边缘（兜底）
检测到的四边形经单应性变换做透视矫正后送分类器。

## 注意点

- **预处理必须与训练一致**：`app.py` 的 `_transform` 复用 `from data.data import LetterboxResize`；改它必须和数据组 `train_tf`/`eval_tf` 三处同步，否则推理几何错位、准确率掉
- `app.py` 顶部的 `warnings.filterwarnings` 必须在 `from demo.screen_crop import ...` **之前**（屏蔽 timm/mobile_sam 噪音），不要挪动顺序
- 依赖 `best_model.pth`（模型组训练产出，在项目根），故必须从根目录运行
- Mobile SAM 权重 `mobile_sam.pt`（~40MB）首次自动下载；国内直连 huggingface.co 会卡，先手动下到项目根：
  ```bash
  curl.exe -L https://hf-mirror.com/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt -o mobile_sam.pt
  ```
- 示例图来自项目根 `examples/`；`load_sam()` 在启动时预热，首张拍屏不卡

## 依赖

`gradio` `torch` `torchvision` `Pillow` `opencv-python` `numpy`
拍屏分割：`mobile_sam`（`pip install git+https://github.com/ChaoningZhang/MobileSAM.git`）
