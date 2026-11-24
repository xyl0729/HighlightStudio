# Highlight Studio

**Highlight Studio** 是一款可本地部署的多模态视频自动高光剪辑与动态重构画幅工具。

导入本地视频后，系统自动完成两个核心决策：
1. **选哪一段（高光时间定位）**：结合视频多模态大模型（Video-ORA-4B）、CLIP 视觉语义与 Whisper 语音特征，定位视频中最值得保留的高光片段，并对长区间进行边界校准与多段切分。
2. **每一帧截哪里（主体跟随重构画幅）**：结合目标检测（YOLO11s）、视觉显著性（UNISAL）、候选排序与轨迹平滑，生成覆盖全片的 `9:16` 竖屏或 `16:9` 横屏逐帧裁剪框。

分析完成后，可在内置的非线性剪辑工作台（NLE）中边播放边查看**原片追踪框 + 实时裁剪预览**，通过时间轴或按帧微调起止点、增删/切分片段，最后导出带原音轨的 MP4 成片。

---

## 核心特性

- **多模态高光定位 + 主体跟随裁剪**：内置完整推理管线（`engine/`）及轻量级校准/排序参数，兼顾时间段完整度与跨画幅主体居中构图。
- **双屏实时监看**：支持「原片叠加 / 双屏对比 / 成片构图」三种视图、三分构图线、逐帧步进与「仅播高光」模式（导出前即可直接试看拼接节奏）。
- **专业时间轴交互**：支持拖拽平移、左右手柄修剪、播放头切分（`S`）、设入/出点（`I` / `O`）、按帧精调，以及基于真实裁剪轨迹的平移可视化。
- **批量排队与本地持久化**：支持一次拖入多条视频并分别指定 `9:16` / `16:9`，后台顺序执行；所有任务、已保存片段与导出成片自动写入本地工作目录（默认 `.work/`），关闭网页或重启后端后可从右上角「任务队列」直接恢复。
- **保留原音轨拼接**：导出时按选中帧区间精确裁剪并拼接源视频音轨，重编码封装为 30 FPS、偶数像素对齐的标准 H.264/AAC MP4。

---

## 项目结构

```text
highlight-studio/
├── frontend/                  # React + Vite + TypeScript 非线编前端界面
├── studio/                    # FastAPI 后端服务、任务队列持久化、FFmpeg 转码与导出
├── engine/                    # 多模态高光定位与动态裁剪推理管线（含轻量校准与排序参数）
├── hlcut/                     # 通用视频高光与构图几何工具库
├── tests/                     # 后端与坐标映射自动化测试（基于代码生成合成素材）
├── docs/API.md                # HTTP 接口与数据协议说明
├── start-local.ps1            # Windows 本地后端一键启动脚本
├── requirements.txt           # Web 后端与媒体处理依赖
├── requirements-ai.txt        # 多模态推理依赖
├── THIRD_PARTY_LICENSES.md    # 第三方模型与组件来源及许可说明
└── LICENSE                    # 本项目开源许可（MIT）
```

---

## 环境要求

- **Python**: 3.11
- **Node.js**: 20+（用于启动前端工作台）
- **GPU**: NVIDIA CUDA GPU（实测环境：Windows + NVIDIA RTX 4060 Laptop 8GB VRAM，Video-ORA-4B 以 4-bit 量化运行）
- **FFmpeg**: 通过 `imageio-ffmpeg` 自动提供，无需单独配置系统环境变量

---

## 快速开始

### 1. 安装后端与推理依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 按本机 CUDA 驱动安装对应版本的 PyTorch（以下为 CUDA 12.8 示例）
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
```

### 2. 下载第三方模型权重（需另行获取，合计约 12 GB）

由于第三方基础模型体积较大且各自遵循独立许可协议，本仓库不随附模型权重。请在本地新建模型目录（例如 `D:\models`），按以下目录名放置对应模型：

| 本地目录 / 文件名 | 模型 | 官方来源 | 用途 |
|---|---|---|---|
| `Video-ORA-4B/` | Video-ORA-4B | [HuggingFace: OraRL/Video-ORA-4B](https://huggingface.co/OraRL/Video-ORA-4B) | 视频时间高光定位（4-bit 量化推理） |
| `CLIP-ViT-B-32/` | CLIP ViT-B/32 | [HuggingFace: openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | 视觉语义特征提取 |
| `Whisper-small/` | Whisper small | [HuggingFace: openai/whisper-small](https://huggingface.co/openai/whisper-small) | 语音与音频线索提取 |
| `UNISAL-upstream/` | UNISAL | [GitHub: rdroste/unisal](https://github.com/rdroste/unisal)（Apache-2.0） | 视觉显著性构图估计（直接 `git clone https://github.com/rdroste/unisal UNISAL-upstream`，仓库内已包含 `training_runs/pretrained_unisal/weights_best.pth`） |
| `yolo11s.pt` | YOLO11s | [GitHub: ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 画面主体目标检测 |

放置完成后，模型目录结构应如下所示：

```text
<你的模型目录>/
├── Video-ORA-4B/
├── CLIP-ViT-B-32/
├── Whisper-small/
├── UNISAL-upstream/
└── yolo11s.pt
```

### 3. 启动后端服务

Windows PowerShell（脚本会自动定位仓库内置的 `./engine` 目录）：

```powershell
.\start-local.ps1 -FinalModels 'D:\models'
```

Linux（需配备 NVIDIA CUDA GPU）或自定义命令启动：

```bash
export HIGHLIGHT_FINAL_CODE="$(pwd)/engine"
export HIGHLIGHT_FINAL_MODELS="/path/to/models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m uvicorn studio.app:app --host 127.0.0.1 --port 8000
```

### 4. 启动前端界面

在另一个终端中进入 `frontend/` 目录：

```powershell
cd frontend
npm ci
npm run dev
```

打开浏览器访问：
- 剪辑工作台：<http://127.0.0.1:5173/>
- 后端交互式 API 文档：<http://127.0.0.1:8000/docs>

---

## 使用说明与环境变量

- **批量导入**：可一次拖入或多选多条本地视频，为每条分别指定 `9:16` 或 `16:9`，点击「加入队列」后后台按顺序自动处理。
- **本地任务持久化**：每个任务的状态、推荐片段、手动编辑结果与导出 MP4 均保存在工作目录（默认 `.work/`，已被 `.gitignore` 忽略）。重新打开网页或重启服务后，点击右上角「任务队列」即可直接恢复。
- **输入限制**：单视频上限 `512 MiB`、时长最长 `10 分钟`。预览流标准化为 `30 FPS`、最长边 `≤ 1280px`。
- **常用环境变量**：
  - `HIGHLIGHT_FINAL_MODELS`：本地第三方模型根目录（必填）
  - `HIGHLIGHT_FINAL_CODE`：推理引擎目录（默认自动使用仓库内的 `./engine`）
  - `HIGHLIGHT_WORK_DIR`：任务缓存与持久化工作目录（默认 `.work`）
  - `HIGHLIGHT_ORIGINS`：允许跨域访问的前端地址，多个以逗号分隔（默认 `http://localhost:5173,http://127.0.0.1:5173`）

---

## 运行测试

后端回归测试完全使用 FFmpeg 程序生成的合成有声视频，不依赖任何外部视频素材：

```powershell
.\.venv\Scripts\python.exe -m pip install httpx==0.28.1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

前端类型检查与构建验证：

```powershell
cd frontend
npm run build
```

---

## 开源许可

- 本仓库自有代码采用 [MIT License](LICENSE)。
- 运行所需的外部模型与组件（包括 Video-ORA-4B、CLIP、Whisper、UNISAL、Ultralytics YOLO11s 与 FFmpeg）各自受其原始开源协议约束，详见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。
