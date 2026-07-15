# Error
```bash
Traceback (most recent call last):
  File "/home/hugo/projects/cv/LiteTrack/tracking/create_default_local_file.py", line 5, in <module>
    from lib.test.evaluation import create_default_local_file_ITP_test
  File "/home/hugo/projects/cv/LiteTrack/tracking/../lib/test/evaluation/__init__.py", line 1, in <module>
    from .data import Sequence
  File "/home/hugo/projects/cv/LiteTrack/tracking/../lib/test/evaluation/data.py", line 3, in <module>
    from lib.train.data.image_loader import imread_indexed
ModuleNotFoundError: No module named 'lib.train.data'
```

### # Fix

Download/Copy the lib.train.data from OSTrack code.

# `jpeg4py` 图像解码失败

# (注：jpeg4py维护不活跃，最后一次发布在 2024 年，社区反馈少)

## 📌 背景

在运行 LiteTrack 跟踪模型训练时，遇到了一系列环境依赖和兼容性问题。最终目标是在 Python 3.12 + PyTorch 新版本环境下成功启动训练。以下是完整的问题排查、原因分析和解决方案总结，特别针对图像解码库 `jpeg4py` 的曲折解决过程。

------

## 📋 问题排查历程概览

| 顺序 | 错误现象                                                     | 缺失/问题库      | 解决方法                                                     |
| :--- | :----------------------------------------------------------- | :--------------- | :----------------------------------------------------------- |
| 1    | `ModuleNotFoundError: No module named 'einops'`              | einops           | `pip install einops`                                         |
| 2    | `ModuleNotFoundError: No module named 'imp'` (Python 3.12 移除) | imp (已废弃)     | 修改代码用 `importlib.machinery` 替换，或降级 Python（用户选择修改代码） |
| 3    | `ModuleNotFoundError: No module named 'addict'`              | addict           | `pip install addict`                                         |
| 4    | `ModuleNotFoundError: No module named 'yapf'`                | yapf             | `pip install yapf`                                           |
| 5    | `FileNotFoundError: pretrained_models/cae_base.pth`          | 预训练权重       | 从官方渠道下载权重文件                                       |
| 6    | `OSError: Could not load libjpeg-turbo library` + `ERROR: Could not read image` | jpeg4py + 系统库 | 安装 `libjpeg-turbo8-dev`、`libturbojpeg0-dev`，并用 `ldconfig` 永久添加库路径 |

> 注：问题 2（imp）和 5（权重）由用户自行解决，不在本笔记重点，但作为背景列出。

------

## 🔥 核心问题：`jpeg4py` 图像解码失败

### 错误表现

- 训练数据加载时抛出大量 `ERROR: Could not read image ...` 以及 `OSError: Could not load libjpeg-turbo library`。
- 伴随大量 `Exception ignored in: <function JPEG.__del__>` 和 `AttributeError: 'JPEG' object has no attribute 'decompressor'` 等垃圾信息。

### 根本原因

1. `jpeg4py` 是一个基于 `libjpeg-turbo` 的 Python 图像解码库，用于高速读取 JPEG 图像。
2. 安装 `jpeg4py` 时需要编译 C 扩展，依赖于系统的 `libjpeg-turbo` 开发包（头文件和动态库）。
3. 即使编译成功，运行时仍需动态链接 `libjpeg-turbo` 的共享库（如 `libjpeg.so.8` 或 `libturbojpeg.so`）。若系统找不到这些库，则解码失败。
4. 用户系统原本只安装了标准 `libjpeg`，缺少 `libturbojpeg`，且库路径未被动态链接器缓存。

### 解决步骤（详细）

#### 1️⃣ 安装系统级开发包

```bash
sudo apt update
sudo apt install libjpeg-turbo8-dev    # 提供 libjpeg 头文件和库
sudo apt install libturbojpeg0-dev    # 提供 libturbojpeg 专用库
```

#### 2️⃣ 重新安装 `jpeg4py` 以链接新库

```bash
pip uninstall jpeg4py
pip install --no-cache-dir jpeg4py
```

#### 3️⃣ 验证库是否可被运行时加载

临时设置环境变量（测试用）：

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
python -c "import jpeg4py; img = jpeg4py.JPEG('test.jpg').decode(); print(img.shape)"
```

若输出形状，说明成功。

#### 4️⃣ 永久化库路径（避免每次设置环境变量）

```bash
sudo sh -c 'echo "/usr/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/libjpeg-turbo.conf'
sudo ldconfig
```

此操作将路径写入系统动态链接器缓存，所有进程均可自动找到库。

> **注意**：之后无需再设置 `LD_LIBRARY_PATH`，且该配置对所有 Conda 环境生效。

------

## ⚖️ `jpeg4py` 与 OpenCV (`cv2.imread`) 的详细对比

在解决过程中，我们曾考虑过是否该放弃 `jpeg4py` 改用 OpenCV。以下是从多个维度对二者的深入比较，以供日后决策。

| 对比维度           | `jpeg4py`                                                    | OpenCV (`cv2.imread`)                                       |
| :----------------- | :----------------------------------------------------------- | :---------------------------------------------------------- |
| **底层实现**       | 直接绑定 `libjpeg-turbo`（SIMD 优化）                        | 封装了 `libjpeg-turbo`（或标准 libjpeg），但有额外开销      |
| **性能（单线程）** | 极快，比 PIL 约快 1.3 倍                                     | 良好，但略慢于 `jpeg4py`                                    |
| **性能（多线程）** | 优势明显，可快 9 倍                                          | 多线程提升有限，受 GIL 和 I/O 影响                          |
| **输出色彩格式**   | 直接输出 RGB (H,W,3) numpy 数组                              | 默认输出 BGR，需手动 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| **平台兼容性**     | 主要 Linux，Windows/Mac 支持有限                             | 全平台（Win/Linux/macOS）                                   |
| **依赖与安装**     | 需系统安装 `libjpeg-turbo` 开发库，且有版本匹配问题；`pip install` 需编译 | `pip install opencv-python` 即装即用，自带依赖              |
| **维护状态**       | 不活跃，最后一次发布在 2024 年，社区反馈少                   | 活跃，由 OpenCV 基金会持续维护                              |
| **错误处理**       | 加载失败时抛出异常，但 `__del__` 有时有 bug                  | 失败时返回 `None`，需主动检查                               |
| **适用场景**       | 对解码速度要求极高，且环境可控（如 Linux 服务器）            | 通用、稳定，适合快速开发和跨平台部署                        |

### 性能数据参考（来自公开测试，12,000 张 JPEG 图片）

- `jpeg4py`（单线程）：43.73 秒
- OpenCV（单线程）：68.26 秒
- NVIDIA DALI（GPU 加速）：9.86 秒（单线程），1.16 秒（16 线程）

### 选择建议

- 若追求极致 I/O 性能且环境为 Linux，可选用 `jpeg4py`，**但需忍受潜在的维护问题。**
- 若追求稳定、易用和跨平台，优先选择 OpenCV（或 PIL）。
- 若使用 NVIDIA GPU 且数据量极大，推荐尝试 DALI，可彻底解放 CPU 瓶颈。

------

## 🛠️ 关键操作与知识点总结

### 1. `ldconfig` 与 `LD_LIBRARY_PATH`

- **`ldconfig`**：更新 `/etc/ld.so.cache`，使系统在默认搜索路径（如 `/lib`、`/usr/lib`）之外找到自定义路径的共享库。是系统级永久方案。
- **`LD_LIBRARY_PATH`**：环境变量，仅在当前 shell 生效，优先级高于 `ldconfig`。适合临时测试，不适合长期使用（可能污染环境）。
- **推荐**：对于系统级库（如 `libjpeg`），使用 `ldconfig` 更干净、更稳定。

### 2. Conda 环境隔离与系统依赖

- Conda 环境仅隔离 Python 包，**系统级库**（如 `.so` 文件）仍由操作系统管理。
- 因此，即使切换 Conda 环境，只要系统库已通过 `ldconfig` 配置好，新环境只需 `pip install jpeg4py` 即可正常工作。

### 3. Python 3.12 移除 `imp` 模块的应对

- `imp` 在 Python 3.4 弃用，3.12 正式移除。
- 替代方案：使用 `importlib.machinery.SourceFileLoader` 加载源文件，或用 `importlib.util.spec_from_file_location` 等。
- 若旧项目大量依赖 `imp`，可考虑降级 Python 或花费时间重构代码（用户选择了重构）。

### 4. 预训练权重管理

- 许多模型需下载预训练权重（如 `cae_base.pth`），通常放置于项目指定目录（如 `pretrained_models/`）。
- 解决方式：查阅项目文档，从官方链接下载，或使用提供的脚本。

------

## ✅ 最终状态与后续注意事项

- `jpeg4py` 已正确安装并可通过系统库加载，训练数据加载正常。
- `imp` 问题已通过代码修改解决。
- 预训练权重已下载到位。
- 训练成功启动。

**建议日后维护时注意：**

- 若迁移到新机器，需重复安装系统依赖（`libjpeg-turbo8-dev`、`libturbojpeg0-dev`）并执行 `ldconfig`。
- 若不想被系统库束缚，可考虑改用 OpenCV，只需一次代码修改，日后再无依赖问题。
- 关注 `jpeg4py` 的更新，若长期无维护，建议有计划地迁移到更现代的方案。

------

## 📘 总结

本次环境配置的难点主要集中在 **`jpeg4py` 的系统依赖**和 **Python 3.12 的兼容性**。通过逐步安装缺失包、配置动态库路径、修改废弃 API，最终成功启动训练。整个过程体现了深度学习项目中“环境即基础设施”的重要性，以及系统库管理、Python 版本选择对项目成败的影响。

笔记存档完毕，可供日后参考或分享。
