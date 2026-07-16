# Result

![image-20260716182738871](assets/result.png)

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

# checkpoint 不会自动恢复运行

## 📌 为什么会有 eval 模式的 checkpoint？

在 `base_trainer.py` 的 `save_checkpoint` 方法中，有这样的逻辑：



```python
if   epoch >= int(max_epochs*0.8)-1:
    if self._checkpoint_dir:
        if self.settings.local_rank in [-1, 0]:
            self.save_checkpoint(eval=True)   # 只保存网络权重，不保存 epoch、optimizer 等
elif ...:
    self.save_checkpoint()   # 普通保存，包含完整状态
```

当 `max_epochs=100` 时，`int(100*0.8)-1 = 79`，所以从第 79 轮开始，保存的 checkpoint 都是 **eval 版本**（只有 `net` 权重）。你的 `ep0082` 在 79 之后，因此是 eval 版本，缺少 `epoch`、`optimizer` 等字段。

这就是为什么 `load_checkpoint` 无法恢复 `epoch`，`trainer.epoch` 仍为 0。

------

## 🛠️ 解决方案

### 方案一：手动从文件名提取 epoch（快速继续训练）

在 `train_script.py` 中修改手动加载代码，添加正则提取 epoch 并强制赋值：

```python
import re

checkpoint_rel_path = "checkpoints/train/litetrack/B8_cae_center_got10k_ep100/LiteTrack_ep0082.pth.tar"
checkpoint_path = os.path.join(settings.save_dir, checkpoint_rel_path)
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    trainer.load_checkpoint(checkpoint=checkpoint_path)
    # 从文件名解析 epoch
    match = re.search(r'_ep(\d+)\.pth\.tar$', checkpoint_path)
    if match:
        epoch_num = int(match.group(1))
        trainer.epoch = epoch_num
        print(f"Resumed training from epoch {trainer.epoch}")
    else:
        print("Could not parse epoch from filename, starting from 0.")
else:
    print("Checkpoint not found, starting from scratch.")
```



这样虽然优化器状态没有恢复，但至少可以从第 83 轮继续训练（还剩 18 轮）。不过优化器是重新初始化的，学习率也会从初始值开始，可能导致 loss 突增，但通常仍能收敛。

------

### 方案二：使用第 78 轮（或更早）的完整 checkpoint（推荐）

如果你还保留有第 78 轮或更早的 checkpoint（比如 `LiteTrack_ep0078.pth.tar`），这些是**完整保存**的（包含 epoch、optimizer 等），加载它们可以完美恢复所有状态。检查你的输出目录：

```bash
ls -la /home/hugo/projects/cv/LiteTrack/output/checkpoints/train/litetrack/B8_cae_center_got10k_ep100/
```

如果有 `LiteTrack_ep0078.pth.tar`，建议使用它。修改 `checkpoint_rel_path` 为那个文件，然后正常加载（无需从文件名提取 epoch，因为 checkpoint 内部有 `epoch` 字段）。

------

### 方案三：修改保存策略，让所有 checkpoint 都保存完整状态

如果你希望以后所有 checkpoint 都能恢复，可以修改 `base_trainer.py` 中的 `save_checkpoint` 调用，将 `eval=True` 改为 `eval=False`（或直接去掉 `eval` 分支）。但这样会增大存储开销，不推荐。

------

## ⚠️ 注意事项

- 如果你使用了方案一（优化器未恢复），建议观察前几个 epoch 的 loss 是否异常。如果 loss 很高，可以适当降低学习率或增加 warmup。
- 如果方案一导致训练不稳定，可以尝试使用更早的完整 checkpoint（方案二）。

# 一、local.py 配置是否正确
整体没问题，但有一个小瑕疵：
settings.network_path 你设置成了一个具体的 .pth.tar 文件路径，但根据代码设计，它本意是一个目录（"Where tracking networks are stored"）。不过实际上测试代码并没有使用 network_path，checkpoint 的加载路径是在 lib/test/parameter/litetrack.py 中通过 save_dir 拼接的：

```python
params.checkpoint = os.path.join(save_dir, "checkpoints/train/litetrack/%s/LiteTrack_ep%04d.pth.tar" % (yaml_name,run_epoch))
```

所以 network_path 设成什么都不影响测试，只是个小冗余。
save_dir = /home/hugo/projects/cv/LiteTrack/output ✅ — 这是关键路径，checkpoint 就是通过它拼出来的。
results_path = /home/hugo/projects/cv/LiteTrack/output/test/tracking_results ✅
各数据集路径看起来都合理。

# 二、测试结果存放在哪里
根据 tracker.py 第 50 行的逻辑：

```python
self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
```

结合你的 local.py 配置，假设你运行：

```bash
python tracking/test.py litetrack B8_cae_center_got10k_ep100 --dataset lasot --ep 100
```

结果存放在

```bash
/home/hugo/projects/cv/LiteTrack/output/test/tracking_results/litetrack/B8_cae_center_got10k_ep100_100/
```

具体文件结构：
LaSOT 等离线评估数据集：每个序列一个 {sequence_name}.txt（包含每帧的 bbox）
GOT-10k / TrackingNet：会多一层子目录，如 {results_dir}/got10k/{sequence_name}.txt
每个 .txt 文件内容是该序列每帧的预测框（x, y, w, h），同时还有一个 {sequence_name}_time.txt 记录每帧的推理时间。

## 1. settings.network_path 需要指定正确吗？
不需要，它不影响测试。 整个测试流程中没有任何代码读取 network_path。你设成目录还是文件路径都无所谓，它只是一个遗留的配置项。

## 2. .pth.tar 文件路径在哪里指定的？
在 lib/test/parameter/litetrack.py 第 24 行：

```python
params.checkpoint = os.path.join(save_dir, "checkpoints/train/litetrack/%s/LiteTrack_ep%04d.pth.tar" % (yaml_name, run_epoch))
```

它用 local.py 中的 save_dir（而不是 network_path）+ 固定格式拼接出来的。
对应你的运行参数：
save_dir = /home/hugo/projects/cv/LiteTrack/output
yaml_name = B8_cae_center_got10k_ep100（即 tracker_param）
run_epoch = 100（即 --ep 的值）
最终拼出：

```python
/home/hugo/projects/cv/LiteTrack/output/checkpoints/train/litetrack/B8_cae_center_got10k_ep100/LiteTrack_ep0100.pth.tar
```

然后 checkpoint 的实际加载在 lib/test/tracker/litetrack.py 第 24 行：

```python
network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
```

## 3. 为什么之前不生成结果在 output 里？
因为你用了 --debug 2！
看 lib/test/evaluation/running.py 第 128 行和第 155-156 行：

```python
if _results_exist() and not debug:   # debug=2 时跳过检查，会重新跑
    print('FPS: {}'.format(-1))
    return
...
if not debug:                         # debug=2 时，not 2 = False，不保存！
    _save_tracker_output(seq, tracker, output)

```

**debug 非零时，结果不会被保存到磁盘**。 这是设计如此——debug 模式只跑推理、打印 FPS，但不写结果文件。
如果你想生成结果文件，把 --debug 去掉或设为 0

```bash
python tracking/test.py litetrack B8_cae_center_got10k_ep100 \
    --dataset_name got10k_test \
    --threads 6 --num_gpus 1 \
    --ep 100 99 98
```

这样结果就会保存到：

```bash
output/test/tracking_results/litetrack/B8_cae_center_got10k_ep100_100/got10k/
output/test/tracking_results/litetrack/B8_cae_center_got10k_ep100_099/got10k/
output/test/tracking_results/litetrack/B8_cae_center_got10k_ep100_098/got10k/
```

