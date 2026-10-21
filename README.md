# 77 象棋识别模型 v2

`77xiangqi_recongnize_v2` 当前对外可用主线，以棋子分类为主。

- `piece_classifier` 已训练完成
- 当前可用识别链路暂时假设棋盘定位由用户手动准确框选，再做后续 90 格切分与 15 类分类

棋盘定位这部分，之前已经尝试过多条路线：

1. 纯传统算法
2. 外框 `bbox` 定位
3. 棋子中心和交点定位后恢复棋盘
4. `heatmap` 方案

这些方案都暴露过明显缺陷，暂时还没有形成在真实场景中足够稳定的主线。

由于当前目标是先支持复盘复习，所以现阶段先采用：

```text
用户准确框选棋盘 -> 90格 -> 15分类
```

仓库中仍保留 `board_detector` 相关实验、训练和导出脚本，作为后续继续推进棋盘定位的基础。

Python 的交付目标是：

- 训练棋子分类模型
- 保留并继续迭代棋盘定位实验
- 导出 ONNX
- 提供一份本地可跑的参考串联逻辑

Python 不是线上真源服务，不负责最终 FEN、合法性、门控、状态机和连线动作。

## 项目边界

当前 Python 只负责：

- 当前可用主线：
  - `用户框选棋盘 -> 90格 -> 15分类`
- 实验能力：
  - 棋盘定位相关训练、导出和推理脚本
- 导出：
  - `piece_classifier.onnx`
  - `board_detector.onnx`（实验）
- 给 Rust 提供一份足够直接的参考实现

Rust 负责：

- 90 格分类结果转 FEN
- 合法性
- 连线状态机
- compare / gating / rescue / emit
- 与前端和平台层集成

Rust 边界参考：

- 文档目录：`C:\projects\77xiangqi\docs\link`
- 代码目录：`C:\projects\77xiangqi\tauri_backend\src-tauri\src\link`

## 当前目录

```text
configs/
  label_map.json
  train/
    board_detector.yaml
    piece_classifier.yaml

docs/
  01-project-overview.md

scripts/
  build_board_detector_dataset.py
  build_piece_classifier_dataset.py
  train_board_detector.py
  export_board_detector_onnx.py
  train_piece_classifier.py
  export_piece_classifier_onnx.py

src/xiangqi_recognize_v2/
  board_detector/
  data_prep/
  piece_classifier/
  reference/
  shared/

tests/
```

## 分层说明

`src/xiangqi_recognize_v2/board_detector/`

- `dataset.py`
  - 棋盘定位训练集读取
- `model.py`
  - `MobileNetV3-Small` bbox 回归头
- `train.py`
  - 训练入口
- `export.py`
  - 导出 ONNX
- `predict.py`
  - ONNX 推理参考实现

`src/xiangqi_recognize_v2/piece_classifier/`

- `dataset.py`
  - 15 类 ImageFolder 数据读取
- `model.py`
  - 4 层 `3×3` 卷积，每层 BatchNorm、ReLU、`2×2` 最大池化，再全局平均池化接 15 类线性层
  - 输入 `uint8 NHWC`。`/255` 和转 `NCHW` 在模型内完成
- `train.py`
  - 训练入口
- `export.py`
  - 导出 ONNX
- `predict.py`
  - ONNX 推理参考实现

`src/xiangqi_recognize_v2/reference/`

- 本地参考串联逻辑
- 只做：
  - 图片输入
  - 棋盘 bbox 推理
  - 90 格切分
  - 15 类分类
- 不做：
  - FEN 输出
  - 合法性判断
  - Rust 状态机模拟

## 数据约定

### 1. 棋盘定位数据

目录结构：

```text
data/processed/board_detection/
  train/
    images/
    labels/
  val/
    images/
    labels/
```

标注文件为同名 JSON：

```json
{"bbox": [x1, y1, x2, y2]}
```

含义：

- `x1, y1, x2, y2` 都是原图像素坐标
- `x2 > x1`
- `y2 > y1`

### 2. 棋子分类数据

目录结构：

```text
data/processed/train_cells/
  train/
    empty/
    rk/
    ...
  val/
    empty/
    rk/
    ...
```

标签顺序固定由 [configs/label_map.json](/C:/projects/77xiangqi_recongnize_v2/configs/label_map.json) 定义。

## 数据增强原则

两个模型的增强尽量复用同一套场景库和素材变化，重点覆盖：

- 不同主题
- 无外框
- 粗外框
- 淡线
- 深线
- 不同背景
- 不同棋子局面
- 开局 / 中局 / 残局 / 空棋盘
- 不同截图尺寸
- 不同棋盘位置

其中：

- `board_detector` 更关注整图级变化、棋盘位置和边框风格
- `piece_classifier` 更关注单格 patch 中的主题、线条、背景和棋子变化

当前 `piece_classifier` 还没有加入以下增强：

- 拉伸
- 旋转

如果后续要提升真实场景识别，建议至少补上：

- 拉伸增强
- 旋转增强
- 轻度透视形变
- 亮度 / 对比度 / 压缩噪声 / 轻模糊扰动

这类增强配合稍大一点的模型，通常会比当前版本更适合真实截图或拍摄场景。

当前数据生成层已经重新保留的关键能力：

- 固定 `val_piece_skin_ids`
  - 某些棋子皮肤强制只进验证集，不走随机切分
  - 构建前会校验每一类棋子的源 PNG 在 train/val 间没有完全相同的文件；发现重复即拒绝生成，避免验证集泄漏
- 随机布局
  - 不再只依赖空盘和标准开局
- 棋子光圈增强
  - 默认用于 `piece_classifier`
- 空格增强
  - 默认用于 `piece_classifier`
- 低风险真实干扰
  - `piece_classifier` 和 `board_detector` 都可用
- 棋盘场景放置随机化
  - 棋盘不再总是铺满整张图
  - 默认至少一半宽屏场景，另一半窄屏场景
  - 随机上下留白、左右偏移和整图尺度
  - 背景使用独立的纯色 / 渐变 / 轻纹理 UI 底，不再拿棋盘图当背景
  - 加入更强的 UI 条、侧边面板、底部面板、浮层和局部遮挡
  - 仅对 `board_detector` 少量加入中文文字干扰，例如 `车马炮士象`、`新游戏`、`悔棋`、`沙盘`、`上一步`、`下棋盘`、`分析库`
  - `board_detector` 默认约 `1/5` 样本带大小棋盘干扰，但标注只跟大的主棋盘走
  - `board_detector` 默认约 `10%` 样本为几乎无干扰场景，覆盖用户手动框选棋盘的真实用法
  - 背景只做轻度模糊或不模糊，避免“背景过糊、棋盘过亮”这种过于容易的样本

数据生成配置：

- [configs/datasets/piece_classifier_synthetic.yaml](/C:/projects/77xiangqi_recongnize_v2/configs/datasets/piece_classifier_synthetic.yaml)
- [configs/datasets/board_detector_synthetic.yaml](/C:/projects/77xiangqi_recongnize_v2/configs/datasets/board_detector_synthetic.yaml)

## 命令

环境约定遵循 [AGENTS.md](/C:/projects/77xiangqi_recongnize_v2/AGENTS.md)：

```powershell
.venv\Scripts\Activate.ps1
python --version
python -m pip --version
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

运行测试：

```powershell
python -m pytest -q
```

合成棋子分类数据：

```powershell
python .\scripts\build_piece_classifier_dataset.py
```

合成棋盘定位数据：

```powershell
python .\scripts\build_board_detector_dataset.py
```
训练棋盘定位：

```powershell
python .\scripts\train_board_detector.py
```

开启 ImageNet 预训练：

```powershell
python .\scripts\train_board_detector.py --pretrained
```

从当前最佳权重续训：

```powershell
python .\scripts\train_board_detector.py --resume-checkpoint .\artifacts\checkpoints\board_detector\best.pt
```

导出棋盘定位 ONNX：

```powershell
python .\scripts\export_board_detector_onnx.py --checkpoint .\artifacts\checkpoints\board_detector\best.pt
```

训练棋子分类：

```powershell
python .\scripts\train_piece_classifier.py
```

从当前最佳权重续训：

```powershell
python .\scripts\train_piece_classifier.py --resume-checkpoint .\artifacts\checkpoints\piece_classifier\best.pt
```

导出棋子分类 ONNX：

```powershell
python .\scripts\export_piece_classifier_onnx.py --checkpoint .\artifacts\checkpoints\piece_classifier\best.pt
```

运行参考串联：

```powershell
python -m xiangqi_recognize_v2.cli .\some_image.png --board-detector .\artifacts\models\board_detector.onnx --piece-classifier .\artifacts\models\piece_classifier.onnx
```

输出只包含：

- 棋盘 `bbox`
- 90 格分类结果

## 当前明确删除的内容

以下内容已经不再是本仓库主线，已删除或不再保留：

- 传统棋盘定位算法
- `boardloc` 主链
- `empty_filter`
- Python 端 FEN builder / service 真源定位
- 评估 pipeline
- 各类 report 输出
- `inspect_raw_assets.py`
- 对排脚本

## 当前测试范围

测试只保留最小 smoke 级别：

- 棋盘定位训练 smoke
- 棋子分类训练 smoke
- 两个 ONNX 导出 smoke
- 本地参考串联 smoke

当前不再保留老的评估、对排、传统算法测试。

## 文档约定

当前仓库只保留一份设计总览文档：

- [docs/01-project-overview.md](/C:/projects/77xiangqi_recongnize_v2/docs/01-project-overview.md)

如果代码与文档冲突，以当前代码为准，并立即更新这两份文件：

- [README.md](/C:/projects/77xiangqi_recongnize_v2/README.md)
- [docs/01-project-overview.md](/C:/projects/77xiangqi_recongnize_v2/docs/01-project-overview.md)
