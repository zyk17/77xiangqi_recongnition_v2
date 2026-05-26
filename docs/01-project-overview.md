# 项目总览

## 1. 当前目标

本仓库当前的唯一目标，是为 `77xiangqi` 连线识别链路准备两个可替代 YOLO 的小模型：

1. `board_detector`
   - `MobileNetV3-Small`
   - 输入截图
   - 输出棋盘 `bbox`
2. `piece_classifier`
   - `MobileNetV3-Small`
   - 输入 90 个格子 patch
   - 输出 15 类标签，包含 `empty`

## 2. Python 的职责

Python 当前只做四件事：

- 训练棋盘定位模型
- 训练棋子分类模型
- 导出两个 ONNX
- 提供一份本地可跑的参考串联逻辑

参考串联只到这里：

```text
image -> board bbox -> 90 cells -> 15-class prediction
```

Python 不负责：

- 最终 FEN 真源
- 合法性检测
- 连线状态机
- compare / gating / rescue / emit

## 3. Rust 的职责

Rust 继续负责识别后半段和产品主链：

- 90 格结果转 FEN
- 合法性与状态机
- compare / gating / rescue
- 与前端、平台和 worker 主链集成

对接边界来源：

- `C:\projects\77xiangqi\docs\link`
- `C:\projects\77xiangqi\tauri_backend\src-tauri\src\link`

## 4. 当前分层

- `src/xiangqi_recognize_v2/board_detector/`
  - 棋盘定位模型训练、导出、ONNX 推理
- `src/xiangqi_recognize_v2/piece_classifier/`
  - 15 类分类模型训练、导出、ONNX 推理
- `src/xiangqi_recognize_v2/reference/`
  - 本地参考串联逻辑
- `src/xiangqi_recognize_v2/data_prep/`
  - 合成数据生成

## 5. 数据生成层

当前数据生成层重新保留了历史里已经验证有效的关键能力：

- 固定 `val_piece_skin_ids`
  - 某些皮肤必须稳定放在验证集，不能交给随机切分
- 随机布局
- 棋子光圈增强
  - 默认用于 `piece_classifier`
- 空格增强
  - 默认用于 `piece_classifier`
- 低风险真实干扰
  - `piece_classifier` 与 `board_detector` 共用
- 棋盘场景放置随机化
  - 把合成棋盘放进更大的截图场景里
  - 默认至少一半宽屏场景，另一半窄屏场景
  - 随机顶部 / 底部 / 左右留白
  - 随机整图缩放
  - 背景改为独立的纯色 / 渐变 / 轻纹理 UI 底，不再用棋盘图铺背景
  - 加入更强的 UI 条块、侧边面板、底部面板、浮层和局部遮挡
  - 仅对 `board_detector` 少量加入中文文字干扰，不进入 `piece_classifier` 主分布
  - 约 `1/5` 的 `board_detector` 数据额外带大小棋盘干扰，但只标大的主棋盘
  - 约 `10%` 的 `board_detector` 数据保留为几乎无干扰场景，覆盖用户手动框选棋盘
  - 背景只做轻度模糊或不模糊，不再刻意制造“棋盘特别突出”的容易样本

数据生成入口：

- `scripts/build_piece_classifier_dataset.py`
- `scripts/build_board_detector_dataset.py`

配置文件：

- `configs/datasets/piece_classifier_synthetic.yaml`
- `configs/datasets/board_detector_synthetic.yaml`

## 6. 两类数据的区别

### 6.1 棋子分类数据

输出结构：

```text
data/processed/train_cells/
  train/<label>/
  val/<label>/
```

这条链路必须保留：

- 固定验证集皮肤
- 随机布局
- 棋子光圈增强
- 空格增强
- 低风险真实干扰

### 6.2 棋盘定位数据

输出结构：

```text
data/processed/board_detection/
  train/images
  train/labels
  val/images
  val/labels
```

标签文件：

```json
{"bbox": [x1, y1, x2, y2]}
```

这条链路重点保留：

- 固定验证集皮肤
- 随机布局
- 低风险真实干扰
- 棋盘场景放置随机化

默认不强调：

- 棋子光圈增强
- 空格增强

因为它们对 `board_detector` 不是主收益项。

## 7. 明确删除的东西

本仓库已经明确不再保留：

- 传统棋盘定位算法主链
- `boardloc`
- `empty_filter`
- Python 端 FEN 真源服务定位
- 评估 pipeline
- report 导出
- 对排脚本
- `inspect_raw_assets.py`

## 8. 成功标准

本仓库当前成功标准只有四条：

1. 两个模型都能稳定训练
2. 两个模型都能稳定导出 ONNX
3. 两条数据生成链都能稳定工作
4. 本地参考串联可以稳定跑通 `image -> bbox -> 90格 -> 15分类`
