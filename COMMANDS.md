# 命令大全

本文档汇总当前仓库所有常用命令。

约定：

- 所有命令都在项目根目录执行：`C:\projects\77xiangqi_recongnize_v2`
- 默认使用项目内 `.venv`
- 默认命令行为 PowerShell

## 1. 环境

激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
```

检查 Python / pip：

```powershell
python --version
python -m pip --version
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

单独安装或升级 `ruff`：

```powershell
python -m pip install -U ruff
```

## 2. 代码检查

运行 `ruff`：

```powershell
ruff check .
```

自动修复可修项：

```powershell
ruff check . --fix
```

运行测试：

```powershell
python -m pytest -q
```

只跑某个测试文件：

```powershell
python -m pytest -q .\tests\test_build_dataset_scripts.py
```

## 3. 数据生成
### 3.1 合成棋子分类数据

使用默认配置生成：

```powershell
python .\scripts\build_piece_classifier_dataset.py
```

指定配置文件生成：

```powershell
python .\scripts\build_piece_classifier_dataset.py --config .\configs\datasets\piece_classifier_synthetic.yaml
```

默认配置文件：

```text
configs/datasets/piece_classifier_synthetic.yaml
```

生成输出目录默认是：

```text
data/processed/train_cells
```

### 3.2 合成棋盘定位数据

使用默认配置生成：

```powershell
python .\scripts\build_board_detector_dataset.py
```

指定配置文件生成：

```powershell
python .\scripts\build_board_detector_dataset.py --config .\configs\datasets\board_detector_synthetic.yaml
```

默认配置文件：

```text
configs/datasets/board_detector_synthetic.yaml
```

生成输出目录默认是：

```text
data/processed/board_detection
```

## 4. 训练棋子分类模型

使用默认配置训练：

```powershell
python .\scripts\train_piece_classifier.py
```

指定训练配置：

```powershell
python .\scripts\train_piece_classifier.py --config .\configs\train\piece_classifier.yaml
```

覆盖训练轮数、batch、学习率：

```powershell
python .\scripts\train_piece_classifier.py --epochs 20 --batch-size 256 --lr 0.0005
```

指定设备：

```powershell
python .\scripts\train_piece_classifier.py --device cuda
```

改训练/验证目录：

```powershell
python .\scripts\train_piece_classifier.py --train-dir .\data\processed\train_cells\train --val-dir .\data\processed\train_cells\val
```

改 checkpoint 输出目录：

```powershell
python .\scripts\train_piece_classifier.py --checkpoint-dir .\artifacts\checkpoints\piece_classifier
```

从 checkpoint 继续训练：

```powershell
python .\scripts\train_piece_classifier.py --resume-checkpoint .\artifacts\checkpoints\piece_classifier\best.pt
```

## 5. 导出棋子分类 ONNX

使用默认输出路径导出：

```powershell
python .\scripts\export_piece_classifier_onnx.py --checkpoint .\artifacts\checkpoints\piece_classifier\best.pt
```

指定输出路径：

```powershell
python .\scripts\export_piece_classifier_onnx.py --checkpoint .\artifacts\checkpoints\piece_classifier\best.pt --output .\artifacts\models\piece_classifier.onnx
```

## 6. 训练棋盘定位模型

使用默认配置训练：

```powershell
python .\scripts\train_board_detector.py
```

指定训练配置：

```powershell
python .\scripts\train_board_detector.py --config .\configs\train\board_detector.yaml
```

覆盖训练轮数、batch、学习率：

```powershell
python .\scripts\train_board_detector.py --epochs 20 --batch-size 64 --lr 0.0005
```

指定设备：

```powershell
python .\scripts\train_board_detector.py --device cuda
```

显式开启或关闭 ImageNet 预训练：

```powershell
python .\scripts\train_board_detector.py --pretrained
python .\scripts\train_board_detector.py --no-pretrained
```

改训练/验证目录：

```powershell
python .\scripts\train_board_detector.py --train-images .\data\processed\board_detection\train\images --train-labels .\data\processed\board_detection\train\labels --val-images .\data\processed\board_detection\val\images --val-labels .\data\processed\board_detection\val\labels
```

改 checkpoint 输出目录：

```powershell
python .\scripts\train_board_detector.py --checkpoint-dir .\artifacts\checkpoints\board_detector
```

从 checkpoint 继续训练：

```powershell
python .\scripts\train_board_detector.py --resume-checkpoint .\artifacts\checkpoints\board_detector\best.pt
```

## 7. 导出棋盘定位 ONNX

使用默认输出路径导出：

```powershell
python .\scripts\export_board_detector_onnx.py --checkpoint .\artifacts\checkpoints\board_detector\best.pt
```

指定输出路径：

```powershell
python .\scripts\export_board_detector_onnx.py --checkpoint .\artifacts\checkpoints\board_detector\best.pt --output .\artifacts\models\board_detector.onnx
```

## 8. 参考串联

运行本地参考串联：

```powershell
python -m xiangqi_recognize_v2.cli .\some_image.png --board-detector .\artifacts\models\board_detector.onnx --piece-classifier .\artifacts\models\piece_classifier.onnx
```

显式指定标签文件：

```powershell
python -m xiangqi_recognize_v2.cli .\some_image.png --board-detector .\artifacts\models\board_detector.onnx --piece-classifier .\artifacts\models\piece_classifier.onnx --label-map .\configs\label_map.json
```

输出内容只包含：

- 棋盘 `bbox`
- 90 格分类结果

## 9. 评估 Raw Cases

评估整条参考串联在 `data/eval_cases/*/raw.xx` 上的表现：

```powershell
python .\scripts\eval_reference_pipeline_cases.py
```

指定模型和评估目录：

```powershell
python .\scripts\eval_reference_pipeline_cases.py --board-detector .\artifacts\models\board_detector.onnx --piece-classifier .\artifacts\models\piece_classifier.onnx --eval-root .\data\eval_cases
```

## 10. 调试单个 Raw Case

生成棋盘 ROI、90 格网线和预测标签可视化：

```powershell
python .\scripts\debug_reference_pipeline_case.py .\data\eval_cases\case07
```

默认输出到：

```text
artifacts/debug/<case_name>/
```

其中会包含：

- `overlay.png`
- `board_crop.png`
- `pipeline_result.json`

## 11. 评估 clipped cases

评估 `piece_classifier.onnx` 在 `data/eval_cases/*/clipped.xx` 上的逐格表现：

```powershell
python .\scripts\eval_piece_classifier_cases.py
```

指定模型、标签和评估目录：

```powershell
python .\scripts\eval_piece_classifier_cases.py --model .\artifacts\models\piece_classifier.onnx --label-map .\configs\label_map.json --eval-root .\data\eval_cases
```

## 12. 常用组合流程

### 9.1 从零开始跑棋子分类

```powershell
.venv\Scripts\Activate.ps1
python .\scripts\build_piece_classifier_dataset.py
python .\scripts\train_piece_classifier.py
python .\scripts\export_piece_classifier_onnx.py --checkpoint .\artifacts\checkpoints\piece_classifier\best.pt
```

### 9.2 从零开始跑棋盘定位

```powershell
.venv\Scripts\Activate.ps1
python .\scripts\build_board_detector_dataset.py
python .\scripts\train_board_detector.py
python .\scripts\export_board_detector_onnx.py --checkpoint .\artifacts\checkpoints\board_detector\best.pt
```

### 9.3 完整本地验证

```powershell
.venv\Scripts\Activate.ps1
ruff check .
python -m pytest -q
python -m xiangqi_recognize_v2.cli .\some_image.png --board-detector .\artifacts\models\board_detector.onnx --piece-classifier .\artifacts\models\piece_classifier.onnx
```

## 13. 相关配置文件

训练配置：

- `configs/train/piece_classifier.yaml`
- `configs/train/board_detector.yaml`

数据生成配置：

- `configs/datasets/piece_classifier_synthetic.yaml`
- `configs/datasets/board_detector_synthetic.yaml`

标签顺序：

- `configs/label_map.json`

## 14. 产物位置

默认训练 checkpoint：

- `artifacts/checkpoints/piece_classifier`
- `artifacts/checkpoints/board_detector`

默认 ONNX 输出：

- `artifacts/models/piece_classifier.onnx`
- `artifacts/models/board_detector.onnx`

默认合成数据输出：

- `data/processed/train_cells`
- `data/processed/board_detection`
