# AGENTS

本仓库默认协作约定如下。

## 1. 环境

- 一律在项目根目录的 `.venv` 中执行 Python、pip、pytest、训练、导出和评估命令。
- 禁止把包装到全局 Python。
- Windows / PowerShell 默认命令：

```powershell
.venv\Scripts\Activate.ps1
python --version
python -m pip --version
```

## 2. 依赖安装

- Python 包统一使用：

```powershell
python -m pip install ...
```

- 不直接使用裸 `pip`，避免装错解释器。
- `torch`、`torchvision` 等 GPU 依赖必须显式注明官方 wheel 源。

## 3. 开发流程

- 改代码前先看 `README.md` 和 `docs/` 对应设计文档。
- 新增模块时，优先补最小测试。
- 改动训练、推理、数据契约时，同步更新文档。

## 4. 测试与静态检查

- 改代码后的默认验证顺序：

```powershell
python -m pytest -q
ruff check .
```

- 若只改了局部模块，至少跑相关测试文件；全量改动跑完整 `pytest`。
- `ruff` 未通过不得视为任务完成（可用 `ruff check . --fix` 修可自动修项）。
- 新增脚本至少要保证：
  - 导入成功
  - 参数解析正常
  - 最小 smoke test 可执行

## 5. 数据与产物

- `data/raw/` 为原始棋盘和棋子素材，纳入 Git，便于同步。
- `data/processed/` 为中间数据层，不纳入 Git。
- `models/`、`artifacts/` 为本地产物层，不纳入 Git。
- 需要纳入版本控制的，还有配置、脚本和说明。

## 6. 代码规范

- 默认 Python 3.11。
- 静态检查使用 `ruff`（见 `COMMANDS.md`）。
- 新代码优先使用类型标注。
- 保持模块职责单一：
  - `pipeline/` 放识别流水线
  - `integrations/` 放外部系统契约
  - `utils/` 放纯工具函数
- 没有明确需求时，不引入新的大型框架。

## 7. 命令约定

- 统一使用项目根目录作为工作目录。
- 命令示例：

```powershell
python -m pytest -q
ruff check .
python -m xiangqi_recognize_v2.cli .\some_image.png
python .\scripts\train_board_detector.py
python .\scripts\train_piece_classifier.py
```

## 8. 提交前检查

- `python -m pytest -q` 通过。
- `ruff check .` 通过。
- 文档与配置同步。
- 不提交 `.venv`、`data/processed/`、模型大文件、评估产物。
