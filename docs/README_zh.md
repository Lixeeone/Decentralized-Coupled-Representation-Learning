# DCRL 复现使用指南

本仓库提供论文 **Decentralized Coupled Representation Learning** 的同步 DCRL、异步 A-DCRL、图生成元诊断、Lyapunov 检验、固定流基线和消融实验。主 README 采用正式英文格式；作者、单位、邮箱和原仓库账号均未写入发布文件，BibTeX 全部暂用 `xxx`。

## 最快运行

建议 Python 3.12。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python -m dcrl run --config configs/quickstart.yaml
```

Windows PowerShell 的环境激活命令为 `.venv\Scripts\Activate.ps1`。数值核心只需要 CPU，首次依赖安装后，合成数据实验可以离线运行。

## 实验入口

| 目标 | 命令 |
| --- | --- |
| 快速耦合实验 | `python -m dcrl run --config configs/quickstart.yaml` |
| 合成数据三种子实验 | `python -m dcrl suite --config configs/synthetic_suite.yaml` |
| 算子与抑制项消融 | `python -m dcrl suite --config configs/ablations.yaml` |
| 时间尺度扫描 | `python -m dcrl suite --config configs/timescales.yaml` |
| 图到扩散的诊断 | `python -m dcrl graph --config configs/graph_circle.yaml` |
| Lyapunov 与离散余项 | `python -m dcrl lyapunov --output results/lyapunov` |
| 固定流基线 | `python -m dcrl baselines --config configs/baselines.yaml` |
| 原补充材料参考程序 | `python legacy/reference_diagnostics.py --output_dir results/legacy` |

结果存放于 `results/`。再次运行时使用新的 `--output`，或者明确使用 `--resume`；程序不会静默覆盖已有结果。`summary.json` 给出完成状态与终点指标，`history.csv` 记录轨迹，`aggregate.csv` 汇总多种子均值、样本标准差和发散数量。仓库里的 `examples/validation/` 是实际运行得到的发布验证记录。

## 如何理解复现范围

补充材料本身是精简参考代码，不含原始预训练特征、全部参数和完整实验输出。因此本仓库可以复现已明确配置的算法与数值诊断，但不能声称已逐数值重现论文全部表格。

`legacy/` 保留提供的原参考代码。主包补齐论文算法所述的流形切空间投影、回缩、异步局部空间更新，并增加配置、日志、测试、续跑和基线。`configs/paper_scale/` 保留论文给出的部分样本数、维数、秩和长时程设置；缺失的参数明确记为重构配置。详细对照见 [PAPER_MAPPING.md](PAPER_MAPPING.md)。

ResNet、ViT、VGG、BERT 的入口已经准备好，但需要原特征矩阵或可验证的原模型提取方案。可选特征导出脚本也已提供，尚未在本环境运行神经模型推理。不要用新提取的通用特征替代原特征后，直接宣称论文表格已被复现。

## 指标与匿名发布

所有指标基于原始权重计算，默认不进行 QR 稳定化。秩塌缩不会被补方向的 QR 隐藏；去相关的旋转来自目标特征子空间的 Procrustes 对齐。轨迹目标与终点目标分别记录，不能混为一项指标。

仓库压缩包不包含论文原 PDF、原始对话附件和 `.git` 提交历史。上传 GitHub 后，账号和提交身份由实际发布账号决定；本次交付没有创建线上仓库。正式发表时再更新 `CITATION.bib`。
