# SALF-Pi：预训练基线与噪声机制入口

日期：2026-10-09。对应方法：[SALF-CBM，CVPR 2025](https://arxiv.org/abs/2502.20134)。官方上游：[itaybenou/show-and-tell](https://github.com/itaybenou/show-and-tell)，固定提交 `e45f8645afde37a8a7ae66db6987b5a8a53552a0`，保留 MIT 许可和源历史。

**状态：官方预训练基线已完成复验，加噪路径已通过数值和梯度检查。尚未训练 SALF 的噪声生成器，没有新的加噪性能提升结论。** LF-Pi 机制结果见 [NOISE_MECHANISM_REPORT.md](../LF-Pi/NOISE_MECHANISM_REPORT.md)。

## 选择依据

该方法保留空间概念图，先执行 1×1 概念投影，再做 softmax 空间池化和稀疏分类。因此可以直接检查某个概念是否在背景上激活，并在池化之前添加空间扰动。论文报告的稀疏 ImageNet 准确率为 75.32%，相应 LF-CBM 为 71.95%；其 CUB 优势很小，不能把它称为所有数据集上的准确率最优方法。[CVPR 论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Benou_Show_and_Tell_Visually_Explainable_Deep_Neural_Nets_via_Spatially-Aware_CVPR_2025_paper.pdf)

官方仓库提供预训练 ImageNet 检查点与推理 demo，明确说明完整代码仍待发布。这里验证的是已发布模型的适配入口，不能宣称完整重现了官方训练。无需 SAM 即可运行分类和空间概念图；SAM 属于额外 demo 功能。[官方仓库](https://github.com/itaybenou/show-and-tell)

## 已完成实验

| 实验 | 目的 | 结果 | 结论 |
|---|---|---|---|
| 官方预训练模型复验 | 确認预训练检查点与 1,000 类标签对齐 | ImageNet 全 50,000 张：37,780 正确，**75.56%**；top-5 91.634%；NLL 1.0860；NEC 29.742 | 基线推理可运行；没有进行适配或测试选模 |
| 官方输出一致性 | 防止封装改变基线 | logits / 概念值 / 概念图最大绝对误差均 **0.0**；单图输出保持 `[1,1000]` | 保留官方权重、预处理及空间池化；修正仅针对 `squeeze()` 丢失单图批次维度 |
| 空间扰动梯度 | 验证 CE 能穿过冻结的投影、池化和分类层 | μ 头与 σ 头权重梯度范数分别 0.3675 和 0.003914；基线参数没有梯度 | 可微加噪入口有效；这是未训练生成器的工程检查 |

图见 [released_baseline.png](reports/released_baseline.png)。指标与检查点 SHA256 见 [released_baseline.json](reports/released_baseline.json)、[entry_smoke.json](reports/entry_smoke.json)。复验比论文表格高 0.24 个百分点，不能把这个差异记为本项目提升：本项目未训练该模型，原训练选模和具体发布版本未完全公开。

输入处理严格使用官方 `Resize(短边224) → CenterCrop224 → ImageNet Normalize`，不是 LF-Pi 的 Resize256。核对了每个 WNID 与官方 1,000 类编号。骨干为 `microsoft/resnet-50`，输出 `[N,2048,7,7]`；4741 个概念经 resize 后得到 `[N,4741,12,12]` 概念图。

## 下一阶段固定实验入口

`salf_pi/model.py` 中 `SpatialSALF.encode()` 暴露空间特征，`classify_spatial()` 保持官方概念投影、softmax 池化、归一化和分类头。`SpatialGaussian` 用局部特征与全局上下文预测 μ 和 σ，不输入类别标签、真实概念或前景掩膜。当前只是新初始化模块，**工程 smoke 中的 4 图尺度校准不得用于科学训练或测试**；正式实验必须重新实例化并仅用训练特征估计 `feature_scale`。

首轮不进行参数扫描，先执行以下同预算比较：

| 条件 | μ | σ | 要回答的问题 |
|---|---|---|---|
| 固定官方基线 | 0 | 0 | 原始任务和背景依赖水平 |
| 确定性适配对照 | 学习 | 0 | 是否仍是普通均值适配带来提升 |
| 零均值空间随机扰动 | 0 | 学习 | 随机项是否独立改善概念与背景鲁棒性 |
| 均值 + 随机项 | 学习 | 学习 | 随机项相对同容量确定性对照有无增量 |

所有适配条件先冻结同一个基线、投影和分类头，使用同一训练/验证划分、初始化和训练预算，只训练生成器；不把重拟合头或新增投影监督混入第一轮。按验证集选模并检查 σ 是否坍缩，不能靠增大可视化噪声或看测试结果选择参数。噪声加在空间特征图，展示其空间幅度、概念图和前景/背景干预结果；不能将特征热图冒充 RGB 加噪图。

背景诊断可以先用本地 RIVAL10 的人工物体掩膜及 ImageNet WNID，**优先限制到原始 ImageNet 验证图**，避免把骨干/分类器训练图混作新独立测试集。RIVAL10 的原始分割和 ImageNet 来源需形成清单并审计；目前仅确认本地数据与映射存在，尚未完成这个数据协议。替换背景时保证接收图前景像素不变，供体与接收图不同类，并同步检查概念精度、背景替换准确率及前景定位。掩膜默认仅用于评估，若改成监督信号，所有对照必须使用相同的标注预算。

**进入扩大实验的条件：** 随机方案相对确定性对照在背景改变时有稳定增益，相关概念精确率不降低，同时保留原图准确率；通过后再增加独立训练种子。只降低 NEC、只提高 CLIP cosine、或只产生漂亮热图，不足以支持背景去干扰机制。已有概念图也可能吸收伪相关，空间结构本身不是因果保证。

备选 PGCM（ICML 2026）有公开训练代码与可视原型，更适合其 CLEVR-Hans3 物体配置；CUB 的全局特征配置不用于证明空间背景机制。它使用不同的分割和概念标注预算，且当前任务路径包含硬阈值，需先核查梯度路线。FCBM（AAAI 2026）也已审查，但原生空间机制较弱，本轮未创建其子模块或声称完成训练。

## 复现

权重和逐样本预测在 `artifacts/`，不进入 Git。保留上游需求文件，另提供仅用于本入口的 `requirements-entry.txt`；已在 Python 3.11.15 / PyTorch 2.11.0+cu128 环境运行通过，不要求重装官方 demo 的旧 PyTorch 1.13。

```bash
# 从 SALF-Pi 目录运行
python scripts/fetch_released_checkpoint.py
python scripts/smoke_noise_entry.py
python scripts/evaluate_released_baseline.py
```

GPU 实验共用 `D-Pi/artifacts/resources/gpu_work.lock`。脚本使用现有 ImageNet 验证目录，检查点位置默认 `artifacts/pretrained/imagenet/`。本报告完成的是选型、完整预训练基线复验和可训练加噪入口；上述正式适配与 RIVAL10 机制验证留给下一阶段。
