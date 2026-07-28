# MemoryBear 记忆基准评测

[English](README.md) | **中文**

**MemoryBear** 是 **RedBear AI** 打造的新一代 AI 记忆系统。它的核心突破在于超越了传统"静态知识存储"的局限：受生物大脑认知机制的启发，MemoryBear 构建了一套贯穿**感知 → 提取 → 关联 → 遗忘**全生命周期的智能知识处理框架。

与把知识当作静态数据来检索的传统记忆工具不同，MemoryBear 模拟海马体的记忆编码、大脑新皮层的知识固化，以及基于突触修剪的遗忘机制——让知识具备类生命的动态演化特性，也让 AI 与用户的关系从被动查询转变为主动的认知协助。

本仓库记录了我们在两个公开长期记忆基准上对 MemoryBear 的评测，并提供复现结果所需的全部内容：我们实际运行的评测代码、MemoryBear 为每道题取回的记忆，以及每个基准一条命令即可复跑"答案生成 + 判分"后半段流水线的脚本。

## 评测结果

| 基准 | 题数 | 主要结果 |
| ---- | ---: | ------- |
| **LongMemEval**（ICLR 2025） | 500 | LLM-judge 准确率 **95.0%** |
| **LoCoMo**（ACL 2024） | 1,986 | LLM-judge 准确率 **91.5%** · token-F1 **0.675** |

两项评测均沿用基准自带的 judge 协议与评分代码，未做任何修改。完整的分题型/
分类别表格、详细配置与发布产物：**[docs/results.md](docs/results.md)**。逐题
发布产物同时镜像在 Hugging Face 数据集：
[redbearai/MemoryBear_eval_result](https://huggingface.co/datasets/redbearai/MemoryBear_eval_result)。

## 快速开始

依赖：[uv](https://docs.astral.sh/uv/) 和一个 LLM API key
（任何 OpenAI 兼容的服务商都可以，端点通过 `LLM_BASE_URL` 设置）。

```bash
cp .env.example .env               # 填入 LLM_API_KEY

uv run src/lme/reproduce.py        # LongMemEval（500 题）
uv run src/locomo/reproduce.py     # LoCoMo（1,986 题）
```

每个脚本直接基于发布的 retrieved memories 运行
（`results/<benchmark>/memorybear/*_retrieved_memories.json(.gz)`，即我们评测时
MemoryBear 逐题返回的上下文原文）：把它们交给 reader 大模型，再按基准原版协议
判分——无需访问 MemoryBear 服务。两个脚本都支持断点续跑，结束时打印分类别成绩表，并接受
`--reader-model` / `--judge-model` / `--workers` / `--force` 参数。

想查看源数据集或运行完整内部流水线：

```bash
uv run data/lme/download_lme.py        # LongMemEval 数据集（约 277 MB，来自 Hugging Face）
uv run data/locomo/download_locomo.py  # LoCoMo 数据集（约 2.8 MB，来自 GitHub）
```

## 文档

| 文档 | 内容 |
| ---- | ---- |
| [docs/pipeline.md](docs/pipeline.md) | 四段式流水线（写入 → 检索 → 作答 → 判分）与设计原则 |
| [docs/results.md](docs/results.md) | 完整成绩表、评测配置、发布产物、验证方法 |
| [src/lme/README.md](src/lme/README.md) | LongMemEval 流水线详解（取记忆 → 作答 → 判分） |
| [src/locomo/README.md](src/locomo/README.md) | LoCoMo 流水线详解（双库检索、rerank、LoCoMo 评分） |

## 仓库结构

```
├── docs/               流水线与结果文档（英文）
├── src/
│   ├── lme/            LongMemEval 评测代码 + reproduce.py
│   └── locomo/         LoCoMo 评测代码 + reproduce.py
│       └── task_eval/  LoCoMo 官方评分代码（逐字 vendor）
├── data/
│   ├── lme/            数据集下载脚本（数据集本身已 gitignore）
│   └── locomo/         数据集下载脚本（数据集本身已 gitignore）
├── results/
│   ├── lme/            按系统分目录的发布产物（memorybear/ 及基线系统）
│   └── locomo/         按系统分目录的发布产物（记忆为 gzip）
├── manifests/          内部写入 manifest（仅占位）
└── .env.example        复现只需一个 LLM_API_KEY
```

## 声明

基准相关组件分别按上游项目的许可使用——LongMemEval（MIT）与 LoCoMo
（CC BY-NC 4.0），详见 [NOTICE](NOTICE)。基准数据集不在本仓库分发——
`data/` 下的脚本会从官方渠道下载。

## 联系我们

关于 MemoryBear 或评测的问题，请联系 RedBear AI 团队。
