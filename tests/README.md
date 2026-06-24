# Meta-Agent 评测系统

基于 τ2-bench 评测框架格式，针对 Meta-Agent 的核心能力（记忆、工具选择、RAG检索、鲁棒性）设计的测试套件。

## 目录结构

```
tests/
├── README.md              # 本文件
├── conftest.py             # 公共配置（LLM创建、Agent创建）
├── tau_bench/
│   ├── tasks.json          # 15 个测试任务（τ-bench格式）
│   ├── evaluate.py         # 评测主脚本
│   ├── scorer.py           # 评分逻辑（τ2-bench product scoring）
│   └── results/            # 评测结果
├── tau2-bench-main/        # τ2-bench 官方源码（仅作参考）
└── custom/
    └── results/            # 自定义测试结果
```

## 快速开始

```bash
# 安装依赖
pip install websocket-client

# 运行全部 15 个任务
cd F:\desktop\Meta-Agent-with-More-Agents-main
python tests/tau_bench/evaluate.py

# 运行并生成报告
python tests/tau_bench/evaluate.py --report
```

## 测试分类

| 类别 | 任务数 | 测什么 |
|------|--------|--------|
| memory | 3 | 存储→召回、状态追踪、多轮对话记忆 |
| tool_use | 3 | 正确选择天气/RAG/记忆工具 |
| basic | 2 | 问候、能力查询直接回答 |
| complex | 2 | 多步组合（天气+记忆、RAG+记忆） |
| robustness | 3 | 空输入、超长输入、乱码不崩溃 |
| tool_creation | 1 | ToolAgent 动态工具生成 |
| parallel | 1 | 批量记忆存储 |

## 评分方法

采用 τ2-bench 的 **product scoring**：

```
Score = ENV × ACTION × COMMUNICATE × NL_ASSERTIONS
```

| 维度 | 含义 |
|------|------|
| ENV | 系统是否崩溃（无异常=1.0，崩溃=0.0） |
| ACTION | 是否调用了正确的工具 |
| COMMUNICATE | 答案是否是人类可读的（不含原始Agent数据） |
| NL_ASSERTIONS | 答案是否包含预期关键词 |

## 结果解读

- Score ≥ 0.5 → PASS
- Score < 0.5 → FAIL（需检查哪个维度丢分）
