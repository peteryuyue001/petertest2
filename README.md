# 🧬 Quant Evolution — 量化 AI 进化系统

基于 DeepSeek 大模型驱动的自动策略迭代框架，让 AI 从「写代码的工具」升级为「策略架构师」。

## 🏗 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   🧠 思考层 (LLM)                    │
│            DeepSeek API — 策略生成与分析              │
│        - 阅读回测反馈  - 生成新策略逻辑               │
│        - 分析失败原因  - 自我修复代码                 │
└──────────────┬──────────────────────┬───────────────┘
               │ 生成策略              │ 反馈结果
┌──────────────▼──────────────────────▼───────────────┐
│              📊 实验层 (Backtest Engine)             │
│           VectorBT — 高速向量化回测                   │
│        - 沙箱安全执行  - 自动错误捕获                 │
│        - 权益曲线生成  - 交易记录输出                 │
└──────────────┬──────────────────────┬───────────────┘
               │ 回测结果              │ 绩效指标
┌──────────────▼──────────────────────▼───────────────┐
│             📈 评估层 (Performance Analytics)         │
│            多维度量化评估模型                          │
│    年化收益 | 最大回撤 | 夏普比率 | 卡玛比率 | IR     │
└─────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
quant_evolution/
├── main.py                      # 主控程序（交互式菜单）
├── config.example.py            # 配置文件模板
├── .env.example                 # API Key 模板
├── requirements.txt             # Python 依赖
│
├── data/
│   └── fetcher.py               # AkShare A股数据获取
│
├── template/
│   └── strategy_base.py         # 策略标准模板（约束 AI）
│
├── engine/
│   ├── backtest.py              # VectorBT 回测封装
│   ├── analyzer.py              # 绩效分析引擎
│   └── sandbox.py               # 沙箱执行器
│
├── llm/
│   ├── deepseek_client.py       # DeepSeek API 客户端
│   └── prompts/
│       ├── strategist.txt       # 策略生成 prompt
│       ├── debugger.txt         # 代码修复 prompt
│       └── analyst.txt          # 回测分析 prompt
│
├── strategy_pool/               # AI 生成的策略文件
├── results/                     # 回测结果 JSON
└── data/cache/                  # 本地数据缓存
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/peteryuyue001/petertest2.git
cd petertest2

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制配置模板
cp config.example.py config.py
cp .env.example .env

# 编辑 config.py，填入你的 DeepSeek API Key
```

> ⚠️ `config.py` 和 `.env` 已在 `.gitignore` 中，不会提交到 GitHub。

### 3. 运行主控程序

```bash
python main.py
```

## 📋 进化流程（阶段 1 — 人工闭环）

```
Step 1: 📊 下载 A股数据 → AkShare 获取沪深300日线
Step 2: 🧠 生成初始策略 → DeepSeek 生成 strategy_v1.py
Step 3: 🔬 运行回测 → VectorBT 自动执行
Step 4: 📈 查看绩效报告 → 收益/回撤/夏普/卡玛
Step 5: 🔄 反馈改进 → 将结果发回 AI，生成 strategy_v2.py
Step 6: 重复 3-5，形成进化循环
```

## 🎯 评估指标

| 指标 | 说明 |
|------|------|
| 总收益率 | 策略累计收益 |
| 年化收益率 | 折算年化回报 |
| 最大回撤 | 历史最大亏损幅度 |
| 夏普比率 | 风险调整后收益 |
| 卡玛比率 | 年化收益/最大回撤 |
| 信息比率 | 相对基准的超额稳定性 |
| 胜率 | 盈利交易占比 |

## 🗺 进化路线

| 阶段 | 目标 | 状态 |
|------|------|------|
| 阶段 1 | 人工闭环 — 手动运行，AI 辅助编程 | 🚧 建设中 |
| 阶段 2 | 自动回测流水线 — 批量策略测试 + Excel 汇总 | 📋 计划中 |
| 阶段 3 | 完全自主进化 — 7×24 自动循环迭代 | 📋 计划中 |

## 📄 License

MIT License — 详见 [LICENSE](LICENSE) 文件。