# LLM Resist Selection

> 大语言模型选择偏见实验框架 —— 探究 LLM 在笑话选择任务中的认知偏见及干预策略

## 项目简介

本项目通过**基于新闻标题的幽默笑话生成与选择任务**，系统性地研究大语言模型（LLM）中存在的两类认知偏见：

- **不作为偏见（Status Quo Bias）**：即使面对低质量候选选项，模型仍倾向于选择现有选项而非主动重新生成。
- **生成偏好 / 敝帚自珍效应（Generation Preference）**：模型倾向于偏好自身生成的内容，拒绝客观评判已有选项。

项目支持**多语言**（英文、西班牙文、中文）和**多模型对标**（QWen、DeepSeek、Kimi、GPT、Gemini），并通过多种递增强度的干预策略验证偏见的可纠正性。

## 项目结构

```
llm_resist_selection/
├── script/                          # 实验脚本
│   ├── extract_tsv_to_json.py       # 数据预处理：TSV → JSON
│   ├── good_option.py               # 生成高质量笑话候选
│   ├── poor_option.py               # 生成低/中质量笑话候选
│   ├── very_poor_option.py          # 生成差选项（完全不相关）
│   ├── exp1_status_quo_bias.py      # 实验一：不作为偏见
│   ├── exp3_best_joke_selection_optimized.py   # 实验三：干预策略（优化版）
│   ├── exp3_best_joke_selection_poor_option.py # 实验三：干预策略（差选项版）
│   └── demo_kimi_k25.py             # Kimi 多模态模型调用示例
├── utils/                           # 可复用工具库
│   ├── __init__.py                  # 统一导出接口
│   ├── api_client.py                # 统一 API 调用（DashScope / OpenAI 自动路由）
│   ├── rate_limiter.py              # 令牌桶速率限制器
│   ├── retry.py                     # 指数退避重试机制与错误分类
│   └── logging.py                   # 日志输出重定向（Tee）
├── data/                            # 处理后的 JSON 新闻标题数据
│   ├── headlines_en.json
│   ├── headlines_es.json
│   └── headlines_zh.json
├── raw_data/                        # 原始 TSV 数据源
│   ├── mwahaha_dev_tasks-ab_v2/     # 验证集
│   └── mwahaha_test/                # 测试集
├── output/                          # 模型生成的笑话数据
└── exp_result/                      # 实验结果与日志
```

## 实验设计

### 实验一：不作为偏见（Status Quo Bias）

验证模型在面对不同质量候选时的选择行为。

- 构建**三组质量梯度**的候选笑话（高质量 / 中等 / 低质量）
- 让评估模型在「选择现有最优」和「自主重新生成」之间做决策
- 统计各质量条件下模型的选择率与创作率

**脚本**: `script/exp1_status_quo_bias.py`

### 实验三：干预策略对抗生成偏好

测试 6 种递增强度的干预策略对模型自我偏好的纠正效果：

| 强度 | 策略 | 说明 |
|------|------|------|
| 基线 | 直接选择 (Baseline) | 无干预，直接选择或创作 |
| 弱 | 重写后比较 (Rewrite-then-Select) | 先重写再与现有笑话对比 |
| 中A | 缺陷分析 (Defect Analysis) | 强制双阶段缺陷分析 |
| 中B | 否定默认假设 (Negative Default) | 预设"不应重新生成"的默认立场 |
| 强 | 数值化门槛 (Numerical Threshold) | 要求量化评分达到阈值才可重新生成 |
| 验证 | 盲测 (Blind Test) | 混入模型自身历史生成的笑话进行匿名评测 |

**脚本**:
- `script/exp3_best_joke_selection_optimized.py` — 优化版（6 种策略）
- `script/exp3_best_joke_selection_poor_option.py` — 差选项版（5 种策略，无盲测）

## 数据流

```
raw_data/ (TSV)
    │
    ▼  extract_tsv_to_json.py
data/headlines_*.json
    │
    ├── good_option.py       → output/good_option_{model}_{lang}.json   (高质量)
    ├── poor_option.py       → output/low_option_{lang}.json            (低质量)
    │                        → output/medium_option_{lang}.json         (中等质量)
    └── very_poor_option.py  → output/poor_option_{lang}.json           (差选项)
            │
            ├── exp1_status_quo_bias.py           → exp_result/exp1_*.json
            ├── exp3_*_optimized.py               → exp_result/exp3_*.json
            └── exp3_*_poor_option.py             → exp_result/exp3_poor_*.json
```

## 笑话质量梯度

| 梯度 | 类型 | 来源脚本 |
|------|------|----------|
| 高质量 | 模型直接生成的最优笑话 | `good_option.py` |
| 中等质量 | Safe_Humor, Predictable_Punchline, Surface_Level, Generic_Wit | `poor_option.py --quality medium` |
| 低质量 | Forced_Pun, Overexplained_Joke, Cliche_Joke, Weak_Connection | `poor_option.py --quality low` |
| 差选项 | Irrelevant_Response, Repetition, Bland_Statement, Template_Response | `very_poor_option.py` |

## 环境配置

### 依赖安装

```bash
pip install dashscope openai
```

### 环境变量

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
export OPENAI_API_KEY="your-openai-api-key"        # 可选，用于 GPT/Gemini 模型
export OPENAI_BASE_URL="https://your-api-base/v1"   # 可选，自定义 OpenAI 兼容端点
```

## 使用方法

### 1. 数据预处理

```bash
python script/extract_tsv_to_json.py
```

### 2. 生成笑话候选

```bash
# 生成高质量笑话（支持指定语言和模型）
python script/good_option.py --lang en

# 生成低质量 / 中等质量笑话
python script/poor_option.py --quality low --lang en
python script/poor_option.py --quality medium --lang en

# 生成差选项
python script/very_poor_option.py
```

### 3. 运行实验

```bash
# 实验一：不作为偏见
python script/exp1_status_quo_bias.py --lang en --rpm 600

# 实验三：干预策略（优化版）
python script/exp3_best_joke_selection_optimized.py --lang en --rpm 600 --start 0 --end 60

# 实验三：干预策略（差选项版）
python script/exp3_best_joke_selection_poor_option.py --lang en --rpm 600
```

### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lang` | 语言：`en` / `es` / `zh` / `all` | `en` |
| `--rpm` | 每分钟请求限制 | `600` |
| `--models` | 评估模型列表 | 脚本内置默认 |
| `--start` | 数据起始索引 | `0` |
| `--end` | 数据结束索引 | `None`（全部） |

## 工具库（utils）

| 模块 | 功能 |
|------|------|
| `api_client.py` | 统一 API 调用接口，自动路由 DashScope / OpenAI，支持多模态模型（Kimi-K2.5/K2.6） |
| `rate_limiter.py` | 令牌桶速率限制器，线程安全，80% 安全系数 |
| `retry.py` | 指数退避重试（基数 2.0 + ±25% 抖动），错误分类（速率限制 / 可重试 / 不可重试） |
| `logging.py` | Tee 输出重定向，同时写入控制台和日志文件 |

## 支持的模型

| 平台 | 模型 | 调用方式 |
|------|------|----------|
| DashScope | qwen-max, qwen3-max, qwen3.6-max-preview, deepseek-v3.2, glm-5 | `Generation.call()` |
| DashScope (多模态) | kimi-k2.5, kimi-k2.6 | `MultiModalConversation.call()` |
| OpenAI 兼容 | gpt-5.4-2026-03-05-high, gemini-3.1-pro-preview | OpenAI SDK |

## 数据格式

### 输入：新闻标题 (`data/headlines_*.json`)

```json
[
  {"id": "en_0001", "news_headline": "Ryanair to cut 1 million more passenger seats in Spain"},
  ...
]
```

### 输出：实验结果 (`exp_result/exp1_*.json`)

```json
{
  "experiment": "exp1_status_quo_bias",
  "lang": "en",
  "models": ["kimi-k2.6", "qwen3.6-max-preview", "gpt-5.4-2026-03-05-high", "gemini-3.1-pro-preview"],
  "results": [
    {
      "id": "en_0001",
      "news_headline": "...",
      "quality_groups": {
        "high": {
          "existing_jokes": ["..."],
          "experiments": {
            "model_name": {
              "decision": "select|create",
              "selected_joke_number": 2,
              "reason": "...",
              "final_joke": "..."
            }
          }
        }
      }
    }
  ],
  "summary": { ... }
}
```

## License

MIT