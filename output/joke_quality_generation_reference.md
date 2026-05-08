# 不同质量等级笑话生成资料整理

> 基于 `poor_option.py` 及项目相关脚本，整理笑话质量梯度体系、提示词策略、参数配置与工程实现。

---

## 一、笑话质量梯度总览

项目定义了 **四级质量梯度**，分别由不同脚本生成：

| 梯度 | 子类型（每级4种） | 来源脚本 | 输出文件 |
|------|-------------------|----------|----------|
| **高质量** | 模型直接生成的最优笑话 | `good_option.py` | `good_option_{model}_{lang}.json` |
| **中等质量** | Safe_Humor, Predictable_Punchline, Surface_Level, Generic_Wit | `poor_option.py --quality medium` | `medium_option_{lang}.json` |
| **低质量** | Forced_Pun, Overexplained_Joke, Cliche_Joke, Weak_Connection | `poor_option.py --quality low` | `low_option_{lang}.json` |
| **差选项** | Irrelevant_Response, Repetition, Bland_Statement, Template_Response | `very_poor_option.py` | `poor_option_{lang}.json` |

---

## 二、各质量等级的笑话子类型详解

### 2.1 低质量（Low）—— "尝试搞笑但失败"

核心目标：**可识别为笑话尝试，但不应让人真正笑出来**。

| 类型 | 英文名 | 失败模式 | 质量锚定标尺 |
|------|--------|----------|-------------|
| 强行谐音梗 | Forced_Pun | 双关语生硬、不自然、刻意为之 | 读者反应：翻白眼或叹气；形容词："尬"、"硬凹"、"painful"、"try-hard" |
| 过度解释 | Overexplained_Joke | 先讲笑话再立刻解释为什么好笑，毁掉幽默 | "如果不解释还凑合"，但基础笑话本身也最多算平庸 |
| 老套格式 | Cliche_Joke | 使用极度陈旧的笑话模板 | "像从90年代笑话书里翻出来的"，零惊喜感 |
| 牵强联系 | Weak_Connection | 与标题勉强相关，笑点逻辑牵强附会 | "这跟标题有什么关系？"，像花了5秒随便想的 |

**文化适配要点**：
- **英文**：标准格式如 "Why did X cross the road?", "X walks into a bar..."
- **西班牙文**：本土化格式如 Jaimito/Pepito 系列、"¿Cuál es el colmo de...?"、"¿Qué le dice un X a un Y?"、"¿En qué se parece X a Y?"
- **中文**：小明系列、"为什么X要过马路？"、"X和Y有什么区别？"等

### 2.2 中等质量（Medium）—— "还不错但不值得分享"

核心目标：**明显好于低质量，但明显不如高质量**，是"公司年会级别"的幽默。

| 类型 | 英文名 | 特征 | 质量锚定标尺 |
|------|--------|------|-------------|
| 安全幽默 | Safe_Humor | 温和有趣但安全平庸、无害、容易遗忘 | 公司年会上听到的——礼貌微笑但不会大笑，第二天没人会记得 |
| 可预测笑点 | Predictable_Punchline | 铺垫不错但笑点显而易见 | "嗯，意料之中"——能猜到走向，不够出人意料 |
| 表面观察 | Surface_Level | 只利用标题最表面信息，缺乏深度 | "任何人用10秒钟就能想到"——合格但缺乏洞察力 |
| 通用机智 | Generic_Wit | 有一定机智但非特定于该标题 | "套模板"感——只是换了个话题，缺乏原创火花 |

### 2.3 高质量（High）

由 `good_option.py` 直接生成，提示词要求"简洁、有创意、真正有趣"（concise, creative, genuinely funny），不做质量降级约束。

### 2.4 差选项（Very Poor）

由 `very_poor_option.py` 生成，完全不是笑话：

| 类型 | 英文名 | 描述 |
|------|--------|------|
| 不相关回复 | Irrelevant_Response | 与标题完全无关的随机内容 |
| 简单重复 | Repetition | 对标题的简单复述或改写 |
| 干燥陈述 | Bland_Statement | 与主题相关但完全无幽默的百科式事实 |
| 模板回复 | Template_Response | "太有趣了！"等通用机械式回复 |

---

## 三、提示词工程策略

### 3.1 对比约束（Contrastive Constraint）

每种类型的提示词都注入了**对比约束文案**，通过锚定"失败标准"来引导模型输出到目标质量区间：

**低质量约束模板**：
> IMPORTANT quality constraint: This joke should NOT make anyone genuinely laugh. If a reader's reaction is anything more than an eye-roll or a groan, it's too good. Aim for a joke that people would describe as "painful" or "try-hard".

**中等质量约束模板**：
> IMPORTANT quality constraint: This joke should be DECENT but NOT worth sharing with friends. It's the kind of joke you'd hear at a corporate event — inoffensive, mildly clever, but no one would remember it the next day. It should be clearly better than a terrible joke, but clearly worse than a genuinely witty one.

### 3.2 失败标尺（Failure Anchors）

| 质量等级 | 失败标尺关键词 |
|----------|---------------|
| 低质量 | "painful"、"try-hard"、"尬"、"硬凹"、"90年代笑话书"、"eye-roll or groan" |
| 中等质量 | "corporate event"、"公司年会"、"not worth sharing"、"不值得分享"、"模板感"、"套路" |

### 3.3 提示词中的关键指令

- **格式约束**：`Only return the joke and nothing else. Do NOT include any explanation.`
- **语言约束**（非英文）：`La respuesta debe estar en español.` / `回复必须使用中文。`
- **Overexplained 特殊指令**：要求先讲笑话再紧跟过度解释，以"解释杀死幽默"的方式降质

---

## 四、模型参数配置策略

通过 **temperature 和 top_p 的差异化配置**，在系统层面强化质量梯度分离：

| 质量等级 | temperature | top_p | 设计意图 |
|----------|-------------|-------|----------|
| **低质量** | 1.2（高） | 0.95 | 高随机性 → 增加不连贯感和意外失败 |
| **中等质量** | 0.6（低） | 0.7 | 低随机性 → 保持流畅但平庸、安全、可预测 |
| **高质量** | 0.85 | 0.9 | 平衡创意与稳定性 |
| **差选项** | 0.85 | 0.9 | 标准参数（质量主要靠提示词控制） |

**设计逻辑**：
- 低质量用高 temperature 让模型"容易犯错"，输出更随机、更可能生成不自然的内容
- 中等质量用低 temperature 让模型"安全保守"，输出流畅但缺乏创意突破

---

## 五、工程实现架构

### 5.1 数据流

```
raw_data/ (TSV)
    │ extract_tsv_to_json.py
    ▼
data/headlines_{lang}.json  (输入：id + news_headline)
    │
    ├── good_option.py       → output/good_option_{model}_{lang}.json
    ├── poor_option.py --quality low    → output/low_option_{lang}.json
    ├── poor_option.py --quality medium → output/medium_option_{lang}.json
    └── very_poor_option.py  → output/poor_option_{lang}.json
            │
            ├── exp1_status_quo_bias.py     → exp_result/exp1_*.json
            └── exp3_*                      → exp_result/exp3_*.json
```

### 5.2 核心函数调用链（poor_option.py）

```
main (argparse)
  ├── option_gen_all(quality_level)          # 全语言批量处理
  │     └── process_headlines(...)           # 单语言文件处理
  │           └── generate_all_options(...)  # 单条新闻 → 4种笑话
  │                 └── generate_option(...) # 单种类型笑话生成
  │                       ├── PROMPTS[lang][option_type]  # 提示词选择
  │                       ├── _rate_limiter.acquire()      # 速率限制
  │                       ├── Generation.call(...)          # API调用
  │                       └── 重试逻辑（指数退避）           # 错误处理
  └── process_headlines(...)                  # 单语言处理
```

### 5.3 健壮性机制

| 机制 | 实现方式 | 参数 |
|------|----------|------|
| **速率限制** | 令牌桶算法 (`TokenBucketRateLimiter`) | RPM=600, 安全系数80%, 超时120s |
| **指数退避重试** | `RetryConfig` + 随机抖动 | max_retries=3, base_delay=2s, max_delay=30s, ±25% jitter |
| **错误分类** | `is_rate_limit_error()` + `is_retryable_error()` | 状态码400/429/503/5xx + 关键字匹配 |
| **断点续传** | 基于已处理ID集合的跳过逻辑 | 加载已存在输出文件，跳过已处理项 |
| **自动保存** | 每处理10条写入一次文件 | 防止中断导致数据丢失 |

### 5.4 使用的模型

| 脚本 | 生成模型 | 评估模型 |
|------|----------|----------|
| `poor_option.py` | deepseek-v3.2 | — |
| `good_option.py` | 可配置（qwen-max, kimi-k2.5 等） | — |
| `very_poor_option.py` | deepseek-v3.2 | — |
| `exp1_status_quo_bias.py` | — | kimi-k2.6, qwen3.6-max, gpt-5.4, gemini-3.1 |
| `exp3_*.py` | — | 同上 |

---

## 六、输入输出格式

### 输入格式（headlines_{lang}.json）

```json
[
  {"id": "en_0001", "news_headline": "Ryanair to cut 1 million more passenger seats in Spain"},
  {"id": "zh_0001", "news_headline": "婚姻登记'全国通办'后首个七夕节，众多城市新人领证数量创'小高峰'"}
]
```

### 输出格式（low_option_{lang}.json / medium_option_{lang}.json）

```json
[
  {
    "id": "en_0001",
    "news_headline": "Ryanair to cut 1 million more passenger seats in Spain",
    "Forced_Pun": "生成的强行谐音梗笑话...",
    "Overexplained_Joke": "生成的过度解释笑话...",
    "Cliche_Joke": "生成的老套格式笑话...",
    "Weak_Connection": "生成的牵强联系笑话..."
  }
]
```

---

## 七、实验中的使用方式

### 实验一：不作为偏见（Status Quo Bias）

将三组质量的笑话分别作为候选，让评估模型在"选择现有最优"和"自主重新生成"之间做决策。通过对比**高/中/低质量组的选择率差异**，检验模型是否存在不作为偏见（即即使面对低质量选项也倾向于选择而非重新生成）。

### 实验三：干预策略对抗生成偏好

在差选项/低质量条件下，测试5-6种递增强度的干预策略：

| 强度 | 策略名 | 核心机制 |
|------|--------|----------|
| 基线 | 直接选择 (Baseline) | 无干预 |
| 弱 | 重写后比较 (Rewrite-then-Select) | 先重写再比较 |
| 中A | 缺陷分析 (Defect Analysis) | 强制双阶段缺陷分析 |
| 中B | 否定默认假设 (Negative Default) | 预设"不应重新生成"的默认立场 |
| 强 | 数值化门槛 (Numerical Threshold) | 量化评分达到阈值才可重新生成 |
| 验证 | 盲测 (Blind Test) | 混入模型自身历史生成笑话的匿名评测 |

---

## 八、CLI 使用方式

```bash
# 生成低质量笑话（全语言）
python script/poor_option.py --quality low

# 生成中等质量笑话（仅英文）
python script/poor_option.py --quality medium --lang en

# 生成高质量笑话
python script/good_option.py --lang en --model qwen-max

# 生成差选项
python script/very_poor_option.py --lang en
```

---

## 九、学术研究背景与文献支撑

### 9.1 LLM 幽默生成能力的研究现状

#### 核心发现

近年来多项研究系统评估了 LLM 的幽默生成能力：

- **LLM 幽默水平约等于中低水平人类**：Sakabe et al. (2025) 在日本 Oogiri 幽默任务中发现，GPT-4.1、Gemini 2.5 Pro、Claude Sonnet 4 等模型的幽默生成能力相当于人类中低水平，其中 Gemini 2.5 Pro 表现最佳。LLM 的主要弱点在于 **Empathy（共情）** 维度。
  - 论文：*Assessing the Capabilities of LLMs in Humor: A Multi-dimensional Analysis of Oogiri Generation and Evaluation* (arXiv:2511.09133)

- **LLM 生成幽默趋于公式化和安全**：Jentzsch & Kersting (2023) 发现 ChatGPT 虽然在对话中可能令人发笑，但输出往往是 **formulaic（公式化）** 或 **overly safe（过于安全）**，缺乏真正让人发笑的火花。这与本项目中等质量笑话的设计理念（"公司年会级别"）高度吻合。

- **多技能协同可提升幽默质量**：Kim & Chilton (2025) 提出 HumorSkills 方法，通过**观察→发散→生成→排序**四阶段流程，结合认知、社会和创意技能，使 AI 生成的幽默与人类顶级作品在统计上无显著差异（5分制仅差0.08分）。
  - 论文：*AI Humor Generation: Cognitive, Social and Creative Skills for Effective Humor* (arXiv:2502.07981)

#### 幽默理论基础

| 理论 | 核心观点 | 与项目的关联 |
|------|----------|-------------|
| **Suls 两阶段模型** | 幽默 = 察觉不协调 + 成功解决不协调 | 低质量笑话故意制造「察觉但无法解决」的尴尬感 |
| **良性违反理论** (Benign Violation) | 幽默 = 期望被违反但不构成威胁 | 中等质量笑话的违反太温和，差选项的违反与幽默无关 |
| **幽默评估六维度** | Novelty, Clarity, Relevance, Empathy, Intelligence, Overall Funniness | 低质量笑话在 Novelty 和 Intelligence 维度上刻意降低 |

#### 为什么 LLM 难以生成「刚好差到位」的笑话？

生成特定质量等级的笑话比生成最优笑话更难，因为需要精确控制幽默的多个维度。项目采用的策略是：
1. **提示词层面**：注入对比约束和失败标尺，明确告知模型目标质量区间
2. **参数层面**：通过 temperature/top_p 差异化配置制造系统性差异
3. **类型层面**：将每个质量等级拆分为4种特定的失败模式，避免模型"滑向高质量"

### 9.2 Temperature 参数与创造力的关系

#### 学术共识

Peeperkorn et al. (2024) 在论文 *Is Temperature the Creativity Parameter of Large Language Models?* 中系统研究了 temperature 对创造力的影响：

**核心发现**：
- Temperature 与 **novelty（新颖性）呈弱正相关**
- Temperature 与 **incoherence（不连贯性）呈中等正相关**
- Temperature 与 **cohesion（内聚性）** 和 **typicality（典型性）** 无显著关系
- **结论**：Temperature 对创造力的影响远比"创造力参数"这一标签所暗示的更微妙和有限

**论文**：arXiv:2405.00492, University of Kent & Leiden University & University of Waterloo

#### 与项目设计的关联

| 研究发现 | 项目中的应用 |
|----------|-------------|
| 高 temperature → 更多新颖但不连贯的输出 | 低质量笑话用 temperature=1.2 → 增加"不连贯感"和"意外失败" |
| 低 temperature → 更可预测、更典型的输出 | 中等质量笑话用 temperature=0.6 → 保持"流畅但平庸" |
| Temperature 主要影响采样多样性而非语义空间 | 质量控制不能仅靠 temperature，还需提示词约束配合 |

#### Temperature 的数学原理

```
softmax(z)_i = exp(z_i / t) / Σ_j exp(z_j / t)
```

- **t → 0**：概率分布趋于确定性，总选择最高概率 token（贪婪解码）
- **t = 1**：使用原始 logits 概率
- **t > 1**：概率分布趋于均匀，低概率 token 获得更多机会
- **t → ∞**：完全随机选择

### 9.3 LLM 认知偏见研究

#### 不作为偏见（Status Quo Bias）

**定义**：人倾向于维持现状，即使改变可能带来更好的结果。在 LLM 语境中，表现为模型倾向于选择已有选项，而非生成新内容。

**关键研究**：

1. **BiasBuster 框架** — Echterhoff et al. (2024)
   - 在高风险决策（大学录取）场景中系统测试 LLM 的认知偏见
   - 不作为偏见测试：比较中性提示与含"现状"信息的提示，观察模型是否过度倾向默认选项
   - 评估指标：默认选项被选中的比例是否显著高于 1/N（均匀分布）
   - 论文：*Cognitive Bias in High-Stakes Decision-Making with LLMs* (arXiv:2403.00811, UCSD)

2. **30种认知偏见大规模评估** — Malberg et al. (2024)
   - 覆盖20个 SOTA LLM（1B–175B+ 参数）、200种决策场景、30,000个测试用例
   - 提出系统化框架：模板 → 场景 → 测试用例 → 评估
   - 论文：*A Comprehensive Evaluation of Cognitive Biases in LLMs* (arXiv:2410.15413, TU Munich)

3. **新闻推荐中的不作为偏见** — Lyu et al. (2024)
   - 在新闻推荐系统中，LLM 倾向于偏好之前见过的熟悉内容
   - 论文：*Cognitive Biases in Large Language Models for News Recommendation* (University of Amsterdam)

#### 生成偏好 / 敝帚自珍效应（Generation Preference / IKEA Effect）

**定义**：人倾向于高估自己参与创造的事物的价值。在 LLM 语境中，模型倾向于偏好自身生成的内容而非外部提供的同等质量内容。

**与项目的关联**：
- 实验三通过盲测（Blind Test）验证此效应：将模型自身历史生成的笑话匿名混入候选集
- 如果模型无法识别出自己的笑话却依然偏好"重新生成"，则支持生成偏好假说

### 9.4 认知去偏见（Cognitive Debiasing）干预策略

#### 学术方法分类

Lyu et al. (2025) 在 *Cognitive Debiasing Large Language Models for Decision-Making* 中提出 **Self-Debiasing** 三步法：

1. **偏见判定（Bias Determination）**：识别提示词中可能诱导偏见的元素
2. **偏见分析（Bias Analysis）**：分析偏见如何影响决策过程
3. **认知去偏（Cognitive Debiasing）**：迭代式修正提示词以消除偏见影响

**论文**：arXiv:2504.04141, University of Amsterdam & Shandong University

#### 与项目干预策略的对应

| 项目策略 | 学术对应 | 去偏机制 |
|----------|----------|----------|
| 重写后比较 (Rewrite-then-Select) | Prompt refinement through feedback | 通过重写产生对照，打破对现有选项的锚定 |
| 缺陷分析 (Defect Analysis) | Self-debiasing: Bias Analysis step | 强制模型显式分析候选缺陷，克服默认接受倾向 |
| 否定默认假设 (Negative Default) | Framing manipulation | 通过反转默认框架（"不应重新生成"）降低生成偏好 |
| 数值化门槛 (Numerical Threshold) | Quantitative calibration | 要求数值化评估，减少主观判断中的偏见空间 |
| 盲测 (Blind Test) | Blind evaluation protocol | 消除来源信息，测试纯粹的内容质量判断 |

### 9.5 对比约束提示词工程

#### 理论依据

**Contrastive In-Context Learning** (Zheng et al., 2024) 提出通过对比示例引导 LLM 输出，核心思想是同时提供正面和负面示例来精确描述期望的输出特征。

- 论文：*Customizing Language Model Responses with Contrastive In-Context Learning* (AAAI 2024, arXiv:2401.17390)

**项目中的应用**：
- 低质量提示词中的"should NOT make anyone genuinely laugh"是负面约束
- 中等质量提示词中的"clearly better than a terrible joke, but clearly worse than a genuinely witty one"是双向对比约束
- 通过失败标尺（"90年代笑话书"、"公司年会"）提供具体锚点

---

## 十、参考文献

1. Peeperkorn, M., Kouwenhoven, T., Brown, D., & Jordanous, A. (2024). *Is Temperature the Creativity Parameter of Large Language Models?* arXiv:2405.00492.
2. Echterhoff, J., Liu, Y., Alessa, A., McAuley, J., & He, Z. (2024). *Cognitive Bias in High-Stakes Decision-Making with LLMs.* arXiv:2403.00811.
3. Malberg, S., Poletukhin, R., Schuster, C. M., & Groh, G. (2024). *A Comprehensive Evaluation of Cognitive Biases in LLMs.* arXiv:2410.15413.
4. Lyu, Y., Ren, S., Feng, Y., et al. (2025). *Cognitive Debiasing Large Language Models for Decision-Making.* arXiv:2504.04141.
5. Sakabe, R., Kim, H., Hirasawa, T., & Komachi, M. (2025). *Assessing the Capabilities of LLMs in Humor: A Multi-dimensional Analysis.* arXiv:2511.09133.
6. Kim, S. & Chilton, L. B. (2025). *AI Humor Generation: Cognitive, Social and Creative Skills for Effective Humor.* arXiv:2502.07981.
7. Jentzsch, S. & Kersting, K. (2023). *ChatGPT is fun, but it is not funny! Humor is still challenging Large Language Models.* WASSA @ ACL 2023.
8. Zheng, C., et al. (2024). *Customizing Language Model Responses with Contrastive In-Context Learning.* AAAI 2024, arXiv:2401.17390.
9. Lyu, Y., et al. (2024). *Cognitive Biases in Large Language Models for News Recommendation.* University of Amsterdam.
10. Gorenz, R. & Schwarz, N. (2024). *ChatGPT is funnier than crowd workers but not professional humorists.* Judgment and Decision Making, 19, e41.
