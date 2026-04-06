#step3_new.py 根据新闻标题生成幽默笑话（多语言版本）
import dashscope
from dashscope import Generation
import json
import time
import os

# 设置DashScope API密钥 (替换为你的实际API_KEY)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# Kimi-K2.5 需要使用的 API 端点
KIMI_BASE_URL = 'https://dashscope.aliyuncs.com/api/v1'

# 需要使用 MultiModalConversation 调用的模型列表
MULTIMODAL_MODELS = ["kimi-k2.5"]

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 定义不同语言的提示词模板
PROMPTS = {
    "en": """Create a joke based on this title of a news article:

"{headline}"

The joke should be concise, creative and genuinely funny. Only return the joke and nothing else.""",

    "es": """Create a joke based on this title of a news article:

"{headline}"

The joke should be concise, creative and genuinely funny. Only return the joke and nothing else. All jokes must be in spanish.""",

    "zh": """Create a joke based on this title of a news article:

"{headline}"

The joke should be concise, creative and genuinely funny. Only return the joke and nothing else. All jokes must be in chinese."""
}

def generate_joke(headline, lang, modelname="qwen-max"):
    """
    根据新闻标题生成笑话
    :param headline: 新闻标题
    :param lang: 语言代码 ('en', 'es', 'zh')
    :param modelname: 使用的模型名称
    :return: 生成的笑话内容
    """
    
    # 获取对应语言的提示词
    prompt = PROMPTS[lang].format(headline=headline)
    
    max_retries = 3
    retry_delay = 1  # 重试间隔(秒)
    
    # 判断是否为多模态模型（如 kimi-k2.5）
    is_multimodal = modelname in MULTIMODAL_MODELS
    
    for attempt in range(max_retries):
        try:
            if is_multimodal:
                # Kimi-K2.5 等多模态模型使用 MultiModalConversation.call()
                # 设置对应的 API 端点
                original_base_url = getattr(dashscope, 'base_http_api_url', None)
                dashscope.base_http_api_url = KIMI_BASE_URL
                
                messages = [{
                    "role": "user",
                    "content": [{"text": prompt}]
                }]
                
                response = dashscope.MultiModalConversation.call(
                    api_key=dashscope.api_key,
                    model=modelname,
                    messages=messages,
                    extra_body={"enable_thinking": True},
                    temperature=0.85,
                    top_p=0.9
                )
                
                # 恢复原始 base_url
                if original_base_url:
                    dashscope.base_http_api_url = original_base_url
                
                # 解析API响应
                if response.status_code == 200:
                    # 多模态模型返回格式: content[0]["text"]
                    content = response.output.choices[0].message.content[0]["text"]
                    return content.strip()
                else:
                    if attempt < max_retries - 1:
                        print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {response.code}")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return {"error": True, "message": f"API调用失败: {response.code} - {response.message}"}
            else:
                # 其他模型使用 Generation.call()
                response = Generation.call(
                    model=modelname,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"enable_thinking": True},
                    result_format="message",
                    temperature=0.85,
                    top_p=0.9
                )
                
                # 解析API响应
                if response.status_code == 200:
                    content = response.output.choices[0].message.content
                    # 直接返回生成的笑话内容（已经是纯文本）
                    return content.strip()
                else:
                    # API调用失败
                    if attempt < max_retries - 1:
                        print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {response.code}")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # 所有重试都失败
                        return {"error": True, "message": f"API调用失败: {response.code} - {response.message}"}
                
        except Exception as e:
            # 捕获其他异常
            if attempt < max_retries - 1:
                print(f"处理异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
                continue
            else:
                # 所有重试都失败
                return {"error": True, "message": f"处理异常: {str(e)}"}

def process_headlines(input_file, output_file, lang, modelname="qwen-max", resume=True):
    """
    处理新闻标题文件并生成笑话
    :param input_file: 输入JSON文件路径
    :param output_file: 输出JSON文件路径
    :param lang: 语言代码 ('en', 'es', 'zh')
    :param modelname: 使用的模型名称
    :param resume: 是否启用断点续传
    """
    print(f"开始处理 {lang} 语言文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 断点续传：加载已处理的数据
    processed_data = []
    processed_ids = set()
    if resume and os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
                processed_ids = {item["id"] for item in processed_data}
                print(f"断点续传：已加载 {len(processed_ids)} 条已处理数据")
        except (json.JSONDecodeError, KeyError):
            processed_data = []
            processed_ids = set()
    
    total = len(data)
    error_count = 0
    
    for idx, item in enumerate(data):
        headline = item.get("news_headline", "")
        item_id = item.get("id", "")
        
        # 跳过已处理的数据
        if item_id in processed_ids:
            continue
        
        print(f"[{idx + 1}/{total}] 处理: {item_id}")
        
        # 生成笑话
        joke = generate_joke(headline, lang, modelname)
        
        # 处理错误返回
        if isinstance(joke, dict) and joke.get("error"):
            error_count += 1
            joke_text = f"[ERROR] {joke['message']}"
            print(f"  -> 错误: {joke['message']}")
        else:
            joke_text = joke
            print(f"  -> 笑话: {joke_text[:50]}..." if len(joke_text) > 50 else f"  -> 笑话: {joke_text}")
        
        # 构建输出结果
        result = {
            "id": item_id,
            "news_headline": headline,
            "joke": joke_text
        }
        
        processed_data.append(result)
        
        # 每处理10条保存一次（防止意外中断丢失数据）
        if len(processed_data) % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        # API调用间隔，避免限流
        time.sleep(0.5)
    
    # 最终保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"成功处理 {len(processed_data)} 条数据，错误 {error_count} 条，保存到 {output_file}")

def joke_gen_all(modelname="qwen-max", output_dir=None):
    """
    处理所有语言的新闻标题
    :param modelname: 使用的模型名称
    :param output_dir: 输出目录
    """
    # 使用基于项目根目录的绝对路径
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "output")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    data_dir = os.path.join(PROJECT_ROOT, "data")
    
    # 定义输入输出文件
    files = [
        (os.path.join(data_dir, "headlines_en.json"), os.path.join(output_dir, f"step3_new_{modelname}_en.json"), "en"),
        (os.path.join(data_dir, "headlines_es.json"), os.path.join(output_dir, f"step3_new_{modelname}_es.json"), "es"),
        (os.path.join(data_dir, "headlines_zh.json"), os.path.join(output_dir, f"step3_new_{modelname}_zh.json"), "zh"),
    ]
    
    for input_file, output_file, lang in files:
        process_headlines(input_file, output_file, lang, modelname)

# ================== 使用示例 ==================
if __name__ == "__main__":
    # 可以单独处理某一语言
    # process_headlines("../data/headlines_en.json", "../output/step3_new_qwen-max_en.json", "en", "qwen-max")
    
    # Kimi-K2.5 多模态模型（使用 MultiModalConversation 调用）
    #joke_gen_all(modelname="kimi-k2.5")

    # 或者处理所有语言
    #joke_gen_all(modelname="qwen3-max")
    
    # 也可以使用其他模型
    #joke_gen_all(modelname="deepseek-v3.2")
    joke_gen_all(modelname="glm-5")
    
    # Kimi-K2.5 多模态模型（使用 MultiModalConversation 调用）
    #joke_gen_all(modelname="kimi-k2.5")
    
