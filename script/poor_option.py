#poor_option.py 根据新闻标题生成差选项（多语言版本）
#差选项类型：Irrelevant_Response, Repetition, Bland_Statement, Template_Response
import dashscope
from dashscope import Generation
import json
import time
import os

# 设置DashScope API密钥 (替换为你的实际API_KEY)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 使用的模型
MODEL_NAME = "deepseek-v3.2"

# 差选项类型列表
POOR_OPTION_TYPES = ["Irrelevant_Response", "Repetition", "Bland_Statement", "Template_Response"]

# 定义不同语言、不同差选项类型的提示词模板
PROMPTS = {
    "en": {
        "Irrelevant_Response": """Given the following news headline, generate a short response that is COMPLETELY IRRELEVANT to the headline. The response should have nothing to do with the topic of the news. It should look like a random, off-topic sentence that someone might say in a casual conversation.

News headline: "{headline}"

Only return the irrelevant response and nothing else. Do NOT include any explanation.""",

        "Repetition": """Given the following news headline, generate a response that simply REPEATS or SLIGHTLY REPHRASES the headline without adding any humor, insight, or creativity. The response should be a boring restatement of the same information.

News headline: "{headline}"

Only return the rephrased headline and nothing else. Do NOT include any explanation.""",

        "Bland_Statement": """Given the following news headline, generate a BLAND, FACTUAL statement related to the topic. The response should be relevant to the headline but completely lack any humor, wit, or entertainment value. It should read like a dry encyclopedia entry or a boring comment.

News headline: "{headline}"

Only return the bland statement and nothing else. Do NOT include any explanation.""",

        "Template_Response": """Given the following news headline, generate a GENERIC, TEMPLATE-LIKE response that could apply to almost any news headline. The response should feel mechanical, formulaic, and lacking any specific connection to the actual content of the headline. Use clichéd phrases like "That's interesting!", "Wow, what a story!", "This is so funny!", etc.

News headline: "{headline}"

Only return the template response and nothing else. Do NOT include any explanation.""",
    },

    "es": {
        "Irrelevant_Response": """Dado el siguiente titular de noticias, genera una respuesta corta que sea COMPLETAMENTE IRRELEVANTE al titular. La respuesta no debe tener nada que ver con el tema de la noticia. Debe parecer una frase aleatoria y fuera de tema.

Titular: "{headline}"

Solo devuelve la respuesta irrelevante y nada más. NO incluyas ninguna explicación. La respuesta debe estar en español.""",

        "Repetition": """Dado el siguiente titular de noticias, genera una respuesta que simplemente REPITA o REFORMULE LIGERAMENTE el titular sin añadir humor, perspicacia o creatividad. La respuesta debe ser una reformulación aburrida de la misma información.

Titular: "{headline}"

Solo devuelve el titular reformulado y nada más. NO incluyas ninguna explicación. La respuesta debe estar en español.""",

        "Bland_Statement": """Dado el siguiente titular de noticias, genera una declaración PLANA y FACTUAL relacionada con el tema. La respuesta debe ser relevante al titular pero carecer completamente de humor o valor de entretenimiento. Debe leerse como una entrada de enciclopedia aburrida.

Titular: "{headline}"

Solo devuelve la declaración plana y nada más. NO incluyas ninguna explicación. La respuesta debe estar en español.""",

        "Template_Response": """Dado el siguiente titular de noticias, genera una respuesta GENÉRICA y de PLANTILLA que podría aplicarse a casi cualquier titular. La respuesta debe sentirse mecánica y formulaica. Usa frases cliché como "¡Qué interesante!", "¡Vaya historia!", "¡Esto es muy gracioso!", etc.

Titular: "{headline}"

Solo devuelve la respuesta de plantilla y nada más. NO incluyas ninguna explicación. La respuesta debe estar en español.""",
    },

    "zh": {
        "Irrelevant_Response": """根据以下新闻标题，生成一个与标题完全无关的简短回复。回复内容不能与新闻主题有任何关联，应该像是一句随机的、离题的日常对话。

新闻标题："{headline}"

只返回无关回复，不要包含任何解释。回复必须使用中文。""",

        "Repetition": """根据以下新闻标题，生成一个简单重复或轻微改写标题的回复，不添加任何幽默、见解或创意。回复应该是对同一信息的无聊复述。

新闻标题："{headline}"

只返回改写后的标题，不要包含任何解释。回复必须使用中文。""",

        "Bland_Statement": """根据以下新闻标题，生成一个与主题相关但过于平淡的事实性陈述。回复应与标题相关，但完全缺乏幽默感或娱乐价值，读起来像一条枯燥的百科词条或无聊的评论。

新闻标题："{headline}"

只返回平淡陈述，不要包含任何解释。回复必须使用中文。""",

        "Template_Response": """根据以下新闻标题，生成一个通用的、模板化的回复，这种回复可以套用在几乎任何新闻标题上。回复应该感觉机械、公式化，缺乏与标题实际内容的具体联系。使用类似"这真有趣！"、"哇，这个故事太精彩了！"、"太搞笑了！"等陈词滥调。

新闻标题："{headline}"

只返回模板化回复，不要包含任何解释。回复必须使用中文。""",
    },
}


def generate_poor_option(headline, lang, option_type):
    """
    根据新闻标题生成指定类型的差选项
    :param headline: 新闻标题
    :param lang: 语言代码 ('en', 'es', 'zh')
    :param option_type: 差选项类型
    :return: 生成的差选项内容
    """
    prompt = PROMPTS[lang][option_type].format(headline=headline)

    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = Generation.call(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": True},
                result_format="message",
                temperature=0.85,
                top_p=0.9
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                return content.strip()
            else:
                if attempt < max_retries - 1:
                    print(f"  API调用失败 (尝试 {attempt + 1}/{max_retries}): {response.code}")
                    time.sleep(retry_delay)
                    continue
                else:
                    return {"error": True, "message": f"API调用失败: {response.code} - {response.message}"}

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  处理异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
                continue
            else:
                return {"error": True, "message": f"处理异常: {str(e)}"}


def generate_all_poor_options(headline, lang):
    """
    为一条新闻标题生成所有4种差选项
    :param headline: 新闻标题
    :param lang: 语言代码
    :return: 包含4种差选项的字典
    """
    results = {}
    for option_type in POOR_OPTION_TYPES:
        print(f"    生成 {option_type}...")
        result = generate_poor_option(headline, lang, option_type)

        if isinstance(result, dict) and result.get("error"):
            results[option_type] = f"[ERROR] {result['message']}"
            print(f"    -> 错误: {result['message']}")
        else:
            results[option_type] = result
            display = result[:50] + "..." if len(result) > 50 else result
            print(f"    -> {display}")

        # 每次API调用间隔，避免限流
        time.sleep(0.5)

    return results


def process_headlines(input_file, output_file, lang, resume=True):
    """
    处理新闻标题文件并生成差选项
    :param input_file: 输入JSON文件路径
    :param output_file: 输出JSON文件路径
    :param lang: 语言代码 ('en', 'es', 'zh')
    :param resume: 是否启用断点续传
    """
    print(f"开始处理 {lang} 语言文件: {input_file}")
    print(f"使用模型: {MODEL_NAME}")

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

        print(f"[{idx + 1}/{total}] 处理: {item_id} - {headline[:40]}...")

        # 生成所有4种差选项
        poor_options = generate_all_poor_options(headline, lang)

        # 统计错误数
        for opt_type in POOR_OPTION_TYPES:
            if poor_options[opt_type].startswith("[ERROR]"):
                error_count += 1

        # 构建输出结果
        result = {
            "id": item_id,
            "news_headline": headline,
        }
        result.update(poor_options)

        processed_data.append(result)

        # 每处理10条保存一次（防止意外中断丢失数据）
        if len(processed_data) % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            print(f"  [自动保存] 已保存 {len(processed_data)} 条数据")

    # 最终保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"\n完成! 成功处理 {len(processed_data)} 条数据，错误 {error_count} 条")
    print(f"结果保存到: {output_file}")


def poor_option_gen_all(output_dir=None):
    """
    处理所有语言的新闻标题，生成差选项数据集
    :param output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "output")

    os.makedirs(output_dir, exist_ok=True)

    data_dir = os.path.join(PROJECT_ROOT, "data")

    files = [
        (os.path.join(data_dir, "headlines_en.json"), os.path.join(output_dir, "poor_option_en.json"), "en"),
        (os.path.join(data_dir, "headlines_es.json"), os.path.join(output_dir, "poor_option_es.json"), "es"),
        (os.path.join(data_dir, "headlines_zh.json"), os.path.join(output_dir, "poor_option_zh.json"), "zh"),
    ]

    for input_file, output_file, lang in files:
        print(f"\n{'='*60}")
        print(f"处理语言: {lang}")
        print(f"{'='*60}")
        process_headlines(input_file, output_file, lang)


# ================== 使用示例 ==================
if __name__ == "__main__":
    # 处理所有语言
    poor_option_gen_all()

    # 也可以单独处理某一语言：
    # data_dir = os.path.join(PROJECT_ROOT, "data")
    # output_dir = os.path.join(PROJECT_ROOT, "output")
    # process_headlines(
    #     os.path.join(data_dir, "headlines_en.json"),
    #     os.path.join(output_dir, "poor_option_en.json"),
    #     "en"
    # )
