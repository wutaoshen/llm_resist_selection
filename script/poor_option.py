# poor_option.py 根据新闻标题生成不同质量等级的笑话候选（多语言版本）
# 质量等级：
#   低质量(low): Forced_Pun, Overexplained_Joke, Cliche_Joke, Weak_Connection
#   中等质量(medium): Safe_Humor, Predictable_Punchline, Surface_Level, Generic_Wit
#   高质量(high): 复用 good_option.py 输出，不在此脚本生成
import sys
import json
import os
import time
from dashscope import Generation

# 将项目根目录加入路径，以便导入 utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from utils import get_rate_limiter, RetryConfig, is_rate_limit_error, is_retryable_error

# 使用的模型
MODEL_NAME = "deepseek-v3.2"

# 重试配置（指数退避）
RETRY_CONFIG = RetryConfig(max_retries=3, base_delay=2.0, max_delay=30.0)

# 速率限制器（单例）
_rate_limiter = get_rate_limiter("dashscope", rpm=600)

# 质量等级定义
QUALITY_LEVELS = ["low", "medium"]  # high 由 good_option.py 生成

# 低质量笑话类型
LOW_OPTION_TYPES = ["Forced_Pun", "Overexplained_Joke", "Cliche_Joke", "Weak_Connection"]

# 中等质量笑话类型
MEDIUM_OPTION_TYPES = ["Safe_Humor", "Predictable_Punchline", "Surface_Level", "Generic_Wit"]

# ==================== 提示词模板 ====================

PROMPTS = {
    # ==================== 英文提示词 ====================
    "en": {
        # --- 低质量 ---
        "Forced_Pun": """Given the following news headline, create a joke that FORCES a pun or wordplay. The pun should feel AWKWARD, UNNATURAL, and FORCED — as if the author is desperately trying to be clever but failing. The joke should still be recognizable as an attempt at humor, not gibberish.

IMPORTANT quality constraint: This joke should NOT make anyone genuinely laugh. If a reader's reaction is anything more than an eye-roll or a groan, it's too good. Aim for a joke that people would describe as "painful" or "try-hard".

News headline: "{headline}"

Only return the joke and nothing else. Do NOT include any explanation.""",

        "Overexplained_Joke": """Given the following news headline, create a joke where the punchline is OVER-EXPLAINED. First deliver the joke, then immediately explain why it's supposed to be funny, which KILLS the humor. The joke should be recognizable as an attempt at humor but ruined by unnecessary explanation.

IMPORTANT quality constraint: This joke should NOT make anyone genuinely laugh. The over-explanation should completely drain any humor. A reader should think "that would have been okay if they hadn't explained it" — but even the base joke should be mediocre at best.

News headline: "{headline}"

Only return the joke (with its over-explanation) and nothing else. Do NOT include any explanation of the task.""",

        "Cliche_Joke": """Given the following news headline, create a joke using an EXTREMELY CLICHÉ and OVERUSED joke format. Use tired patterns like "Why did the X cross the road?", "What do you call a X?", "X walks into a bar...", or other predictable formats. The joke should feel stale and unoriginal.

IMPORTANT quality constraint: This joke should NOT make anyone genuinely laugh. It should feel like a joke recycled from a 1990s joke book. A reader should immediately recognize they've heard this exact format hundreds of times and feel zero surprise.

News headline: "{headline}"

Only return the joke and nothing else. Do NOT include any explanation.""",

        "Weak_Connection": """Given the following news headline, create a joke that is LOOSELY related to the headline but with a WEAK, UNCONVINCING punchline. The joke should attempt to connect to the news topic but the humor should feel FORCED and the logic should be a STRETCH. It should be recognizable as a joke attempt but not actually funny.

IMPORTANT quality constraint: This joke should NOT make anyone genuinely laugh. The connection between the headline and the punchline should feel like a reach — as if someone spent 5 seconds thinking of any remotely related joke. A reader should think "what does that even have to do with the headline?".

News headline: "{headline}"

Only return the joke and nothing else. Do NOT include any explanation.""",

        # --- 中等质量 ---
        "Safe_Humor": """Given the following news headline, create a joke that is MILDLY AMUSING but SAFE and UNREMARKABLE. It should be the kind of joke that makes someone smile politely but not actually laugh out loud. Avoid anything too creative, surprising, or edgy. Keep it pleasant but forgettable.

IMPORTANT quality constraint: This joke should be DECENT but NOT worth sharing with friends. It's the kind of joke you'd hear at a corporate event — inoffensive, mildly clever, but no one would remember it the next day. It should be clearly better than a terrible joke, but clearly worse than a genuinely witty one.

News headline: "{headline}"

Only return the joke and nothing else. Do NOT include any explanation.""",

        "Predictable_Punchline": """Given the following news headline, create a joke where the PUNCHLINE IS PREDICTABLE. A reader should be able to guess where the joke is going before reaching the end. The setup should be decent, but the payoff should be OBVIOUS and EXPECTED. It should still work as a joke, just not a surprising one.

IMPORTANT quality constraint: This joke should be DECENT but NOT worth sharing with friends. The setup should show some competence, but the punchline should land with a "yeah, I saw that coming" reaction. It should be clearly better than a cringe-worthy joke, but clearly worse than one that delivers a genuine surprise.

News headline: "{headline}"

Only return the joke and nothing else. Do NOT include any explanation.""",

        "Surface_Level": """Given the following news headline, create a joke that only uses the SURFACE-LEVEL, MOST OBVIOUS aspect of the headline. Don't dig deeper into the implications or find unexpected angles. Just make a straightforward, somewhat amusing observation that anyone could have made. It should be adequate but LACKING DEPTH or INSIGHT.

IMPORTANT quality constraint: This joke should be DECENT but NOT worth sharing with friends. It should read like something anyone could come up with in 10 seconds. It should be clearly better than a nonsensical or cringe joke, but clearly worse than an insightful or clever observation.

News headline: "{headline}"

Only return the joke and nothing else. Do NOT include any explanation.""",

        "Generic_Wit": """Given the following news headline, create a joke that shows SOME WIT but is NOT particularly TARGETED or SPECIFIC to this headline. The joke should be decent enough — grammatically correct, properly structured — but could almost work with many similar headlines. It should lack that special spark of originality.

IMPORTANT quality constraint: This joke should be DECENT but NOT worth sharing with friends. It should feel like a "template joke" where someone just swapped in the topic. It should be clearly better than an awkward or forced joke, but clearly worse than one with a unique, headline-specific twist.

News headline: "{headline}"

Only return the joke and nothing else. Do NOT include any explanation.""",
    },

    # ==================== 西班牙文提示词 ====================
    "es": {
        # --- 低质量 ---
        "Forced_Pun": """Dado el siguiente titular de noticias, crea un chiste que FUERCE un juego de palabras. El juego de palabras debe sentirse TORPE, ANTINATURAL y FORZADO, como si el autor estuviera intentando desesperadamente ser ingenioso pero fracasando. El chiste debe ser reconocible como un intento de humor, no sin sentido.

Restricciones de calidad IMPORTANTES: Este chiste NO debería hacer reír genuinamente a nadie. Si la reacción del lector es algo más que poner los ojos en blanco o un quejido, es demasiado bueno. Apunta a un chiste que la gente describiría como "doloroso" o "forzado".

Titular: "{headline}"

Solo devuelve el chiste y nada más. NO incluyas ninguna explicación. La respuesta debe estar en español.""",

        "Overexplained_Joke": """Dado el siguiente titular de noticias, crea un chiste donde el remate esté SOBRE-EXPLICADO. Primero cuenta el chiste, luego explica inmediatamente por qué se supone que es gracioso, lo que MATA el humor. El chiste debe ser reconocible como un intento de humor pero arruinado por la explicación innecesaria.

Restricciones de calidad IMPORTANTES: Este chiste NO debería hacer reír genuinamente a nadie. La sobre-explicación debe drenar completamente cualquier humor. El lector debería pensar "eso habría estado bien si no lo hubieran explicado" — pero incluso el chiste base debería ser mediocre en el mejor caso.

Titular: "{headline}"

Solo devuelve el chiste (con su sobre-explicación) y nada más. La respuesta debe estar en español.""",

        "Cliche_Joke": """Dado el siguiente titular de noticias, crea un chiste usando un formato EXTREMADAMENTE CLICHÉ y SOBREUSADO de la tradición humorística hispana. Usa patrones trillados como los "chistes de Jaimito/Pepito", "¿Cuál es el colmo de...?", "¿Qué le dice un X a un Y?", "¿En qué se parece X a Y?" u otros formatos clásicos del humor en español. El chiste debe sentirse gastado y poco original.

Restricciones de calidad IMPORTANTES: Este chiste NO debería hacer reír genuinamente a nadie. Debe sentirse como un chiste reciclado de un libro de chistes de los años 90. El lector debería reconocer inmediatamente que ha escuchado este formato cientos de veces y sentir cero sorpresa.

Titular: "{headline}"

Solo devuelve el chiste y nada más. NO incluyas ninguna explicación. La respuesta debe estar en español.""",

        "Weak_Connection": """Dado el siguiente titular de noticias, crea un chiste que esté VAGAMENTE relacionado con el titular pero con un remate DÉBIL y POCO CONVINCENTE. El chiste debe intentar conectarse con el tema pero el humor debe sentirse FORZADO y la lógica debe ser REBUSCADA. Debe reconocerse como un intento de chiste pero no ser realmente gracioso.

Restricciones de calidad IMPORTANTES: Este chiste NO debería hacer reír genuinamente a nadie. La conexión entre el titular y el remate debe sentirse forzada — como si alguien hubiera pensado 5 segundos en cualquier chiste remotamente relacionado. El lector debería pensar "¿qué tiene que ver eso con el titular?".

Titular: "{headline}"

Solo devuelve el chiste y nada más. NO incluyas ninguna explicación. La respuesta debe estar en español.""",

        # --- 中等质量 ---
        "Safe_Humor": """Dado el siguiente titular de noticias, crea un chiste que sea LIGERAMENTE DIVERTIDO pero SEGURO y SIN NADA ESPECIAL. Debe ser el tipo de chiste que hace sonreír educadamente pero no reír a carcajadas. Evita cualquier cosa demasiado creativa, sorprendente o atrevida. Mantenlo agradable pero olvidable.

Restricciones de calidad IMPORTANTES: Este chiste debe ser DECENTE pero NO vale la pena compartirlo con amigos. Es el tipo de chiste que escucharías en un evento corporativo — inofensivo, ligeramente ingenioso, pero nadie lo recordaría al día siguiente. Debe ser claramente mejor que un chiste terrible, pero claramente peor que uno genuinamente ingenioso.

Titular: "{headline}"

Solo devuelve el chiste y nada más. NO incluyas ninguna explicación. La respuesta debe estar en español.""",

        "Predictable_Punchline": """Dado el siguiente titular de noticias, crea un chiste donde el REMATE SEA PREDECIBLE. Un lector debería poder adivinar hacia dónde va el chiste antes de llegar al final. La premisa debe ser decente, pero el desenlace debe ser OBVIO y ESPERADO. Debe funcionar como chiste, pero no sorprender.

Restricciones de calidad IMPORTANTES: Este chiste debe ser DECENTE pero NO vale la pena compartirlo con amigos. La premisa debe mostrar cierta competencia, pero el remate debe provocar una reacción de "sí, lo veía venir". Debe ser claramente mejor que un chiste vergonzoso, pero claramente peor que uno que entrega una sorpresa genuina.

Titular: "{headline}"

Solo devuelve el chiste y nada más. NO incluyas ninguna explicación. La respuesta debe estar en español.""",

        "Surface_Level": """Dado el siguiente titular de noticias, crea un chiste que solo use el aspecto MÁS SUPERFICIAL y OBVIO del titular. No profundices en las implicaciones ni busques ángulos inesperados. Solo haz una observación directa y algo divertida que cualquiera podría haber hecho. Debe ser adecuado pero CARECER DE PROFUNDIDAD.

Restricciones de calidad IMPORTANTES: Este chiste debe ser DECENTE pero NO vale la pena compartirlo con amigos. Debe leerse como algo que cualquiera podría pensar en 10 segundos. Debe ser claramente mejor que un chiste sin sentido o vergonzoso, pero claramente peor que una observación perspicaz o ingeniosa.

Titular: "{headline}"

Solo devuelve el chiste y nada más. NO incluyas ninguna explicación. La respuesta debe estar en español.""",

        "Generic_Wit": """Dado el siguiente titular de noticias, crea un chiste que muestre ALGO DE INGENIO pero que NO sea particularmente ESPECÍFICO para este titular. El chiste debe ser decente — gramaticalmente correcto, bien estructurado — pero podría funcionar con muchos titulares similares. Debe carecer de esa chispa especial de originalidad.

Restricciones de calidad IMPORTANTES: Este chiste debe ser DECENTE pero NO vale la pena compartirlo con amigos. Debe sentirse como un "chiste plantilla" donde alguien solo cambió el tema. Debe ser claramente mejor que un chiste torpe o forzado, pero claramente peor que uno con un giro único y específico del titular.

Titular: "{headline}"

Solo devuelve el chiste y nada más. NO incluyas ninguna explicación. La respuesta debe estar en español.""",
    },

    # ==================== 中文提示词 ====================
    "zh": {
        # --- 低质量 ---
       "Forced_Pun": """根据以下新闻标题，创作一个强行使用谐音梗或双关语的笑话。双关应该显得生硬、不自然、刻意为之，就像作者拼命想要表现幽默但失败了一样。笑话应该仍然能被识别为一个幽默尝试，而不是胡言乱语。

重要质量约束：这个笑话不应该让任何人真正笑出来。如果读者的反应超过了翻白眼或叹气，那就说明太好了。目标是那种人们会形容为"尬"或"硬凹"的笑话。

新闻标题："{headline}"

只返回笑话本身，不要包含任何解释。回复必须使用中文。""",

       "Overexplained_Joke": """根据以下新闻标题，创作一个笑点被过度解释的笑话。先讲笑话，然后立刻解释为什么它应该是好笑的，从而毁掉幽默感。笑话应该能被识别为一个幽默尝试，但被不必要的解释破坏了。

重要质量约束：这个笑话不应该让任何人真正笑出来。过度解释应该彻底抽干所有幽默感。读者应该觉得"如果不解释的话还凑合"——但即使是基础笑话本身也最多算平庸。

新闻标题："{headline}"

只返回笑话（包含过度解释的部分），不要包含对任务本身的解释。回复必须使用中文。""",

        "Cliche_Joke": """根据以下新闻标题，使用极其老套和过时的笑话格式创作一个笑话。使用诸如"小明系列"、"为什么X要过马路？"、"X和Y有什么区别？"等陈旧的模式。笑话应该感觉过时且毫无新意。

重要质量约束：这个笑话不应该让任何人真正笑出来。它应该像是从90年代笑话书里翻出来的。读者应该立刻意识到自己已经听过这种格式几百遍了，完全没有惊喜感。

新闻标题："{headline}"

只返回笑话本身，不要包含任何解释。回复必须使用中文。""",

       "Weak_Connection": """根据以下新闻标题，创作一个与标题勉强相关但笑点牵强的笑话。笑话应该试图与新闻主题建立联系，但幽默感应该是勉强的，逻辑应该是牵强附会的。它应该能被识别为一个笑话尝试，但实际上并不好笑。

重要质量约束：这个笑话不应该让任何人真正笑出来。标题和笑点之间的联系应该很牵强——像是某人花了5秒钟随便想的一个勉强相关的笑话。读者应该觉得"这跟标题有什么关系？"。

新闻标题："{headline}"

只返回笑话本身，不要包含任何解释。回复必须使用中文。""",

        # --- 中等质量 ---
        "Safe_Humor": """根据以下新闻标题，创作一个温和有趣但安全平庸的笑话。它应该是那种让人礼貌微笑但不会真正大笑的笑话。避免太有创意、太出人意料或太尖锐的内容。保持愉快但容易被遗忘。

重要质量约束：这个笑话应该还不错，但不值得分享给朋友。就像在公司年会上听到的那种——无害、有点小聪明，但第二天没人会记得。它应该明显好于一个糟糕的笑话，但明显不如一个真正机智的笑话。

新闻标题："{headline}"

只返回笑话本身，不要包含任何解释。回复必须使用中文。""",

       "Predictable_Punchline": """根据以下新闻标题，创作一个笑点可预测的笑话。读者应该在看到结尾之前就能猜到笑话的走向。铺垫可以还不错，但笑点应该是显而易见、在意料之中的。它仍然应该作为笑话成立，只是不够出人意料。

重要质量约束：这个笑话应该还不错，但不值得分享给朋友。铺垫应该表现出一定水平，但笑点应该让人觉得"嗯，意料之中"。它应该明显好于一个令人尴尬的笑话，但明显不如一个能带来真正惊喜的笑话。

新闻标题："{headline}"

只返回笑话本身，不要包含任何解释。回复必须使用中文。""",

        "Surface_Level": """根据以下新闻标题，创作一个只利用标题最表面、最明显信息的笑话。不要深入挖掘深层含义或寻找意想不到的角度。只做一个任何人都能想到的直白、稍微有趣的观察。笑话应该还算合格，但缺乏深度和洞察力。

重要质量约束：这个笑话应该还不错，但不值得分享给朋友。它读起来像是任何人用10秒钟就能想到的东西。它应该明显好于一个莫名其妙或令人尴尬的笑话，但明显不如一个有洞察力或巧妙的观察。

新闻标题："{headline}"

只返回笑话本身，不要包含任何解释。回复必须使用中文。""",

       "Generic_Wit": """根据以下新闻标题，创作一个有一定机智但不特别针对这条标题的笑话。笑话应该还算不错——语法正确、结构完整——但几乎可以套用在许多类似的标题上。它应该缺乏那种特别的原创火花。

重要质量约束：这个笑话应该还不错，但不值得分享给朋友。它应该给人一种"套模板"的感觉，只是换了个话题而已。它应该明显好于一个生硬或尴尬的笑话，但明显不如一个有独特标题针对性的笑话。

新闻标题："{headline}"

只返回笑话本身，不要包含任何解释。回复必须使用中文。""",
    },
}


def generate_option(headline, lang, option_type, quality_level):
    """
    根据新闻标题生成指定类型和质量等级的笑话
    :param headline: 新闻标题
    :param lang: 语言代码 ('en', 'es', 'zh')
    :param option_type: 笑话类型
    :param quality_level: 质量等级 ('low', 'medium')
    :return: 生成的笑话内容
    """
    prompt = PROMPTS[lang][option_type].format(headline=headline)

    # 低质量使用高temperature增加随机性和不连贯感，中等质量用低temperature保持流畅但平庸
    temperature = 1.2 if quality_level == "low" else 0.6
    top_p = 0.95 if quality_level == "low" else 0.7

    for attempt in range(RETRY_CONFIG.max_retries):
        try:
            if not _rate_limiter.acquire(timeout=120):
                print(f"  [RateLimiter] 获取令牌超时, 跳过本次尝试")
                continue

            response = Generation.call(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": False},
                result_format="message",
                temperature=temperature,
                top_p=top_p
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                return content.strip()
            else:
                status_code = response.status_code
                error_msg = getattr(response, 'message', str(response))
                if is_retryable_error(error_msg, status_code) and attempt < RETRY_CONFIG.max_retries - 1:
                    delay = RETRY_CONFIG.get_delay(attempt)
                    keyword = "速率限制" if is_rate_limit_error(error_msg, status_code) else "可重试"
                    print(f"  [{keyword}错误] (状态码={status_code}), 等待{delay:.1f}s后重试 "
                          f"({attempt + 1}/{RETRY_CONFIG.max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    return {"error": True, "message": f"API调用失败: {response.code} - {error_msg}"}

        except Exception as e:
            error_str = str(e)
            if is_retryable_error(e) and attempt < RETRY_CONFIG.max_retries - 1:
                delay = RETRY_CONFIG.get_delay(attempt)
                print(f"  [可重试异常] {error_str[:100]}, 等待{delay:.1f}s后重试 "
                      f"({attempt + 1}/{RETRY_CONFIG.max_retries})")
                time.sleep(delay)
                continue
            else:
                return {"error": True, "message": f"处理异常: {error_str}"}

    return {"error": True, "message": "API调用失败，已达最大重试次数"}


def generate_all_options(headline, lang, quality_level):
    """
    为一条新闻标题生成指定质量等级的所有4种笑话
    :param headline: 新闻标题
    :param lang: 语言代码
    :param quality_level: 质量等级 ('low', 'medium')
    :return: 包含4种笑话的字典
    """
    option_types = LOW_OPTION_TYPES if quality_level == "low" else MEDIUM_OPTION_TYPES

    results = {}
    for option_type in option_types:
        print(f"    生成 [{quality_level}] {option_type}...")
        result = generate_option(headline, lang, option_type, quality_level)

        if isinstance(result, dict) and result.get("error"):
            results[option_type] = f"[ERROR] {result['message']}"
            print(f"    -> 错误: {result['message']}")
        else:
            results[option_type] = result
            display = result[:50] + "..." if len(result) > 50 else result
            print(f"    -> {display}")


    return results


def process_headlines(input_file, output_file, lang, quality_level, resume=True):
    """
    处理新闻标题文件并生成指定质量等级的笑话
    :param input_file: 输入JSON文件路径
    :param output_file: 输出JSON文件路径
    :param lang: 语言代码 ('en', 'es', 'zh')
    :param quality_level: 质量等级 ('low', 'medium')
    :param resume: 是否启用断点续传
    """
    print(f"开始处理 {lang} 语言文件 (质量等级: {quality_level}): {input_file}")
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

    option_types = LOW_OPTION_TYPES if quality_level == "low" else MEDIUM_OPTION_TYPES
    total = len(data)
    error_count = 0

    for idx, item in enumerate(data):
        headline = item.get("news_headline", "")
        item_id = item.get("id", "")

        # 跳过已处理的数据
        if item_id in processed_ids:
            continue

        print(f"[{idx + 1}/{total}] 处理: {item_id} - {headline[:40]}...")

        # 生成所有4种笑话
        options = generate_all_options(headline, lang, quality_level)

        # 统计错误数
        for opt_type in option_types:
            val = options.get(opt_type, "")
            if isinstance(val, str) and val.startswith("[ERROR]"):
                error_count += 1

        # 构建输出结果
        result = {
            "id": item_id,
            "news_headline": headline,
        }
        result.update(options)

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


def option_gen_all(quality_level="low", output_dir=None):
    """
    处理所有语言的新闻标题，生成指定质量等级的笑话数据集
    :param quality_level: 质量等级 ('low', 'medium')
    :param output_dir: 输出目录
    """
    if quality_level not in ("low", "medium"):
        raise ValueError(f"quality_level 必须是 'low' 或 'medium'，收到: {quality_level}")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "output")

    os.makedirs(output_dir, exist_ok=True)

    data_dir = os.path.join(PROJECT_ROOT, "data")

    prefix = "low_option" if quality_level == "low" else "medium_option"

    files = [
        (os.path.join(data_dir, "headlines_en.json"), os.path.join(output_dir, f"{prefix}_en.json"), "en"),
        (os.path.join(data_dir, "headlines_es.json"), os.path.join(output_dir, f"{prefix}_es.json"), "es"),
        (os.path.join(data_dir, "headlines_zh.json"), os.path.join(output_dir, f"{prefix}_zh.json"), "zh"),
    ]

    for input_file, output_file, lang in files:
        print(f"\n{'='*60}")
        print(f"处理语言: {lang} | 质量等级: {quality_level}")
        print(f"{'='*60}")
        process_headlines(input_file, output_file, lang, quality_level)


# ================== 使用示例 ==================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成不同质量等级的笑话候选选项")
    parser.add_argument('--quality', type=str, default='low', choices=['low', 'medium'],
                        help='质量等级: low(低质量), medium(中等质量)')
    parser.add_argument('--lang', type=str, default=None, choices=['en', 'es', 'zh'],
                        help='指定单一语言处理，不指定则处理全部语言')
    args = parser.parse_args()

    if args.lang:
        # 处理单一语言
        data_dir = os.path.join(PROJECT_ROOT, "data")
        output_dir = os.path.join(PROJECT_ROOT, "output")
        os.makedirs(output_dir, exist_ok=True)

        prefix = "low_option" if args.quality == "low" else "medium_option"
        process_headlines(
            os.path.join(data_dir, f"headlines_{args.lang}.json"),
            os.path.join(output_dir, f"{prefix}_{args.lang}.json"),
            args.lang,
            args.quality
        )
    else:
        # 处理所有语言
        option_gen_all(quality_level=args.quality)
