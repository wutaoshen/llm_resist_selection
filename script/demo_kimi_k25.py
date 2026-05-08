# demo_qwen36.py - 测试 Qwen3.6 系列模型调用的简单示例
import dashscope
import os

# 设置 DashScope API 密钥
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

# 本次测试的目标模型列表
TEST_MODELS = ["qwen3.6-27b"]

# 测试用的新闻标题
test_headlines = [
    "Ryanair to cut 1 million more passenger seats in Spain",
    "Do body wipes actually work? Experts weigh in",
    "These colleges are welcoming pets in dorms to reduce students' stress and anxiety"
]

def test_qwen36(headline, modelname):
    """
    使用指定的 Qwen3.6 模型根据新闻标题生成笑话
    """
    prompt = f"""Create a joke based on this title of a news article:

"{headline}"

The joke should be concise, creative and genuinely funny. Only return the joke and nothing else."""

    print(f"\n{'-'*60}")
    print(f"[模型] {modelname}")
    print(f"[标题] {headline}")
    print(f"{'-'*60}")

    try:
        # 参考官方多模态调用框架，使用 MultiModalConversation.call()
        messages = [{
            "role": "user",
            "content": [{"text": prompt}]
        }]

        response = dashscope.MultiModalConversation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model=modelname,
            messages=messages
        )

        if response.status_code == 200:
            joke = response.output.choices[0].message.content[0]["text"]
            print(f"生成的笑话: {joke}")
            return joke
        else:
            print(f"API 调用失败: {response.code} - {response.message}")
            return None

    except Exception as e:
        print(f"发生异常: {str(e)}")
        return None

def main():
    """
    主函数：遍历 Qwen3.6 系列模型进行测试
    """
    print("="*60)
    print("Qwen3.6 系列模型测试 Demo")
    print(f"测试模型: {', '.join(TEST_MODELS)}")
    print("="*60)

    # 检查 API 密钥
    if not dashscope.api_key:
        print("\n⚠️  警告: 请设置 DASHSCOPE_API_KEY 环境变量")
        print("示例: set DASHSCOPE_API_KEY=your-actual-api-key")
        return

    # 按模型维度聚合结果
    all_results = {}
    for model_name in TEST_MODELS:
        print(f"\n{'='*60}")
        print(f"开始测试模型: {model_name}")
        print(f"{'='*60}")

        results = []
        for i, headline in enumerate(test_headlines, 1):
            print(f"\n[{model_name}] [{i}/{len(test_headlines)}] 正在测试...")
            joke = test_qwen36(headline, modelname=model_name)
            results.append({
                "headline": headline,
                "joke": joke
            })
        all_results[model_name] = results

    # 总结
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    for model_name, results in all_results.items():
        success_count = sum(1 for r in results if r["joke"] is not None)
        print(f"[{model_name}] 成功: {success_count}/{len(test_headlines)}")

if __name__ == "__main__":
    main()
