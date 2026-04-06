# demo_kimi_k25.py - 测试 Kimi-K2.5 模型调用的简单示例
import dashscope
import os

# 设置 DashScope API 密钥
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

# 测试用的新闻标题
test_headlines = [
    "Ryanair to cut 1 million more passenger seats in Spain",
    "Do body wipes actually work? Experts weigh in",
    "These colleges are welcoming pets in dorms to reduce students' stress and anxiety"
]

def test_kimi_k25(headline, modelname="kimi-k2.5"):
    """
    使用 Kimi-K2.5 根据新闻标题生成笑话
    """
    prompt = f"""Create a joke based on this title of a news article:

"{headline}"

The joke should be concise, creative and genuinely funny. Only return the joke and nothing else."""

    print(f"\n{'='*60}")
    print(f"新闻标题: {headline}")
    print(f"{'='*60}")
    
    try:
        # Kimi-K2.5 是多模态模型，需要使用 dashscope.MultiModalConversation.call()
        messages = [{
            "role": "user",
            "content": [{"text": prompt}]
        }]
        
        response = dashscope.MultiModalConversation.call(
            api_key=dashscope.api_key,
            model=modelname,
            messages=messages
        )
        
        if response.status_code == 200:
            # 按照官方模板格式获取结果
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
    主函数：测试 Kimi-K2.5 模型
    """
    print("="*60)
    print("Kimi-K2.5 模型测试 Demo")
    print("="*60)
    
    # 检查 API 密钥
    if not dashscope.api_key:
        print("\n⚠️  警告: 请设置 DASHSCOPE_API_KEY 环境变量")
        print("示例: set DASHSCOPE_API_KEY=your-actual-api-key")
        return
    
    # 测试每个标题
    results = []
    for i, headline in enumerate(test_headlines, 1):
        print(f"\n[{i}/{len(test_headlines)}] 正在测试...")
        joke = test_kimi_k25(headline, modelname="kimi-k2.5")
        results.append({
            "headline": headline,
            "joke": joke
        })
    
    # 总结
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    success_count = sum(1 for r in results if r["joke"] is not None)
    print(f"成功: {success_count}/{len(test_headlines)}")

if __name__ == "__main__":
    main()
