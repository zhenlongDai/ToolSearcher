import requests
import json

# 你的服务地址
URL = "http://33.130.107.27:1350/retrieve"

# 构造输入数据
payload = {
    "category": None,
    "query": "Searches Reddit posts.",
    "topk": 5,
    "return_scores": True
}

# 发送 POST 请求
try:
    response = requests.post(URL, json=payload, timeout=60)
    print(payload['query'])
    response.raise_for_status()  # 检查 HTTP 200，否则抛异常
    data = response.json()

    print("=== 检索结果 ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))

except requests.exceptions.RequestException as e:
    print(f"❌ 请求失败: {e}")