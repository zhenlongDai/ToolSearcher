# quick_test.py
import requests
import time
from concurrent.futures import ThreadPoolExecutor
import statistics

URL = "http://33.130.107.27:1350/retrieve"
NUM_WORKERS = 120
TOTAL_REQUESTS = 500

def send_request(request_id):
    payload = {
        "category": None,
        "query": """
        "category_name": "Business",
        "tool_name": "YC Hacker news official",
        "api_name": "new stories",
        "api_description": "The official hacker news API",
        """,
        "topk": 5,
        "return_scores": True
    }
    
    start = time.time()
    try:
        response = requests.post(URL, json=payload, timeout=300)
        latency = (time.time() - start) * 1000
        return {
            'success': response.status_code == 200,
            'latency': latency,
            'status': response.status_code
        }
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {
            'success': False,
            'latency': latency,
            'error': str(e)
        }

print(f"开始测试: {NUM_WORKERS} 并发, {TOTAL_REQUESTS} 请求\n")

start_time = time.time()
results = []

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(send_request, i) for i in range(TOTAL_REQUESTS)]
    for future in futures:
        results.append(future.result())

total_time = time.time() - start_time

# 统计
successful = [r for r in results if r['success']]
failed = [r for r in results if not r['success']]
latencies = [r['latency'] for r in successful]

print(f"\n{'='*60}")
print("测试结果:")
print(f"{'='*60}")
print(f"总请求数: {len(results)}")
print(f"成功: {len(successful)} ({len(successful)/len(results)*100:.2f}%)")
print(f"失败: {len(failed)} ({len(failed)/len(results)*100:.2f}%)")
print(f"总耗时: {total_time:.2f}s")
print(f"QPS: {len(results)/total_time:.2f}")

if latencies:
    print(f"\n延迟统计 (ms):")
    print(f"  最小: {min(latencies):.2f}")
    print(f"  最大: {max(latencies):.2f}")
    print(f"  平均: {statistics.mean(latencies):.2f}")
    print(f"  中位: {statistics.median(latencies):.2f}")
    sorted_lat = sorted(latencies)
    print(f"  P95: {sorted_lat[int(len(sorted_lat)*0.95)]:.2f}")
    print(f"  P99: {sorted_lat[int(len(sorted_lat)*0.99)]:.2f}")

print(f"{'='*60}\n")