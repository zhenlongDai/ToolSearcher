import re
import json

def clean_backslashes(text: str) -> str:
    """删除连续超过3个的反斜杠"""
    return re.sub(r'\\{4,}', '', text)


def clean_json_strings(obj):
    """
    递归处理JSON对象，清理所有字符串值中的连续反斜杠
    
    Args:
        obj: 任意JSON对象（dict, list, str, int, float, bool, None）
        
    Returns:
        处理后的对象
    """
    if isinstance(obj, dict):
        # 字典：递归处理所有key和value
        return {k: clean_json_strings(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        # 列表：递归处理所有元素
        return [clean_json_strings(item) for item in obj]
    
    elif isinstance(obj, str):
        # 字符串：清理反斜杠
        return clean_backslashes(obj)
    
    else:
        # 其他类型（int, float, bool, None）：直接返回
        return obj


# 测试用例
if __name__ == "__main__":
    # 复杂嵌套的JSON对象
    test_data = {
        "name": "test\\\\\\\\path",
        "value": 123,
        "config": {
            "path": "C:\\\\\\\\\\Windows\\\\System",
            "enabled": True,
            "nested": {
                "deep": "very\\\\\\\\deep\\\\path",
                "count": 42
            }
        },
        "items": [
            "item\\\\\\\\one",
            {
                "id": 1,
                "desc": "test\\\\\\\\\\\\description"
            },
            ["list\\\\\\\\item", "normal\\\\text", 999]
        ],
        "empty": None,
        "boolean": False
    }
    
    print("=== 原始数据 ===")
    print(json.dumps(test_data, indent=2, ensure_ascii=False))
    
    # 清理数据
    cleaned_data = clean_json_strings(test_data)
    
    print("\n=== 清理后数据 ===")
    print(json.dumps(cleaned_data, indent=2, ensure_ascii=False))
    
    # 验证
    print("\n=== 对比 ===")
    print(f"原始: {test_data['name']!r}")
    print(f"清理: {cleaned_data['name']!r}")
    print(f"\n原始: {test_data['config']['path']!r}")
    print(f"清理: {cleaned_data['config']['path']!r}")
    print(f"\n原始: {test_data['items'][0]!r}")
    print(f"清理: {cleaned_data['items'][0]!r}")