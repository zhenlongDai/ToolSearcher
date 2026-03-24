import re

def parse_tool_list(s: str):
    """
    从形如
    <tool_list>
    a.b.c,d.e.f
    </tool_list>
    的字符串中解析出 API 名列表。
    """
    start_tag = "<tool_list>"
    end_tag = "</tool_list>"

    # 找到标签位置
    start = s.find(start_tag)
    end = s.find(end_tag)

    if start == -1 or end == -1 or end < start:
        return []  # 或者 raise ValueError

    # 取出中间内容
    inner = s[start + len(start_tag):end]

    # 去掉首尾空白和换行
    inner = inner.strip()

    if not inner:
        return []

    # 按逗号分隔并 strip 每一项
    apis = [item.strip() for item in inner.split(",") if item.strip()]
    return apis




def parse_tool_apiname_lists_from_retrieval_content(retrieval_content):
    """
    从 retrieval_content 中提取每段的 category_name, tool_name, api_name。
    
    参数
    ----
    retrieval_content : str 或 dict
        - 如果是 dict，假定有键 "result"，内容为多行文本；
        - 如果是 str，直接视为多行文本。
    
    返回
    ----
    List[Dict[str, str]]，每个元素形如：
        {
            "category_name": "...",
            "tool_name": "...",
            "api_name": "..."
        }
    """
    # 1. 取出原始文本
    if isinstance(retrieval_content, dict):
        text = retrieval_content.get("result", "")
    else:
        text = retrieval_content
    
    text = text.replace("\\n", "\n")

    # 2. 按 doc 分段：doc 1:, doc 2:, ...（仅用于切块，不保留 doc_id）
    doc_pattern = re.compile(r"doc\s+\d+:\s*(.*?)(?=doc\s+\d+:|$)", re.DOTALL | re.IGNORECASE)
    
    results: List[Dict[str, str]] = []
    
    for match in doc_pattern.finditer(text):
        doc_block = match.group(1)
        # 3. 抽取 category_name/tool_name/api_name
        cat_match = re.search(r"category_name:\s*([^\n\r]+)", doc_block, re.IGNORECASE)
        tool_match = re.search(r"tool_name:\s*([^\n\r]+)", doc_block, re.IGNORECASE)
        api_match = re.search(r"api_name:\s*([^\n\r]+)", doc_block, re.IGNORECASE)
        
        category_name = cat_match.group(1).strip() if cat_match else ""
        tool_name = tool_match.group(1).strip() if tool_match else ""
        api_name = api_match.group(1).strip() if api_match else ""
       
        if category_name and tool_name and api_name:
            results.append(
                f"{category_name}.{tool_name}.{api_name}"
            )
    return results

if __name__ == '__main__':
    s = "<tool_list>\nlogistics.suivi_colis.latest,logistics.suivi_colis.all,logistics.create_container_tracking.get_tracking_data,logistics.aftership.getlastcheckpointtrackinginfobytrackingnumber,events.ticketmaster.getsingleevent\n</tool_list>"
    print(parse_tool_list(s))
    # 输出：
    # ['logistics.suivi_colis.latest',
    #  'logistics.suivi_colis.all',
    #  'logistics.create_container_tracking.get_tracking_data',
    #  'logistics.aftership.getlastcheckpointtrackinginfobytrackingnumber',
    #  'events.ticketmaster.getsingleevent']