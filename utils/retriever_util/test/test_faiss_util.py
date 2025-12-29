import os
import shutil
import numpy as np
from src.utils.retriever.faiss_util import FaissUtil

def test_faiss_util():
    dim = 16
    index_dir = "./src/utils/retriever/faiss_test"
    index_name = "test.index"
    index_path = os.path.join(index_dir, index_name)
    text_path = index_path + ".pkl"

    # 清理旧测试目录
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)

    # 1. 初始化并添加向量和文本
    faiss_util = FaissUtil(dim, index_name=index_name, index_dir=index_dir)
    vectors = np.random.rand(5, dim).astype('float32')
    texts = [f"文本{i}" for i in range(5)]
    faiss_util.add(vectors, texts)
    assert len(faiss_util) == 5, "添加向量后数量应为5"
    assert faiss_util.texts == texts, "文本内容应与添加时一致"

    # 2. 保存索引和文本
    faiss_util.save()
    assert os.path.exists(index_path), "索引文件未保存成功"
    assert os.path.exists(text_path), "文本文件未保存成功"

    # 3. 二次加载，自动load
    faiss_util2 = FaissUtil(dim, index_name=index_name, index_dir=index_dir)
    assert len(faiss_util2) == 5, "二次加载后数量应为5"
    assert faiss_util2.texts == texts, "二次加载后文本应一致"

    # 4. 检索
    query = vectors[0]
    results = faiss_util2.search(query, topk=3)

    print("Top3 results:", results)

    # 5. 添加更多向量和文本并保存
    more_vectors = np.random.rand(2, dim).astype('float32')
    more_texts = ["新文本1", "新文本2"]
    faiss_util2.add(more_vectors, more_texts)
    assert len(faiss_util2) == 7, "添加更多向量后数量应为7"
    faiss_util2.save()

    # 6. 再次加载，检查数量和文本
    faiss_util3 = FaissUtil(dim, index_name=index_name, index_dir=index_dir)
    assert len(faiss_util3) == 7, "第三次加载后数量应为7"
    assert faiss_util3.texts[-2:] == more_texts, "新添加文本应被保存和加载"

    print("所有测试通过！")

if __name__ == "__main__":
    test_faiss_util()