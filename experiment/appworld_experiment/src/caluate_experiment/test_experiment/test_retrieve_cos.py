import src.utils.retriever
from src.utils.retriever.embeddingFactory import EmbeddingFactory
from appworld.task import Task
from appworld import update_root
import json
import torch
from tqdm import tqdm
def get_all_apis():
    task = Task.load("82e2fac_1", load_ground_truth=False)
    api_docs_json_list = []
    for app_name, api_docs in task.api_docs.items():
        # if app_name not in ["spotify"]:
        #     continue
        for api_name, api_doc in api_docs.items():
            api_docs_json_list.append(api_doc)
            # if api_name == "show_account":
            #     print(api_doc)
    # len(api_docs_json_list)
    Lennum = len(api_docs_json_list)
    print(f"Total API docs: {Lennum}")
    return [json.dumps(doc) for doc in api_docs_json_list]
if __name__ == "__main__":

    update_root("./appworld")

    EmbeddingFactory.auto_import_all(src.utils.retriever)
    query = "Access the Spotify API to retrieve the user's song, album, and playlist libraries."
    
    retriever = EmbeddingFactory.create("Qwen3_Embedding", model_name= "Qwen3_Embedding", model_path="/data/LLMs/Qwen3-Embedding-0.6B")
    query_embeding = retriever.get_query_embedding(query)
    docs = get_all_apis()
    docA ='''{
           app_name": "spotify", "api_name": "show_account", "path": "/spotify/account", "method": "GET", "description": "Show your account information. Unlike show_profile, this includes private information"
           '''
    doc_embedding = retriever.get_sentence_embeddings(docs)
    # 存下来，再排序
    results = []
    for i, doc_embedding in tqdm(enumerate(doc_embedding)):
        sim = retriever.model.similarity(query_embeding, doc_embedding)
        print(f"Document {i} similarity: {sim}")
        results.append((i, sim))
    # 按相似度排序
    results = sorted(results, key=lambda x: x[1],  reverse=True)
    print("Top 20 similar documents:")
    for i in range(20):
        doc_index, sim = results[i]
        print(f"Rank {i+1}: Document {doc_index} with similarity {sim}")
        print(docs[doc_index])
