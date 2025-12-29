import src.utils.retriever
from src.utils.retriever.embeddingFactory import EmbeddingFactory

def test1():
    query = "return maximum value"
    docs = ["def f(a,b): if a>b: return a else return b", "def f(a,b): if a<b: return a else return b"]
    retriever = EmbeddingFactory.create("unixcoder", model_name= "unixcoder", model_path="/data/dzl/package/model/unixcoder-base")
    query_embeding = retriever.get_query_embedding(query)
    doc_embeddings = retriever.get_sentence_embeddings(docs)
    for i, doc_embedding in enumerate(doc_embeddings):
        sim = retriever.compute_similaritie(query_embeding, doc_embedding)
        print(f"Document {i} similarity: {sim}")

def test2():
    query = "Access the Spotify API to retrieve the user's song, album, and playlist libraries"
if __name__ == "__main__":
       

    EmbeddingFactory.auto_import_all(src.utils.retriever)
    query = "return maximum value"
    docs = ["def f(a,b): if a>b: return a else return b", "def f(a,b): if a<b: return a else return b"]
    retriever = EmbeddingFactory.create("unixcoder", model_name= "unixcoder", model_path="/data/dzl/package/model/unixcoder-base")
    query_embeding = retriever.get_query_embedding(query)
    doc_embeddings = retriever.get_sentence_embeddings(docs)
    for i, doc_embedding in enumerate(doc_embeddings):
        sim = retriever.compute_similaritie(query_embeding, doc_embedding)
        print(f"Document {i} similarity: {sim}")
