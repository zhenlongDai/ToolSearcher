from src.utils.retriever.embeddingFactory import EmbeddingFactory
from src.utils.retriever.embeddingBase import EmbeddingBase
from transformers import AutoTokenizer, AutoModel,AutoConfig
import torch    
from sentence_transformers import SentenceTransformer
#from sentence_transformers.util.tensor import normalize_embeddings,_convert_to_batch_tensor

@EmbeddingFactory.register("Qwen3_Embedding")
class Qwen3Embedding(EmbeddingBase):
    def __init__(self, model_name, model_path, **kwargs):
        super().__init__(model_name, model_path, **kwargs)
        # Load the model
        self.model = SentenceTransformer(model_path).cuda()    
        config = AutoConfig.from_pretrained(model_path, padding_side='left')
        self.dim = config.hidden_size
        print("model_path:", model_path)
        print("Qwen3Embedding model loaded. dim:", self.dim)

    def get_embedding_dimension(self) -> int:
        return self.dim
    
    def encode(self, texts):
        with torch.no_grad():
            sentence_embeddings = self.model.encode(texts)
            sentence_embeddings = torch.tensor(sentence_embeddings)
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings

    def get_query_embedding(self, query: str) -> torch.Tensor:
        query_embedding = self.model.encode(query, prompt_name="query")
        return query_embedding
    
@EmbeddingFactory.register("unixcoder")
class UnixCoderEmbedding(EmbeddingBase):

    def __init__(self, model_name, model_path, **kwargs):
        super().__init__(model_name, model_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModel.from_pretrained(self.model_path)
        self.dim = model.config.hidden_size
        self.max_seq_length = 512 
        #self.model = torch.nn.DataParallel(model).cuda()
        self.model = torch.nn.DataParallel(model, device_ids=[0]).cuda(0)
        self.model.eval()


    def get_embedding_dimension(self) -> int:
        return self.dim
    
    def get_query_embedding(self, query: str) -> torch.Tensor:
        return self.encode([query])[0]
    
    def encode(self, batch_texts):
        source_ids = []
        for i in range(len(batch_texts)):
            source_ids.append(self.unixcoder_tokenize(batch_texts[i], self.tokenizer, self.max_seq_length))
        source_ids = torch.tensor(source_ids, dtype=torch.long).cuda()
        mask = source_ids.ne(self.tokenizer.pad_token_id)
        token_embeddings = self.model(source_ids, attention_mask=mask)[0]
        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    
    def unixcoder_tokenize(self, text, tokenizer, max_length):
        """
        Converts text to a list of token ids.
        :param text: The text to be converted
        :param tokenizer: The tokenizer to use
        :param max_length: The maximum input length
        :return: A list of token ids
        """
        tokens = tokenizer.tokenize(text)
        tokens = tokens[:max_length - 4]
        tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + tokens + [tokenizer.sep_token]
        tokens_id = tokenizer.convert_tokens_to_ids(tokens)
        padding_length = max_length - len(tokens_id)
        tokens_id += [tokenizer.pad_token_id] * padding_length
        return tokens_id 
    



# RetrieverFactory.register("APIPredictor")
# class LLMRetriever(RetrieverBase):
#     def get_sentence_embeddings(self, texts):
#          print("Generating UnixCoderRetriever embeddings...")