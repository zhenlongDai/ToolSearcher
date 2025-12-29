from typing import Any
import json
from appworld.task import Task
from src.utils.common.api_predictor import VALID_MODES_LITERAL
from src.utils.common.api_predictor import APIPredictor as _APIPredictor
from src.utils.common.usage_tracker import Usage
from src.utils.retriever.retriever import FaissEmbeddingRetriever

class APIEmbeddingPredictor(_APIPredictor):  # type: ignore[misc]
    def __init__(
        self,
        model_config: dict[str, Any],
        prompt_file_path: str,
        demo_task_ids: list[str],
        max_predicted_apis: int = 20,
        app_api_separator: str = ".",
        is_simple: bool = False,
        mode: VALID_MODES_LITERAL = "predicted", 
    ):
        super().__init__(
            prompt_file_path=prompt_file_path,
            demo_task_ids=demo_task_ids,
            max_predicted_apis=max_predicted_apis,
            app_api_separator=app_api_separator,
            mode=mode,
        )
        assert model_config['topk'] == max_predicted_apis, "faiss topk must equal to max_predicted_apis"
        
        self.is_simple = is_simple
        #移除model_config中的topk|is_simple，FaissUtil不需要该参数
        if 'topk' in model_config:
            model_config.pop('topk')
        if 'is_simple' in model_config:
            model_config.pop('is_simple')

        self.retriever = FaissEmbeddingRetriever(**model_config)
        if not self.retriever.faiss_util.haved_state:
            apidoc_task_id = "82e2fac_1"
            apidoc_task = Task.load(apidoc_task_id, load_ground_truth=False)
            self.api_docs = self.build_retriever_messages(apidoc_task)
            self.retriever.add_texts(self.api_docs)
            self.retriever.save_embedding_dataset()
        self.test_task = Task.load("82e2fac_1", load_ground_truth=False)
        
        print("self.max_predicted_apis:", self.max_predicted_apis)

    def build_retriever_messages(
        self, test_task: Task
    ) -> list[dict[str, Any]]:
        api_docs_list = []
        for app_name, api_docs in test_task.api_docs.items():
            if app_name not in self.exclude_app_names: 
                for api_name, api_doc in api_docs.items():
                    if self.is_simple:
                        str_doc = json.dumps({
                            "app_name": app_name,
                            "api_name": api_name,
                            "description": api_doc["description"]
                        })
                    else:
                        str_doc = json.dumps(api_doc)

                    api_docs_list.append(str_doc) 
        print(f"Build {len(api_docs_list)} api docs for retriever.")
        print(f"Example api doc:", api_docs_list[0])
        return api_docs_list
    
    def predict(
        self, task: Task, topk: int = None
    ) -> tuple[list[str], dict[str, Any]]:
        if self.mode != "predicted":
            predicted_apis = self.non_predicted_apis(task)
            content = "\n".join(predicted_apis)
            return predicted_apis, {"content": content, "standardized_usage": Usage()}
        
        if topk is not None:
            #print("Using custom topk for prediction:", topk)
            results = self.retriever.search(task.instruction, topk)
        else:
            results = self.retriever.search(task.instruction, self.max_predicted_apis)

        predicted_output = ""
        for item in results:
            api_doc = json.loads(item['texts'])
            api_info = api_doc['app_name'] + self.app_api_separator + api_doc['api_name']
            predicted_output += api_info + "\n"
        predicted_output = predicted_output.strip()

        
        predicted_apis = self.predicted_output_to_apis(task, predicted_output, topk)
        content = "\n".join(predicted_apis)
        return predicted_apis, {"content": content, "standardized_usage": Usage()}
    
    def predict_by_instruction(
        self, instruction: str, igonre_complete_api = False
    ) -> tuple[list[str], dict[str, Any]]:
    
        results = self.retriever.search(instruction, self.max_predicted_apis)
        predicted_output = ""
        #print("instruction:", instruction)
        for item in results:
            api_doc = json.loads(item['texts'])
            #print(item)
            api_info = api_doc['app_name'] + self.app_api_separator + api_doc['api_name']
            predicted_output += api_info + "\n"
        
        #input("Press Enter to continue...")
        predicted_output = predicted_output.strip()
        predicted_apis = self.predicted_output_to_apis(self.test_task, predicted_output, self.max_predicted_apis, igonre_complete_api)
        content = "\n".join(predicted_apis)
        return predicted_apis, {"content": content, "standardized_usage": Usage()}
    
    #output = {'content': 
    # 'supervisor.complete_task\nsupervisor.show_account_passwords\nsupervisor.show_profile\nphone.login\nphone.show_contact_relationships\nphone.search_contacts\nvenmo.login\nvenmo.add_friend\nvenmo.remove_friend', 
    # 'refusal': None, 'role': 'assistant', 
    # 'annotations': [], 'audio': None, 'function_call': None, 
    # 'tool_calls': None, 
    # 'standardized_usage': Usage(tokens=Tokens(input_cache_miss=7380, input_cache_hit=0, input_cache_write=0, output=51), cost_per_token=CostPerToken(input_cache_miss=7.5e-07, input_cache_hit=7.5e-07, input_cache_write=0.0, output=3e-06), cost=Cost(input_cache_miss=0.005535, input_cache_hit=0.0, input_cache_write=0.0, output=0.000153))}

if __name__ == "__main__":
    from appworld.task import Task
    from appworld import update_root
    update_root("./appworld")  
    predictor = APIEmbeddingPredictor(
        model_config={
            'embedding_name': 'unixcoder',
            'embedding_kwargs': {
                "model_name": "unixcoder",
                "model_path": "/data/dzl/package/model/unixcoder-base"
            },
            'faiss_kwargs': {
                'index_name': 'appworld_index',
                'index_dir': './src/retrieve_dataset/faiss_indexes',
            },
            'topk': 10
        },
        prompt_file_path="",
        demo_task_ids=[],
        max_predicted_apis=10,
        app_api_separator=".",
        mode="predicted",
    )
    task = Task.load("82e2fac_1", load_ground_truth=False)
    print("instruction:", task.instruction)
    predicted_apis, output = predictor.predict(task)
    print(predicted_apis)
    print(output)