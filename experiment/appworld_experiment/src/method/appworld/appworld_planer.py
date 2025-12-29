from src.method.model.planer import planer, plan_object
from src.utils.api_EmbeddingPredictor import APIEmbeddingPredictor


class AppWorldPlaner(planer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        api_predictor_config = kwargs.get("api_predictor_config", {})
        
        self.predictor = APIEmbeddingPredictor(
            model_config= api_predictor_config,
            prompt_file_path="",
            demo_task_ids=[],
            max_predicted_apis=api_predictor_config.get("max_predicted_apis", api_predictor_config['topk']),
            app_api_separator=".",
            is_simple= api_predictor_config.get("is_simple", False)
        )
        
    def convert_apis_list_to_string(self, apis_list: list[str]) -> str:
        apis_list = list(set(apis_list))
        #print(len(apis_list))
        return "\n".join(apis_list)
    
    def retrieve_docs_by_plan_prior(
        self, instruction: str, plan_prior_list: list[plan_object]) -> str:
        retrieved_docs = []
        instruction_predicted_apis, _ = self.predictor.predict_by_instruction(instruction)  
        plan_prior_samples_retrieved_apis = []
        plan_prior_samples_retrieved_apis.extend(instruction_predicted_apis)

        for plan_prior in plan_prior_list:
            for plan_step in plan_prior.plan_steps:
                predicted_apis, _ = self.predictor.predict_by_instruction(plan_step, igonre_complete_api=True)    
                plan_prior.retrieved_apis.append({f"{plan_step}":predicted_apis[:self.sub_plan_topk]})
                plan_prior_samples_retrieved_apis.extend(predicted_apis[:self.sub_plan_topk])
        retrieved_docs = self.convert_apis_list_to_string(plan_prior_samples_retrieved_apis)
       
        return retrieved_docs, plan_prior_list,plan_prior_samples_retrieved_apis,instruction_predicted_apis