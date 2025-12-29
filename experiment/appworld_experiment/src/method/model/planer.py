from src.utils.common.io import read_file
from typing import Any, cast
import os
from src.utils.language_model import LanguageModel
from src.utils.common.prompts import load_prompt_to_chat_messages
from src.utils.common.text import render_template
from src.utils.api_EmbeddingPredictor import APIEmbeddingPredictor
from src.utils.process_config import set_api_key_by_model_config
import re

class plan_object:
    def __init__(self, plan_steps: list[str], original_plan: str, retrieved_apis: list[str]=None):
        self.plan_steps = plan_steps
        self.original_plan = original_plan
        if retrieved_apis is None:
            self.retrieved_apis = []
        else:
            self.retrieved_apis = retrieved_apis

class planer:
    #与appworld无关，尽量独立执行，参数可扩展且通用
    def __init__(self, **kwargs):

        plan_prompt_template_path = kwargs.get("plan_prompt_template_path", "")
        self.plan_prompt_template = cast(str, read_file(plan_prompt_template_path.replace("/", os.sep)))

        plan_prior_template_path = kwargs.get("plan_prior_template_path", "")
        self.plan_prior_template = cast(str, read_file(plan_prior_template_path.replace("/", os.sep)))

        self.max_plan_samples_num = kwargs.get("max_plan_samples_num", 1)
        self.LLM_model_config = kwargs.get("model_config", None)  #传入一个语言模型实例

        set_api_key_by_model_config(kwargs.get("model_config", {}))
        self.language_model = LanguageModel(**self.LLM_model_config)
        self.plan_prior_list = []
        self.sub_plan_topk = kwargs.get("sub_plan_topk", 3)
        
    def construct_plan_prompt(
        self,
        instruction,
        api_documentation_string: str,
        app_descriptions: str = "",
    ) -> str:
        prompt_content = render_template(
            self.plan_prompt_template,
            api_documentation_string=api_documentation_string,
            instruction=instruction,
            app_descriptions=app_descriptions,
        )
        return prompt_content
    
    def construct_plan_prior_prompt(
        self,
        instruction,
        app_descriptions,
    ) -> str:
        prompt_content = render_template(
            self.plan_prior_template,
            instruction=instruction,
            app_descriptions=app_descriptions,
        )
        return prompt_content
    
    def robust_post_process_plan_text(self, plan: str) -> list[str]:
        # 提取所有 <step>...</step> 内容，忽略换行和空格
        steps = re.findall(r"<step>\s*(.*?)\s*</step>", plan, re.DOTALL)
        # 如果没有 <step> 标签，则按行分割并去除空行
        if not steps:
            steps = [step.strip() for step in plan.split("\n") if step.strip()]
        return steps

    def post_process_plan(
        self,
        plan: str,
    ) -> list[str]:
        plan_steps = self.robust_post_process_plan_text(plan)
        #print("Extracted plan steps:", plan_steps)
        plan_obj = plan_object(plan_steps=plan_steps, original_plan=plan)
        return plan_obj
    
    def sample_plans(
        self,
        prompt_messages: str,
    ) -> list[plan_object]:
        prompt_messages = load_prompt_to_chat_messages(prompt_messages, skip_system_message=False)
        plan_samples = []
        for _ in range(self.max_plan_samples_num):
            plan_message = self.language_model.generate(
                prompt_messages,
                cache_control_at=-1
            )
            # print("prompt_messages", prompt_messages)
            # print("plan_message", plan_message)
            # input("Press Enter to continue...")
            plan_obj = self.post_process_plan(plan_message['content'])
            plan_samples.append(plan_obj)
        return plan_samples
    
    def retrieve_docs_by_plan_prior(
        self, instruction: str, plan_prior_list: list[plan_object]) -> str:
        raise NotImplementedError("retrieve_docs_by_plan_prior method is not implemented yet.")


    def show_plan_plan_prior_list(self):
        for idx, plan_prior in enumerate(self.plan_prior_list):
            print(f"Plan Prior Sample {idx + 1}:")
            print(f"Original Plan:\n{plan_prior.original_plan}")
            print("Plan Steps:")
            for step in plan_prior.plan_steps:
                print(f"- {step}")
            print("Retrieved APIs:")
            for apis in plan_prior.retrieved_apis:
                print(f"- {apis}")
            print("\n")

    def generate_plan(self, is_sample_plans:bool, instruction: str, app_descriptions: str) -> str:
        
        if is_sample_plans:
            plan_prior_prompt_messages = self.construct_plan_prior_prompt(instruction, app_descriptions)
            self.plan_prior_list = self.sample_plans(plan_prior_prompt_messages)
            api_documentation_string, self.plan_prior_list, plan_prior_samples_retrieved_apis,instruction_predicted_apis = self.retrieve_docs_by_plan_prior(instruction, self.plan_prior_list)
            #self.show_plan_plan_prior_list()
            
        
        

        plan_prompt_messages = self.construct_plan_prompt(
            instruction=instruction,
            api_documentation_string=api_documentation_string,
            app_descriptions=app_descriptions,
        )
        prompt_messages = load_prompt_to_chat_messages(plan_prompt_messages, skip_system_message=False)
        message_ = self.language_model.generate(
            prompt_messages,
            cache_control_at=-1
        )
        generated_text = message_["content"] or ""
        return generated_text, api_documentation_string, plan_prior_samples_retrieved_apis,instruction_predicted_apis