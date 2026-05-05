from typing import Any

from appworld.task import Task
from src.utils.common.api_predictor import VALID_MODES_LITERAL
from src.utils.common.api_predictor import APIPredictor as _APIPredictor
from src.utils.common.usage_tracker import Usage
from src.utils.language_model import LanguageModel
import os
from src.utils.process_config import set_api_key_by_model_config
from src.utils.json_util import save_data_to_json, load_list_from_json



class APIPredefiner:  # type: ignore[misc]
    def __init__(
        self,
        api_file_path: str,
    ):
     
      data_list = load_list_from_json(api_file_path)
      self.api_dict = {}
      self.categories = ['api_docs','supervisor','amazon','phone','file_system','spotify','venmo','gmail','simple_note','todoist']
      for data in data_list:
          selected_apis = self.consturct_apis(data["selected_apis"])
          self.api_dict[data["index"]] = selected_apis
      
    
    def consturct_apis(self, selected_apis: list[str]):
      #print(selected_apis)
      overall_apis = []
      overall_apis.extend(["supervisor.complete_task","supervisor.show_account_passwords","supervisor.show_profile"])
      app_name_set = set()
      for api in selected_apis:
          app_name = api.split('.', 1)[0]
          if app_name not in self.categories: continue
          if app_name != "supervisor":
              app_name_set.add(app_name)

      for app_name in app_name_set:
          overall_apis.append(f"{app_name}.login")

      for api in selected_apis:
        app_name = api.split('.', 1)[0]
        if app_name not in self.categories: continue
        
        overall_apis.append(api)

      overall_apis = list(set(overall_apis))
      #print(overall_apis)
      #input("press")
      return overall_apis

    def predict(
        self, task: Task, lm_calls_log_file_path: str | None = None
    ) -> tuple[list[str], dict[str, Any]]:

        task_id = task.id
        predicted_apis = self.api_dict[task_id]
        content = "\n".join(predicted_apis)
        #print(f"predicted_apis: {predicted_apis}")
        #input("press")
        return predicted_apis, {"content": content, "standardized_usage": Usage()}
       
        
        
    
  
