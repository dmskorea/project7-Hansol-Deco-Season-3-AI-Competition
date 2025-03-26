from dataclasses import dataclass

from .exp_2 import Experiment_2  # 별도 파일에 선언된 dataclass



@dataclass    
class Experiment_3(Experiment_2):
    exp_name : str = "exp_3"
    data_pipeline : str = "pipeline_2" # 주요변경
    version_prompt_precendent : str = "exp_2" # 주요변경
    version_prompt_question : str = "exp_1" # 주요변경