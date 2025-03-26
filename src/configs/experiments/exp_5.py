from dataclasses import dataclass, field

from .exp_4 import Experiment_4  # 별도 파일에 선언된 dataclass



@dataclass    
class Experiment_5(Experiment_4):
    exp_name : str = "exp_5"
    version_prompt_precendent : str = "exp_3" # 주요변경
    version_prompt_question : str = "exp_2" # 주요변경
    prompt_template_format : str = "exp_2" # 주요변경