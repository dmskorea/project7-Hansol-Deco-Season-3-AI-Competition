from dataclasses import dataclass, field

from .exp_3 import Experiment_3  # 별도 파일에 선언된 dataclass



@dataclass    
class Experiment_4(Experiment_3):
    exp_name : str = "exp_4"
    model_kwargs : dict = field(default_factory=lambda: 
        {
            "temperature" : 0.1,
            "max_new_tokens" : 200,
            "frequency_penalty" : 2.0,
            "presence_penalty" : 0.1,
        }
    )