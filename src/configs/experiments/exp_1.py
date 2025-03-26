from dataclasses import dataclass

from .exp_0 import Experiment_0  # 별도 파일에 선언된 dataclass



@dataclass    
class Experiment_1(Experiment_0):
    exp_name : str = 'exp_1'
    train : str = "sample/v1/train.csv"
    test :str = "sample/v1/test.csv"
    model_strategy : str = 'load_vllm'