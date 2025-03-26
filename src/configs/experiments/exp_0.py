from dataclasses import dataclass, field

from .base import Experiment  # 별도 파일에 선언된 dataclass



@dataclass    
class Experiment_0(Experiment):
    '''
    실험 파라미터
    chain / data / prompt / RAG / model로 나누어 관리
    '''
    exp_name :str = "exp_0"
    # chain 파라미터
    chain_strategy : str = "baseline"
    chain_type :str = "stuff"

    # data 파라미터
    train: str = 'raw/train.csv'
    test: str =  'raw/test.csv'
    data_encoding : str = "utf-8-sig"
    data_pipeline : str =  "pipeline_0"

    # prompt 파라미터
    prompt_template_format : str = "exp_0"
    version_prompt_precendent : str = "exp_0"
    version_prompt_question : str = "exp_0"

    # RAG 파라미터
    embedding_model_name_precendent :str = "jhgan/ko-sbert-nli"

    retriever_precendent_name : str = "FAISSVSUnit"
    retriever_precendent_params : tuple = field(default_factory=lambda: 
        (
            (
                {
                    "search_type" : "similarity",
                    "search_kwargs" : {
                        "k" : 5
                    }
                },
            )
        )
    )
    splitter_precendent_name : str = None
    splitter_precendent_kwargs : dict = None

    embedding_model_name_guideline :str = None
    retriever_guideline_name : str = None
    retriever_guideline_params : tuple = None
    splitter_guideline_name : str = None
    splitter_guideline_kwargs : dict = None

    # model 파라미터
    model_strategy : str = 'load_base'
    model_name : str = "NCSOFT/Llama-VARCO-8B-Instruct"
    model_kwargs : dict = field(default_factory=lambda: 
        {
            "temperature" : 0.1
        }
    )
