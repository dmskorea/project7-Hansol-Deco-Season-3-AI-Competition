import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_community.llms.vllm import VLLM
from langchain_community.llms import HuggingFacePipeline



class LLMDefineTool:
    '''
    LLM 훈련 안해봤기때문에
    일단 유인원식 클래스
    새로운 케이스가 필요한 경우 몽키패치로 메소드 추가
    '''
    def load_base(
            model_name: str = "NCSOFT/Llama-VARCO-8B-Instruct", 
            cache_dir: str = None,
            model_kwargs: dict = {}
            ):
        """
        허깅페이스 모델 그대로 쓰는 함수
        추후 훈련 고려해서 변경
        """
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir = cache_dir,
            quantization_config=bnb_config,
            device_map="auto"
        )

        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            **model_kwargs
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        return llm



    def load_vllm(
            model_name: str = "NCSOFT/Llama-VARCO-8B-Instruct", 
            cache_dir: str = None,
            model_kwargs: dict = {}
            ):
        llm = VLLM(
            model=model_name,
            download_dir = cache_dir,
            dtype="float16",  # or torch.float16, for FP16 precision&#8203;:contentReference[oaicite:2]{index=2}
            vllm_kwargs={     # extra vLLM.LLM options passed through to the vLLM engine
                "quantization": "bitsandbytes",      # use 4-bit bitsandbytes quantization&#8203;:contentReference[oaicite:3]{index=3}
                "load_format": "bitsandbytes",       # load weights in BitsAndBytes 4-bit format&#8203;:contentReference[oaicite:4]{index=4}
                "gpu_memory_utilization": 0.9,
                # "trust_remote_code": True  # enable if the model requires remote code execution
            },
            **model_kwargs,
        )

        return llm
    

def load_LLM(case = "load_vllm", model_name = "NCSOFT/Llama-VARCO-8B-Instruct", cache_dir = None, model_kwargs = {}):
    '''
    control_params는 일단 llm 출력에 영향주는
    model_name
    temperature
    top_p 
    top_k
    max_new_tokens
    이렇게만 사용
    '''
    return getattr(LLMDefineTool, case)(model_name, cache_dir, model_kwargs)