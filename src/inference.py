#!/usr/bin/env python3
import os
import json

from tqdm import tqdm
import pandas as pd # 여기에 있으면 안됨. 함수화 필요

from configs import ConfigDefineTool
from data_engineering.dataset.precendent import CSVPrecendentDataset, install_pipeline # 이전사례 데이터 추상화 실패
from data_engineering.prompt_engineering.LLM_template import get_prompt_template
from data_engineering.prompt_engineering.precendent_to_docs import get_prompt_precendent
from data_engineering.prompt_engineering.precendent_to_question import get_prompt_question
from data_engineering.dataset.guideline import PDFDataset
from data_engineering.RAG import get_splitter, VectorstoreDefineTool
from core import load_LLM, ChainControlTool
from utils.now import get_now



def inference_exp(exp_name = "exp_1", num_output = -1):
    """
    (1) 선례 데이터 전처리 후 선례 docs 생성, 선례 데이터 
    (2) DF-기반 Retrievr와 PDF-기반 Retriever 중 선택(또는 병합) 가능
    (3) 결과를 CSV로 저장
    """
    config = ConfigDefineTool(exp_name)
    env = config.get_env()  # 한번만 호출하고 반환값 사용
    exp = config.get_exp()  # 한번만 호출하고 반환값 사용
    now = get_now("Asia/Seoul")

    # 경로 - path manager 추상화 실패
    path_train = os.path.join(env.PATH_DATA_DIR, exp.train)
    path_test = os.path.join(env.PATH_DATA_DIR, exp.test)
    paths_pdf = os.path.join(env.PATH_DATA_DIR, 'raw', '건설안전지침')
    paths_pdf = [os.path.join(paths_pdf, x) for x in os.listdir(paths_pdf)]
    path_exp = os.path.join(env.PATH_LOG_DIR, "exp", now)
    path_model = os.path.join(env.PATH_MODEL_DIR, exp.model_name)


    # 변수 선언
    encoding = exp.data_encoding
    pipeline = exp.data_pipeline
    version_prompt_precendent = exp.version_prompt_precendent
    version_prompt_question = exp.version_prompt_question

    retriever_precendent_name = exp.retriever_precendent_name
    retriever_guideline_name = exp.retriever_guideline_name
    embedding_model_name_precendent = exp.embedding_model_name_precendent
    embedding_model_name_guideline = exp.embedding_model_name_guideline
    prompt_template_format = exp.prompt_template_format
    retriever_precendent_params = exp.retriever_precendent_params
    retriever_guideline_params = exp.retriever_guideline_params
    splitter_precendent_name = exp.splitter_precendent_name
    splitter_precendent_kwargs = exp.splitter_precendent_kwargs
    splitter_guideline_name = exp.splitter_guideline_name
    splitter_guideline_kwargs = exp.splitter_guideline_kwargs
    
    model_strategy = exp.model_strategy
    model_name = exp.model_name
    model_kwargs = exp.model_kwargs

    chain_strategy = exp.chain_strategy
    chain_type = exp.chain_type

    # 테스트 데이터 로드
    pipeline = install_pipeline(pipeline)
    test_data = pd.read_csv(path_test, encoding = encoding)
    test_data = pipeline(test_data)

    
    # precendent 데이터
    if retriever_precendent_name is not None:
        # 데이터 로드
        precendent = pd.read_csv(path_train, encoding = encoding)
        precendent = pipeline(precendent) # 이전사례 데이터와 질문용 테스트 데이터는 전처리가 달라질 수 있음
        # 이전사례데이터 문서화 프롬프트 엔지니어링
        precendents = [] # 베이스라인모델 형식 유지하기 위해 이렇게 짬
        for i, row in precendent.iterrows():
            prec = get_prompt_precendent(row, exp = version_prompt_precendent)
            precendents.append(prec)
        # 청킹(일단 넣었지만 안 할 것 같음)
        if splitter_precendent_name is not None:
            splitter_precendent = get_splitter(
                splitter_name = splitter_precendent_name, 
                **splitter_precendent_kwargs
                )
        else:
            splitter_precendent = None
        # 벡터스토어 생성
        retriever_builder_precendent = VectorstoreDefineTool(
            factory_name = retriever_precendent_name, 
            embedding_model_name = embedding_model_name_precendent
            )
        # 벡터스토어에 문서 저장
        retriever_builder_precendent.add_documents(precendents, splitter_precendent)
        # 리트리버 생성 - 추후 1vector storage 2 retriever 컴바인 방식 쓸 때 변경 필요(add_retriever -> add_retrievers)
        retriever_builder_precendent.add_retriever(name = retriever_precendent_name, **retriever_precendent_params[0])
        retriever_precendent = retriever_builder_precendent.get_retrievers()[retriever_precendent_name]
    else:
        retriever_precendent = None

    # guideline 데이터
    if retriever_guideline_name is not None:
        # 데이터 로드
        guidelines = PDFDataset(paths_pdf)
        # 청킹
        if splitter_guideline_name is not None:
            splitter_guideline = get_splitter(
                splitter_name = splitter_guideline_name, 
                **splitter_guideline_kwargs
                )
        else:
            splitter_guideline = None
        # 벡터스토어 생성
        retriever_builder_guideline = VectorstoreDefineTool(
            factory_name = retriever_guideline_name, 
            embedding_model_name = embedding_model_name_guideline
            )
        # 벡터스토어에 문서 저장
        retriever_builder_guideline.add_documents(guidelines, splitter_guideline)
        # 리트리버 생성 - 추후 1vector storage 2 retriever 컴바인 방식 쓸 때 변경 필요(add_retriever -> add_retrievers)
        retriever_builder_guideline.add_retriever(name = retriever_guideline_name, **retriever_guideline_params[0])
        retriever_guideline = retriever_builder_guideline.get_retrievers()[retriever_guideline_name]
    else:
        retriever_guideline = None
    
    retrievers = {
        "precendent" : retriever_precendent,
        "guideline" : retriever_guideline
    }
    print("벡터스토어 완료")

    # LLM 모델 로드
    os.makedirs(path_model, exist_ok=True)
    llm = load_LLM(model_strategy, model_name, path_model, model_kwargs)


    print("모델로드완료")
    # LLM 체인
    prompt_template_format = get_prompt_template(exp = prompt_template_format)
    chain = ChainControlTool(
        llm=llm, 
        strategy=chain_strategy, 
        retrievers=retrievers, 
        prompt_template_format=prompt_template_format, 
        chain_type = chain_type
        )

    # 추론
    print("추론시작")
    i=0
    test_results = []
    for idx, row in tqdm(test_data.iterrows()):
        if i == num_output:
            break
        question = get_prompt_question(row, exp = version_prompt_question)
        print(question)
        result = chain.invoke(question)
        # 여기서 참고문서 기록 후 아래에서 결과와 같이 저장할 필요 있음 
        print(result)
        # final_result = result['result'] # 결과에 참조문서 출력해야 함함
        final_result = result
        test_results.append(final_result)
        i+=1
    
    # 저장
    os.makedirs(path_exp, exist_ok=True)
    path_param = os.path.join(path_exp, 'exp.json')
    path_result = os.path.join(path_exp, 'result.csv')
    with open(path_param, 'w') as f:
        json.dump(exp.to_dict(), f)
    pd.DataFrame(test_results, columns=['answer']).to_csv(path_result, index=False)



if __name__ == "__main__":
    import torch
    exp = input("실험 케이스:\n")
    num_output = int(input("추론 갯수\n"))

    print("CUDA 사용 가능:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU 개수:", torch.cuda.device_count())
        print("사용 중인 GPU:", torch.cuda.get_device_name(0))


    inference_exp(exp, num_output)
    