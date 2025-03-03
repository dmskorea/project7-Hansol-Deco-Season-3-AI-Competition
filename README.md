#### 로컬실험결과 리더보드
- model : MLP-KTLim/llama-3-Korean-Bllossom-8B (float16)
- vector_store.as_retriever: similarity : 10
- pipeline: temperature 0.1, 0.2
- pipeline: max_new_tokens 64
- RetrievalQA.from_chain_type : stuff

| 유형 |  valid  |  test | running time            | code                   | 비고                                        |
|------|---------|-------|-------------------------|------------------------|---------------------------------------------|
| 1등  |         | 0.534 |                         |                        |                                             |
| FINE |         |       |                         | rag_model_v9_0_0.ipynb | 파인튜닝 모델 적용 (PDF파일 학습)   |
| FINE |         |       |                         | rag_model_v8_0_0.ipynb | 파인튜닝 모델 적용 (중요정보 추출)  |
| FINE |         |       |                         | rag_model_v7_0_0.ipynb | 증강 데이터 활용   |              
| RAG  |         |       |                         | rag_model_v6_0_0.ipynb | context 내용 제공  |
| RAG  |  0.633  | 0.476 | valid:13min, test: 1hr  |                        | pipeline: max_new_tokens 64 -> 80 |
| RAG  |         |       |                         |                        | pipeline: max_new_tokens 64 -> 70 |
| RAG  |         |       |                         |                        | pipeline: max_new_tokens 64 -> 50 |
| RAG  |         |       |                         |                        | pipeline: temperature 0.1 -> 0.3 |
| RAG  |         |       |                         |                        | pipeline: temperature 0.1 -> 0.05 |
| RAG  |  0.629  | 0.474 | valid:20min, test: 1hr  |                        | pipeline: temperature 0.1 -> 0.2 |
| RAG  |  0.624  |       |                         |                        | vector_store.as_retriever: similarity k 10 -> 7 |
| RAG  |  0.620  |       |                         |                        | vector_store.as_retriever: similarity k 10 -> 5 |
| RAG  |  0.110  |       | valid:45min             |                        | vector_store.as_retriever: similarity k 10 -> 30 |
| RAG  |  0.626  | 0.473 | valid:11min, test:48min | rag_model_v5_4_3.ipynb | question-answer 정보 추가 (사고월, 사고요일, 사고시간) |
| RAG  |  0.510  |       | valid:11min             | rag_model_v5_4_3.ipynb | question-answer 정보 요약해서 핵심만 제공  |
| RAG  |  0.624  | 0.469 | valid:11min, test:48min | rag_model_v5_4_2.ipynb | 프롬프트 수정 (베스트정답 제시) |
| RAG  |  0.620  | 0.468 | valid:11min, test:48min | rag_model_v5_4.ipynb   | 프롬프트 수정 (인적사고유형별 베스트정답 제시) |




