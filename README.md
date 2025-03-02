#### 실험 리더보드

| 유형 |  valid  |  test | running time            | code                            | 비고                                        |
|------|---------|-------|-------------------------|---------------------------------|---------------------------------------------|
| 1등  |         | 0.534 |                         |                                 |                                             |
| RAG  |         |       | valid:11min, test:48min | rag_model_v9_0_0.ipynb | 파인튜닝 모델 적용 (PDF파일 학습)                      |
| RAG  |         |       | valid:11min, test:48min | rag_model_v8_0_0.ipynb | 파인튜닝 모델 적용 (중요정보 추출)                     |
| RAG  |         |       | valid:11min, test:48min | rag_model_v7_0_0.ipynb | 증강 데이터 활용                                      |
| RAG  |         |       | valid:11min, test:48min | rag_model_v6_0_0.ipynb | context 내용 제공                                    |
| RAG  |         |       | valid:11min, test:48min |                        | similarity k 10 -> 7 |
| RAG  |         |       | valid:11min, test:48min |                        | similarity k 10 -> 5 |
| RAG  |         | 0.112 | valid:11min, test:48min |                        | similarity k 10 -> 30 |
| RAG  |  0.626  | 0.473 | valid:11min, test:48min | rag_model_v5_4_3.ipynb | question-answer 정보 추가 (사고월, 사고요일, 사고시간) |
| RAG  |  0.624  | 0.469 | valid:11min, test:48min | rag_model_v5_4_2.ipynb | 프롬프트 수정 (베스트정답 제시)                        |
| RAG  |  0.620  | 0.468 | valid:11min, test:48min | rag_model_v5_4.ipynb   | 프롬프트 수정 (인적사고유형별 베스트정답 제시)          |




