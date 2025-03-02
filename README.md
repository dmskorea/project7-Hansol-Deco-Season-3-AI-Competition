#### 실험 리더보드

| 유형 |  valid  |  test | running time            | code                            | 비고  |
|------|---------|-------|-------------------------|---------------------------------|-------|
| 1등  |         | 0.534 |                         |                                 |       |
| RAG  |         |       | valid:11min, test:48min | rag_model_v5_4_6.ipynb | similarity_score_threshold 적용 (score_threshol:0.7) |
| RAG  |         |       | valid:11min, test:48min | rag_model_v5_4_5.ipynb | similarity_score_threshold 적용 (score_threshol:0.6) |
| RAG  |         |       | valid:11min, test:48min | rag_model_v5_4_4.ipynb | similarity_score_threshold 적용 (score_threshol:0.5) |
| RAG  |  0.626  | 0.473 | valid:11min, test:48min | rag_model_v5_4_3.ipynb | question-answer 정보 추가 |
| RAG  |  0.624  | 0.469 | valid:11min, test:48min | rag_model_v5_4_2.ipynb | 프롬프트 수정 |
| RAG  |  0.620  | 0.468 | valid:11min, test:48min | rag_model_v5_4.ipynb  | 프롬프트 수정 |




