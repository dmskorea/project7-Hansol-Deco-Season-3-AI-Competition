

# ----- 최종 결과 CSV 저장 -----
submission = pd.read_csv(sample_submission_path, encoding='utf-8-sig')
submission.iloc[:, 1] = test_results  # 예: answer 열만 교체
# 필요 시 임베딩 컬럼 추가
# for i in range(pred_embeddings.shape[1]):
#     submission[f'emb_{i}'] = pred_embeddings[:, i]

submission.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
