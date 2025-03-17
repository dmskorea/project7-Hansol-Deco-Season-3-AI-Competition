import numpy as np
from sentence_transformers import SentenceTransformer



def cosine_similarity(a, b):
    """코사인 유사도 계산"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0


def jaccard_similarity(text1, text2):
    """자카드 유사도 계산"""
    set1, set2 = set(text1.split()), set(text2.split())  # 단어 집합 생성
    intersection = len(set1.intersection(set2))  # 교집합 크기
    union = len(set1.union(set2))  # 합집합 크기
    return intersection / union if union != 0 else 0


def calculate_similarities(trues, preds):
    embedding_model_name = "jhgan/ko-sbert-sts"
    st_embedding = SentenceTransformer(embedding_model_name)
    cossims = []
    jaccardsims = []
    for text_true, text_pred in zip(trues, preds):
        embedding_true = st_embedding.encode(text_true)
        embedding_pred = st_embedding.encode(text_pred)
        cossim = cosine_similarity(embedding_true, embedding_pred)
        jaccardsim = jaccard_similarity(text_true, text_pred)
        cossims.append(cossim)
        jaccardsims.append(jaccardsim)
    return cossims, jaccardsims


def scoring(cossims, jaccardsims):
    l = 0
    s = 0
    for cossim, jaccardsim in zip(cossims, jaccardsims):
        s += max(cossim,0)*0.7 + max(jaccardsim, 0)*0.3
        l += 1
    return s/l

if __name__ == "__main__":
    import pandas as pd
    pred_sample = pd.read_csv("/workspace/Storage/hansoldeco3/Data/sample_result.csv")
    true_sample = pd.read_csv("/workspace/Storage/hansoldeco3/Data/sample/v1/test.csv")
    cossims, jaccardsims = calculate_similarities(true_sample['재발방지대책 및 향후조치계획'], pred_sample['answer'])
    score = scoring(cossims, jaccardsims)
    print(score)

    
    pred_sample = pd.read_csv("/workspace/Storage/hansoldeco3/Data/vllm_result.csv")
    true_sample = pd.read_csv("/workspace/Storage/hansoldeco3/Data/sample/v1/test.csv")
    cossims, jaccardsims = calculate_similarities(true_sample['재발방지대책 및 향후조치계획'], pred_sample['answer'])
    score = scoring(cossims, jaccardsims)
    print(score)