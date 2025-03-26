from typing import List

from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .base import BaseVSUnit



class FAISSVSUnit(BaseVSUnit):
    '''
    recursive character text splitter와 FAISS를 사용한 벡터스토어 생성 클래스
    '''
    def __init__(self):
        self.spliter = None

    def create_vectorstore(self, docs: List[str], embedding_model_name: str) -> FAISS:
        embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vector_store = FAISS.from_texts(docs, embedding)
        return vector_store