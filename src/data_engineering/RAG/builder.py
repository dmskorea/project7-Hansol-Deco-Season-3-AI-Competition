from typing import Iterable, List, Dict, Any
import importlib

from tqdm import tqdm

from .vector_store import BaseVSUnit, FAISSVSUnit



class VectorstoreDefineTool:
    def __init__(self, factory_name: str, embedding_model_name: str):
        """
        :param factory: 벡터스토어 생성에 사용할 팩토리
        :param embedding_model_name: 사용할 임베딩 모델 이름
        """
        self.factory_name = factory_name
        self.embedding_model_name = embedding_model_name
        self.docs: List[str] = []
        self.vector_store = None
        self.retrievers: dict[Any] = {}

        module = importlib.import_module(".vector_store", package=__package__)
        cls = getattr(module, factory_name)
        self.factory = cls()

    def add_documents(self, docs: Iterable[str], spliter = None) -> "VStoreDefineTool":
        """
        문서를 추가합니다.
        만약 텍스트 스플리터가 설정되어 있다면, 각 문서를 청킹하여 추가합니다.
        """
        for doc in tqdm(docs, desc="Adding documents"):
            if spliter is not None:
                # 문서를 청킹하여 여러 청크로 분할
                chunks = spliter.split_text(doc)
                self.docs.extend(chunks)
            else:
                self.docs.append(doc)
        self.build_vectorstore()
        return self

    def build_vectorstore(self) -> "VStoreDefineTool":
        """
        저장된 문서를 기반으로 벡터스토어를 생성합니다.
        """
        self.vector_store = self.factory.create_vectorstore(self.docs, self.embedding_model_name)
        return self

    def add_retriever(self, name, search_type: str = "similarity", search_kwargs: Dict[str, Any] = {"k": 5}) -> "VStoreDefineTool":
        """
        생성된 벡터스토어에 리트리버를 추가합니다.
        하나의 벡터스토어에 여러 리트리버를 추가할 수 있습니다.
        """
        if not self.vector_store:
            raise ValueError("먼저 build_vectorstore()를 호출하여 벡터스토어를 생성하세요.")
        retriever = self.vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        self.retrievers[name] = retriever
        return self
    
    def add_retrievers(self, param_groups): # 추후 적용... 한 벡터스토어에 두 리트리버 쓰고 컴바인 할 때
        for params in param_groups:
            self.add_retriever(**params)

    def get_vectorstore(self) -> Dict[str, Any]:
        """
        생성된 벡터스토어를 반환합니다.
        """
        return self.vector_store

    def get_retrievers(self) -> List[Any]:
        """
        생성된 모든 리트리버를 반환합니다.
        """
        return self.retrievers
    
