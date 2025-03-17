from typing import List, Any
from abc import ABC, abstractmethod



# 1. Abstract Factory: 벡터스토어 생성 인터페이스 정의
class BaseVSUnit(ABC):
    @abstractmethod
    def create_vectorstore(self, docs: List[str], embedding_model_name: str) -> Any:
        """
        주어진 문서 리스트와 임베딩 모델 이름으로 벡터스토어 인스턴스를 생성합니다.
        """
        pass



