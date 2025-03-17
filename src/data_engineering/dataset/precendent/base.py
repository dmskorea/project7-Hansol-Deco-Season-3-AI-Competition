import abc
from typing import List, Any, Dict, Callable



class BasePrecendentDataset(metaclass=abc.ABCMeta):
    """
    이전 사례 데이터 모델의 기본 클래스
    """
    def __init__(
        self,
        parser = None,
        pipeline = None
    ):
        self.parser = parser
        self.pipeline = pipeline


    @abc.abstractmethod
    def get_source(self):
        pass

    def parse(self, records):
        if self.parser is not None:
            return self.parser(records)
        return records
    
    def process(self, records):
        if self.pipeline is not None:
            return self.pipeline(records)
        return records
