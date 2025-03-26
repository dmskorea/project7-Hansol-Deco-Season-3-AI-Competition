
from typing import Dict, Any

import pandas as pd

from .base import BasePrecendentDataset

class CSVPrecendentDataset(BasePrecendentDataset):
    """
    CSV 파일을 Dataset 형태로 로드하는 클래스
    """
    def __init__(
        self, csv_path: str, encoding: str = 'utf-8-sig',
        parser = None,
        pipeline = None,
        **read_csv_kwargs
    ):
        """
        csv_path: 데이터 파일 경로
        encoding: 파일 인코딩
        parser: 데이터 전처리 함수 (옵션)
        pipeline: 추가 변환을 위한 함수 (옵션)
        read_csv_kwargs: pandas read_csv 옵션들
        """
        self.csv_path = csv_path
        self.parser = parser
        self.pipeline = pipeline
        
        # CSV 파일을 한 번 로드하여 DataFrame으로 유지
        self.data = self.get_source(csv_path, encoding, read_csv_kwargs)
        
        # (선택적) 파싱 및 추가 변환 적용
    def get_source(self, csv_path, encoding, read_csv_kwargs):
        data = pd.read_csv(csv_path, encoding=encoding, **read_csv_kwargs)
        if self.parser:
            data = self.parser(data)
        if self.pipeline:
            data = self.pipeline(data)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        DataLoader에서 개별 row(dict) 반환
        """
        row = self.data.iloc[idx].to_dict()
        return row

