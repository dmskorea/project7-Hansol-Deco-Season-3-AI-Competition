import importlib

def get_splitter(splitter_name: str, **kwargs):
    """
    주어진 클래스 경로 문자열로부터 텍스트 스플리터 클래스를 동적으로 로드하여 인스턴스를 반환합니다.
    """
    module = importlib.import_module(".chunking", package=__package__)
    cls = getattr(module, splitter_name)
    return cls(**kwargs)

if __name__ == "__main__":
    print(get_splitter("RecursiveCharacterTextSplitter"), {"chunk_size":100, "chunk_overlap" : 20})