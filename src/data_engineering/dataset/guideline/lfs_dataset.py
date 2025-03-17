from torch.utils.data import Dataset
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



class PDFDataset(Dataset):
    def __init__(self, pdf_paths, chunk_size=1000, chunk_overlap=100):
        """
        Lazy Loading을 유지하면서 PDF별로 chunking을 수행하는 Dataset
        Args:
            pdf_paths (list): PDF 파일 경로 리스트
            chunk_size (int): 하나의 chunk 크기
            chunk_overlap (int): chunk 간 오버랩 크기
        """
        self.pdf_paths = pdf_paths
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.chunks = self._build_index()

    def _build_index(self):
        """각 PDF를 페이지별로 로드한 후 chunking하여 인덱스 생성"""
        chunks = []
        for pdf_path in self.pdf_paths:
            loader = PyPDFLoader(pdf_path)
            doc_list = loader.load()  # PDF의 페이지별 Document 리스트
            split_docs = self.text_splitter.split_documents(doc_list)  # chunking 수행
            chunks.extend(split_docs)  # 모든 chunk를 리스트에 추가
        return chunks

    def __len__(self):
        return len(self.chunks)  # 전체 chunk 개수 반환

    def __getitem__(self, idx):
        return self.chunks[idx].page_content  # 특정 chunk 반환
