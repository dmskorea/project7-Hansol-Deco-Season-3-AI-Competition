from langchain.chains import RetrievalQA

from .base import BaseChainUnit



class BaselineChainUnit(BaseChainUnit):
    def __init__(self, llm, retriever, prompt_template, chain_type="stuff"):
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
    
    def invoke(self, query: str):
        return self.chain.invoke(query)
