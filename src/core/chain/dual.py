from .base import BaseChainUnit



class DualChainUnit(BaseChainUnit):
    def __init__(self, llm, retrievers: dict, prompt_template):
        # retrievers: {"guideline": guideline_retriever, "precendent": precendent_retriever}
        self.llm = llm
        self.retrievers = retrievers
        self.prompt_template = prompt_template
        
    def invoke(self, query: str):
        guideline_docs = self.retrievers["guideline"].invoke(query)
        precendent_docs = self.retrievers["precendent"].invoke(query)
        
        guideline_context = "\n".join([doc.page_content for doc in guideline_docs])
        precendent_context = "\n".join([doc.page_content for doc in precendent_docs])
        
        formatted_prompt = self.prompt_template.format(
            context_guideline=guideline_context,
            context_precendent=precendent_context,
            question=query
        )
        return self.llm.invoke(formatted_prompt)
