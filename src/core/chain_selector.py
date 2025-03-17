from langchain.prompts import PromptTemplate

from .chain import BaselineChainUnit, DualChainUnit


# ChainControlTool 클래스: strategy 문자열을 받아 적절한 전략을 선택하는 Factory 역할 수행
class ChainControlTool:
    def __init__(self, llm, strategy: str, retrievers: dict, prompt_template_format: str, chain_type:str="stuff"):
        """
        Parameters:
            llm: LLM 호출 함수 또는 객체 (callable)
            strategy: "baseline" 또는 "dual" 문자열로 선택
            retrievers:
                - baseline: {"precendent": <retriever>}
                - dual: {"guideline": <retriever>, "precendent": <retriever>}
            prompt_template:
                - baseline: PromptTemplate 객체 (input_variables=["context", "question"])
                - dual: PromptTemplate 객체 (input_variables=["guideline_context", "precendent_context", "question"])
            chain_type: RetrievalQA.from_chain_type에 사용할 체인 타입 ("stuff", "map_reduce" 등)
        """
        self.llm = llm
        self.strategy_name = strategy.lower()
        self.retrievers = retrievers
        self.prompt_template_format = prompt_template_format
        self.chain_type = chain_type

        self.strategy = self._create_strategy()


    def _create_strategy(self) -> BaselineChainUnit:
        if self.strategy_name == "baseline": # 조건문기반 함수나 클래스로 변경 필요
            # baseline 모드는 retrievers에 "precendent" 키만 존재해야 함
            if self.retrievers["precendent"] is None or self.retrievers["guideline"] is not None:
                raise ValueError("baseline 모드는 'precendent' retriever 하나만 사용해야 합니다.")
            
            self.prompt_template = PromptTemplate(
                input_variables = ["context", "question"],
                template = self.prompt_template_format.template
            )

            return BaselineChainUnit(
                llm=self.llm,
                retriever=self.retrievers["precendent"],
                prompt_template=self.prompt_template,
                chain_type=self.chain_type
            )
        
        elif self.strategy_name == "dual":
            # dual 모드는 retrievers에 "guideline"과 "precendent"가 모두 있어야 함
            if self.retrievers["precendent"] is None or self.retrievers["guideline"] is None:
                raise ValueError("dual 모드는 'guideline'과 'precendent' retriever 모두 필요합니다.")
            # 프롬프트 템플릿에 필요한 변수가 존재하는지 
            
            self.prompt_template = PromptTemplate(
                input_variables = ["context_precendent", "context_guideline", "question"],
                template = self.prompt_template_format.template
            )
            
            return DualChainUnit(
                llm=self.llm,
                retrievers=self.retrievers,
                prompt_template=self.prompt_template
            )
        else:
            raise ValueError(f"지원하지 않는 전략입니다: {self.strategy_name}")

    def invoke(self, query: str):
        return self.strategy.invoke(query)