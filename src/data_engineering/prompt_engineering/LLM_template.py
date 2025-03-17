from dataclasses import dataclass, asdict


'''
context와 question을 가지고 최종적으로 LLM에 입력할 프롬프트
'''


@dataclass
class Template:
    template: str = """
    ### 지침: 당신은 건설 안전 전문가입니다.
    질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
    - 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
    - 다음과 같은 조치를 취할 것을 제안합니다: 와 같은 표현을 포함하지 마세요.

    {context}

    ### 질문:
    {question}

    [/INST]
    """

    def to_dict(self):
        asdict(self)

    @classmethod
    def exp_0(cls):
        return cls()
    
    @classmethod
    def exp_1(cls):
        return cls(
            template = """지침: 당신은 건설 안전 전문가입니다.
            최근 발생한 사고와 관련하여 어떤 조치를 취해야하는지 알려주세요.
            ### 질문 : {question}

            ### 다음 문서를 참고하세요
            - 이전 사례 데이터 : {context_precendent}
            - 사고 관련 문서 : {context_guideline}

            답변 형식은 대책별로 짧은 문장으로 답하고, 원인 분석은 절대 말하지 말고 대책만 답하세요.
"""
        )
    
    @classmethod
    def exp_2(cls):
        return cls(
            template = """[지침]
당신은 건설 안전 전문가입니다. 최근 발생한 사고와 관련하여 어떤 조치를 취해야하는지 정답을 맞추시오. 절대 부가설명을 하지 말고 간략히 적으시오.

[관련문서]
{context_guideline}
[문제]
{context_precendent}
{question}
"""
        )
    


def get_prompt_template(exp = "exp_0"):
    return getattr(Template, exp)()

if __name__ == "__main__":
    prompot_template = get_prompt_template(exp = "exp_0")
    print(prompot_template.template)