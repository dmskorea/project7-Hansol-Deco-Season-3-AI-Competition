import pandas as pd



class Question:
    def exp_0(row):
        question = (
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
            f"대책은 무엇인가요?"
            )
        return question
    
    def exp_1(row):
        components = []
        components.append("다음과 같은 상황에서 공사 중 사고가 발생했습니다.")
        
        # 시간 관련 항목
        if not pd.isna(row.get('계절')):
            components.append(f"계절: {row['계절']}")
        if not pd.isna(row.get('시간대')):
            components.append(f"시간대: {row['시간대']}")
        if not pd.isna(row.get('날씨')):
            components.append(f"날씨: {row['날씨']}")
        if not pd.isna(row.get('기온')):
            components.append(f"기온: {row['기온']}도")
        if not pd.isna(row.get('습도')):
            components.append(f"습도: {row['습도']}%")
        if not pd.isna(row.get('장소')):
            components.append(f"장소: {row['장소']}")
        
        # 구조물 층수: 지상, 지하
        level_parts = []
        if not pd.isna(row.get('지상')):
            level_parts.append(f"지상 {row['지상']}층")
        if not pd.isna(row.get('지하')):
            level_parts.append(f"지하 {row['지하']}층")
        if level_parts:
            components.append("구조물: " + ", ".join(level_parts) + "에서 사고 발생")
        
        # 공사종류 관련 (대분류, 중분류, 소분류)
        cons_parts = []
        if not pd.isna(row.get('공사종류_대분류')):
            cons_parts.append(f"대분류: {row['공사종류_대분류']}")
        if not pd.isna(row.get('공사종류_중분류')):
            cons_parts.append(f"중분류: {row['공사종류_중분류']}")
        if not pd.isna(row.get('공사종류_소분류')):
            cons_parts.append(f"소분류: {row['공사종류_소분류']}")
        if cons_parts:
            components.append("공사종류: " + ", ".join(cons_parts))
        
        # 공종 관련 (대분류, 소분류)
        work_parts = []
        if not pd.isna(row.get('공종_대분류')):
            work_parts.append(f"대분류: {row['공종_대분류']}")
        if not pd.isna(row.get('공종_소분류')):
            work_parts.append(f"중분류: {row['공종_소분류']}")
        if work_parts:
            components.append("공종: " + ", ".join(work_parts))
        
        # 작업 프로세스 및 사고 원인
        if not pd.isna(row.get('작업프로세스')):
            components.append(f"작업 프로세스: {row['작업프로세스']}")
        if not pd.isna(row.get('사고원인')):
            components.append(f"사고 원인: {row['사고원인']}")
        
        # 사고 인지 시차
        if not pd.isna(row.get('사고인지시차')):
            components.append(f"사고 인지 시차: {row['사고인지시차']}")
        
        # 인적 피해 및 근무 종류
        if not pd.isna(row.get('인적사고')):
            components.append(f"인적 피해: {row['인적사고']}")
        if not pd.isna(row.get('근무종류')):
            components.append(f"근무 종류: {row['근무종류']}")
        
        # 물적 피해: 사고객체 대분류, 소분류, 피해 내용
        material_parts = []
        if not pd.isna(row.get('사고객체_대분류')):
            material_parts.append(f"{row['사고객체_대분류']}인")
        if not pd.isna(row.get('사고객체_소분류')):
            material_parts.append(f"{row['사고객체_소분류']}가")
        if not pd.isna(row.get('물적사고')):
            material_parts.append(f"{row['물적사고']}")
        if material_parts:
            components.append("물적 피해: " + " ".join(material_parts))
        
        # 사고 부위
        if not pd.isna(row.get('부위2')):
            components.append(f"사고 부위: {row['부위2']}")
        
        # 사고객체 (대분류와 소분류)
        if not pd.isna(row.get('사고객체_대분류')):
            obj_str = f"사고객체: {row['사고객체_대분류']}"
            if not pd.isna(row.get('사고객체_소분류')):
                obj_str += f" (소분류: {row['사고객체_소분류']})"
            components.append(obj_str)
        
        
        components.append("이에 대한 '대책'은 무엇인가요?")
        # 모든 항목을 쉼표로 구분하여 결합
        return ", ".join(components)
    

    def exp_2(row):
        components = []
        components.append("다음과 같은 상황에서 공사 중 사고가 발생했습니다.")
        
        # 시간 관련 항목
        if not pd.isna(row.get('계절')):
            components.append(f"계절: {row['계절']}")
        if not pd.isna(row.get('시간대')):
            components.append(f"시간대: {row['시간대']}")
        if not pd.isna(row.get('날씨')):
            components.append(f"날씨: {row['날씨']}")
        if not pd.isna(row.get('기온')):
            components.append(f"기온: {row['기온']}도")
        if not pd.isna(row.get('습도')):
            components.append(f"습도: {row['습도']}%")
        if not pd.isna(row.get('장소')):
            components.append(f"장소: {row['장소']}")
        
        # 구조물 층수: 지상, 지하
        level_parts = []
        if not pd.isna(row.get('지상')):
            level_parts.append(f"지상 {row['지상']}층")
        if not pd.isna(row.get('지하')):
            level_parts.append(f"지하 {row['지하']}층")
        if level_parts:
            components.append("구조물: " + ", ".join(level_parts) + "에서 사고 발생")
        
        # 공사종류 관련 (대분류, 중분류, 소분류)
        cons_parts = []
        if not pd.isna(row.get('공사종류_대분류')):
            cons_parts.append(f"대분류: {row['공사종류_대분류']}")
        if not pd.isna(row.get('공사종류_중분류')):
            cons_parts.append(f"중분류: {row['공사종류_중분류']}")
        if not pd.isna(row.get('공사종류_소분류')):
            cons_parts.append(f"소분류: {row['공사종류_소분류']}")
        if cons_parts:
            components.append("공사종류: " + ", ".join(cons_parts))
        
        # 공종 관련 (대분류, 소분류)
        work_parts = []
        if not pd.isna(row.get('공종_대분류')):
            work_parts.append(f"대분류: {row['공종_대분류']}")
        if not pd.isna(row.get('공종_소분류')):
            work_parts.append(f"중분류: {row['공종_소분류']}")
        if work_parts:
            components.append("공종: " + ", ".join(work_parts))
        
        # 작업 프로세스 및 사고 원인
        if not pd.isna(row.get('작업프로세스')):
            components.append(f"작업 프로세스: {row['작업프로세스']}")
        if not pd.isna(row.get('사고원인')):
            components.append(f"사고 원인: {row['사고원인']}")
        
        # 사고 인지 시차
        if not pd.isna(row.get('사고인지시차')):
            components.append(f"사고 인지 시차: {row['사고인지시차']}")
        
        # 인적 피해 및 근무 종류
        if not pd.isna(row.get('인적사고')):
            components.append(f"인적 피해: {row['인적사고']}")
        if not pd.isna(row.get('근무종류')):
            components.append(f"근무 종류: {row['근무종류']}")
        
        # 물적 피해: 사고객체 대분류, 소분류, 피해 내용
        material_parts = []
        if not pd.isna(row.get('사고객체_대분류')):
            material_parts.append(f"{row['사고객체_대분류']}인")
        if not pd.isna(row.get('사고객체_소분류')):
            material_parts.append(f"{row['사고객체_소분류']}가")
        if not pd.isna(row.get('물적사고')):
            material_parts.append(f"{row['물적사고']}")
        if material_parts:
            components.append("물적 피해: " + " ".join(material_parts))
        
        # 사고 부위
        if not pd.isna(row.get('부위2')):
            components.append(f"사고 부위: {row['부위2']}")
        
        # 사고객체 (대분류와 소분류)
        if not pd.isna(row.get('사고객체_대분류')):
            obj_str = f"사고객체: {row['사고객체_대분류']}"
            if not pd.isna(row.get('사고객체_소분류')):
                obj_str += f" (소분류: {row['사고객체_소분류']})"
            components.append(obj_str)
        
        
        components.append("이에 대한 '대책'은 무엇인가요?")
        # 모든 항목을 쉼표로 구분하여 결합

        question = ", ".join(components)
        question = f"""
[질문]
{question}
[답변]
"""
        return question



def get_prompt_question(data, exp:str = "exp_0", kwargs:dict = {}):
    return getattr(Question, exp)(data, **kwargs)