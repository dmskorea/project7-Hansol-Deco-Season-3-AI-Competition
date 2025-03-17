from typing import List, Iterable, Union
import itertools

import pandas as pd
import numpy as np

from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, kruskal, f_oneway, kstest
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import chi2_contingency



def is_associated_categorical_numeric(groups: List[Iterable[Union[int, float]]]) -> bool:
    '''
    뉴메릭 데이터와 카테고리컬 데이터의 연관성 여부 체크
    세부 요구사항 명시해야 함
    '''
    parametric = True
    equal_var = False   
    # groups 결과에 따른 대처
    if len(groups)==1:
        return None
    lengroups = list(map(len, groups))
    if 0 in lengroups:
        pass
    if any(map(lambda x: x<100, lengroups)):
        parametric = False
    else:
        # 정규성검정
        normality = [shapiro(group)[1] if len(group) < 5000 else kstest(group, 'norm')[1] for group in groups] # 최대표본크기 적용
        normality = [0 if p_value < 0.05 else 1 for p_value in normality] # 유의수준준
        if sum(normality) != len(normality):
            parametric = False
    if parametric:
        stat, p_value = levene(*groups)
        if p_value < 0.05:
            equal_var = False
        else:
            equal_var = True

    # 그룹 수, 정규성에 따라 검정 방법 달라짐(t-test, mann, anova, kruskal)
    if len(groups)==2:
        if parametric:
            stat, p_value_ass = ttest_ind(*groups, equal_var = equal_var)
        else:
            stat, p_value_ass = mannwhitneyu(*groups, alternative='two-sided')
    else:
        if parametric:
            if equal_var:
                stat, p_value_ass = f_oneway(*groups)
            else:
                pass
        else:
            stat, p_value_ass = kruskal(*groups)
            
    # 그룹 간 차이가 유의하게 날 경우 연관성 있는 것으로 판단단
    if p_value_ass < 0.05:
        return True
    else:
        return False




# Shannon 엔트로피 계산
def entropy(data, bins):
    hist, _ = np.histogram(data, bins=bins)
    probs = hist / np.sum(hist)
    return -np.sum(probs * np.log(probs + 1e-9))  # log(0) 방지

# 공동 엔트로피 계산
def joint_entropy(x, y, bins):
    hist2d, _, _ = np.histogram2d(x, y, bins=bins)
    probs = hist2d / np.sum(hist2d)
    return -np.sum(probs * np.log(probs + 1e-9))

# Mutual Information 계산
def mutual_information(x, y, bins):
    h_x = entropy(x, bins)
    h_y = entropy(y, bins)
    h_xy = joint_entropy(x, y, bins)
    return h_x + h_y - h_xy

# MIC 계산
def calculate_mic(x, y, max_bins=20):
    max_mi = 0
    for bins in range(2, max_bins + 1):
        mi = mutual_information(x, y, bins)
        max_mi = max(max_mi, mi)
    # 정규화 (최대 엔트로피로 나눔)
    norm_mic = max_mi / np.log(min(len(x), len(y)))
    return norm_mic

def is_associated_numeric(s1:Iterable[Union[int, float]], s2:Iterable[Union[int, float]]) -> bool:
    '''
    두 뉴메릭 데이터의 연관성 여부 체크
    세부 요구사항 명시해야 함
    추후 전략을 바꿀 수 있음
    '''
    # Pearson
    pearson_corr, pearson_p = pearsonr(s1, s2)
    
    # Spearman
    spearman_corr, spearman_p = spearmanr(s1, s2)
    
    # Kendall
    kendall_corr, kendall_p = kendalltau(s1, s2)
    
    # MIC
    s = filter(lambda x: ~np.isnan(x[0] + x[1]), zip(s1, s2))
    s1_, s2_ = zip(*s)
    mic_corr = calculate_mic(s1_, s2_)

    result = {
        "pearson":(pearson_corr, pearson_p),
        "spearman":(spearman_corr, spearman_p),
        "kendall":(kendall_corr, kendall_p),
        "mic":mic_corr,
    }

    # 해석
    analysis_pearson = True if abs(result['pearson'][0]) > 0.3 and result['pearson_p'][1] < 0.05 else False
    analysis_spearman = True if abs(result['spearman'][0]) > 0.3 and result['spearman_p'][1] < 0.05 else False
    analysis_kendall = True if abs(result['kendall'][0]) > 0.2 and result['kendall_p'][1] < 0.05 else False
    analysis_mic = True if abs(result['mic']) > 0.3 else False
    analysis = any([analysis_pearson, analysis_spearman, analysis_kendall, analysis_mic])
    
    return analysis



def is_associated_categorical(s1:Iterable[Union[int, float]], s2:Iterable[Union[int, float]]) -> bool:
    '''
    두 카테고리컬 데이터의 연관성 여부 체크
    세부 요구사항 명시해야 함
    '''
    ct = pd.crosstab(s1, s2)
    chi2, p, dof, expected = chi2_contingency(ct)
    n = ct.sum().sum()  # 전체 샘플 수
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

    if p< 0.05 and cramers_v>0.4: # 중간 강도 이상의 연관성만 체크
        # print(f"{col1}, {col2} 연관성 존재")
        # print(f"Cramér's V: {cramers_v}")
        return True
    else:
        # print(f"{col1}, {col2} 연관성 없음")
        return False

def is_associated(df:pd.DataFrame, meta:dict, col1:str, col2:str):
    '''
    연관성 체크하는 방법은 조건이나 목적에 따라 여러가지가 있는데, 메타데이터 보고 이걸 판단해주고 실행하기 위한 함수
    데이터 자체에 대한 관리는 일단 판다스 dataframe에 맡긴다...
    메타데이터는 dict이고 json으로 저장되며 정해진 규칙을 따름. EDA1에서 추가로 모수, 비모수 여부도 넣어야 할 듯.
    일단 한번에 하는 함수로 남기고 필요시 나중에 EDA자동화로 결과 데이터 디파인된 방식으로 저장 후 해석따로하도록 고도화
    지금 이 상태의 문제는, 분석 별 요구사항이 달라 결측처리 등 대처도 다른데 함수하나 안에 묶어놓았다는 점
    '''
    type1_col1 = meta[col1]["type"]
    type1_col2 = meta[col2]["type"]

    case_dict = {
        "ordinal" : ['numeric', 'categorical'],
        "nominal" : ['categorical'],
        "ratio" : ["numeric"],
        "interval" : ["numeric"]
    }
    result = False
    for type2_col1, type2_col2 in itertools.product(case_dict[type1_col1], case_dict[type1_col2]):
        if type2_col1 == "numeric" and type2_col2 == "categorical":
            # 여기에 전처리 필요
            groups = [df[df[col2] == cat][col1] for cat in df[col2].dropna().unique()]
            result = is_associated_categorical_numeric(groups)
        if result:
            break

        if type2_col1 == "categorical" and type2_col2 == "numeric":
            # 여기에 전처리 필요
            groups = [df[df[col1] == cat][col2] for cat in df[col1].dropna().unique()]
            result = is_associated_categorical_numeric(groups)
        if result:
            break

        if type2_col1 == "numeric" and type2_col2 == "numeric":
            # 여기에 전처리 필요
            col_numeric1 = df[col1]
            col_numeric2 = df[col2]
            result = is_associated_numeric(col_numeric1, col_numeric2)
        if result:
            break
            
        if type2_col1 == "categorical" and type2_col2 == "categorical":
            # 여기에 전처리 필요
            col_categorical1 = df[col1]
            col_categorical2 = df[col2]
            result = is_associated_categorical(col_categorical1, col_categorical2)
        if result:
            break
        
    return result