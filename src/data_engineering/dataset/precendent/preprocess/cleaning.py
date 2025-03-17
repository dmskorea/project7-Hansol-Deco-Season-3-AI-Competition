import datetime

import numpy as np


def preprocess_baseline(df):
    df_ = df.copy()
    
    df_['공사종류(대분류)'] = df_['공사종류'].str.split(' / ').str[0]
    df_['공사종류(중분류)'] = df_['공사종류'].str.split(' / ').str[1]
    df_['공종(대분류)'] = df_['공종'].str.split(' > ').str[0]
    df_['공종(중분류)'] = df_['공종'].str.split(' > ').str[1]
    df_['사고객체(대분류)'] = df_['사고객체'].str.split(' > ').str[0]
    df_['사고객체(중분류)'] = df_['사고객체'].str.split(' > ').str[1]
    return df_

def preprocess_id(df):
    col = 'ID'
    df_ = df.copy()
    df_ = df_.set_index(col)
    return df_

def preprocess_datetime(s):
    d, m, t = s.split()
    dt = d+" "+t
    dt = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M")
    if m == "오후" and dt.hour != 12:
        dt = dt + datetime.timedelta(hours = 12)
    return dt

def preprocess_dt(df, col):
    df_ = df.copy()
    df_[col] = df_[col].map(preprocess_datetime, na_action = 'ignore')
    return df_

def preprocess_tr(df):
    col1 = "사고인지 시간"
    col2 = "근무종류"
    df_ = df.copy()
    wttr = df_[col1].str.split('-', n=1, expand=True)
    wttr.columns = ['wt','tr']
    wt = wttr['wt']
    tr = wttr['tr']
    df_[col1] = tr.str.strip().replace('',np.nan)
    wt= wt.str.strip().replace('',np.nan)
    df_.insert(df.columns.get_loc(col1) + 1, col2, wt)
    return df_

def preprocess_temp_1(df):
    col = "기온"
    df_ = df.copy()
    df_[col] = df_[col].str.replace("℃","")
    df_.replace({col:{"":np.nan}}, inplace = True)
    df_[col] = df_[col].astype(float)
    return df_

def preprocess_humid_1(df):
    col = "습도"
    df_ = df.copy()
    df_[col] = df_[col].str.replace("%","")
    df_.replace({col:{"":np.nan}}, inplace = True)
    df_[col] = df_[col].astype(float)
    return df_

def preprocess_area_1(df):
    col = "연면적"
    df_ = df.copy()
    df_[col] = df_[col].str.replace(",","")
    df_[col] = df_[col].str.replace("㎡","")
    df_.replace({col:{"":np.nan}}, inplace = True)
    df_.replace({col:{"-":np.nan}}, inplace = True)
    df_[col] = df_[col].astype(float)
    return df_

def preprocess_floor_1(df):
    col = "층 정보"
    col1 = "지상"
    col2 = "지하"
    df_ = df.copy()
    df_.replace({col:{"-":np.nan}}, inplace = True)
    df_ = preprocess_hierarchy(df_, col, [col1, col2], sep = ",")
    
    df_[col1] = df_[col1].str.replace(r'\D', '', regex = True)
    df_[col2] = df_[col2].str.replace(r'\D', '', regex = True)

    df_[col1] = df_[col1].astype(float)
    df_[col2] = df_[col2].astype(float)
    return df_

def preprocess_cause_1(df):
    col = "사고원인"
    df_ = df.copy()
    df_[col] = df_[col].map(lambda x: np.nan if len(x)==1 else x, na_action = "ignore")
    return df_
    

def preprocess_hierarchy(df, col:str, levels:list, sep:str):
    level = len(levels)
    df_ = df.copy()
    frac = df_[col].str.split(sep, n=level-1, expand=True)
    frac.columns = levels
    frac = frac.apply(lambda x: x.str.strip())
    
    for i, column in enumerate(frac.columns):
        df_.insert(df.columns.get_loc(col) + i + 1, column, frac[column])
    df_.drop(col, axis=1, inplace = True)
    return df_


def preprocess_season(df):
    col1 = "발생일시"
    col2 = "계절"
    df_ = df.copy()
    df_[col2] = np.nan
    df_[col2] = df_[col2].astype("string[pyarrow]")
    winter = df_[col1].dt.month.isin([1,2,11,12])
    summer = df_[col1].dt.month.isin([5,6,7,8])
    spring = df_[col1].dt.month.isin([3,4])
    fall = df_[col1].dt.month.isin([9,10])
    df_.loc[winter,col2] = '겨울'
    df_.loc[summer,col2] = '여름'
    df_.loc[spring,col2] = '봄'
    df_.loc[fall,col2] = '가을'
    df_.insert(df_.columns.get_loc(col1)+1, col2, df_.pop(col2))
    return df_
    
def preprocess_daytime(df):
    col1 = "발생일시"
    col2 = "시간대"
    df_ = df.copy()
    df_[col2] = np.nan
    df_[col2] = df_[col2].astype("string[pyarrow]")
    morning = (df_[col1].dt.hour>=6) & (df_[col1].dt.hour<13)
    afternoon = (df_[col1].dt.hour>=13) & (df_[col1].dt.hour<19)
    evening = (df_[col1].dt.hour>=19) & (df_[col1].dt.hour<23)
    night = (df_[col1].dt.hour==23) & (df_[col1].dt.hour<6)
    df_.loc[morning,col2] = '아침'
    df_.loc[afternoon,col2] = '낮'
    df_.loc[evening,col2] = '저녘'
    df_.loc[night,col2] = '새벽'
    df_.insert(df_.columns.get_loc(col1)+1, col2, df_.pop(col2))
    return df_


def preprocess_recogdelay(df):
    col1 = "발생일시"
    col2 = "사고인지 시간"
    col3 = "사고인지시차"
    df_ = df.copy()
    df_[col3] = (df_[col2]-df_[col1]).dt.seconds/3600
    df_.insert(df_.columns.get_loc(col2)+1, col3, df_.pop(col3))
    return df_
    

def preprocess_temp_2(df):
    col1 = "기온"
    col2 = "발생일시"
    df_ = df.copy()
    
    mask = (~df_[col2].dt.month.isin([5,6,7,8]) & (df_[col1] > 30)) | (df_[col1] > 40) | (df_[col1]<-33)
    
    df_.loc[mask, col1] = np.nan
    return df_

def preprocess_humid_2(df):
    col = "습도"
    df_ = df.copy()
    mask = df_[col] > 100
    
    df_.loc[mask, col] = np.nan
    return df_

def preprocess_part_1(df):
    col1 = "부위1"
    col2 = "부위2"
    df_ = df.copy()
    df_.replace({col1:{"":np.nan}}, inplace = True)
    df_.replace({col2:{"":np.nan}}, inplace = True)
    return df_

def preprocess_wt_1(df):
    col = '근무종류'
    df_ = df.copy()
    df_.replace({col:{"기타":np.nan}}, inplace = True)
    return df_

def preprocess_ct2_1(df):
    col = '공사종류_중분류'
    df_ = df.copy()
    df_.replace({col:{"기타":np.nan}}, inplace = True)
    return df_

def preprocess_ct3_1(df):
    col = '공사종류_중분류'
    df_ = df.copy()
    df_.replace({col:{"기타":np.nan}}, inplace = True)
    return df_

def preprocess_areafloor(df):
    col1 = '연면적'
    col2 = '지상'
    col3 = '지하'
    df_ = df.copy()
    mask = (df_[[col1, col2, col3]] == 0).all(axis=1)
    df_.loc[mask, [col1, col2, col3]] = np.nan
    return df_

def preprocess_hd_1(df):
    col = '인적사고'
    df_ = df.copy()
    df_.replace({col:{"기타":np.nan}}, inplace = True)
    df_.replace({col:{"분류불능":np.nan}}, inplace = True)
    return df_

def preprocess_md_1(df):
    col = '물적사고'
    df_ = df.copy()
    df_.replace({col:{"기타":np.nan}}, inplace = True)
    return df_

def preprocess_mt1_1(df):
    col = '공종_대분류'
    df_ = df.copy()
    df_.replace({col:{"기타":np.nan}}, inplace = True)
    return df_

def preprocess_mt2_1(df):
    col = '공종_소분류'
    df_ = df.copy()
    df_.replace({col:{"기타":np.nan}}, inplace = True)
    return df_

def preprocess_ao1_1(df):
    col = '사고객체_대분류'
    df_ = df.copy()
    df_.replace({col:{"기타":np.nan}}, inplace = True)
    return df_

def preprocess_ao2_1(df):
    col = '사고객체_소분류'
    df_ = df.copy()
    df_.replace({col:{"기타":np.nan}}, inplace = True)
    return df_

def preprocess_wp_1(df):
    col = '작업프로세스'
    df_ = df.copy()
    df_.replace({col:{"기타":np.nan}}, inplace = True)
    return df_

def preprocess_place_1(df):
    col = '장소'
    df_ = df.copy()
    df_[col] = df_[col].map(lambda x: x.replace('/','').replace('/','기타').strip())
    df_.replace({col:{"":np.nan}}, inplace = True)
    return df_

def preprocess_part_1(df):
    col = '부위1'
    df_ = df.copy()
    df_ = df_.drop(col, axis=1)
    return df_