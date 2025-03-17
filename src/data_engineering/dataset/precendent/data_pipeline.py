from .preprocess.cleaning import *
from .preprocess.feature_engineering import *



class Pipelines:
    def pipeline_0(df):
        df = preprocess_baseline(df)
        return df
    
    def pipeline_1(df):
        df = preprocess_dt(df, "발생일시")
        df = preprocess_tr(df)
        df = preprocess_dt(df, "사고인지 시간")
        
        df = preprocess_temp_1(df)
        df = preprocess_humid_1(df)
        df = preprocess_area_1(df)
        df = preprocess_cause_1(df)
        df = preprocess_floor_1(df)
        df = preprocess_hierarchy(df, "공사종류", ["공사종류_대분류", "공사종류_중분류","공사종류_소분류"], "/")
        df = preprocess_hierarchy(df, "공종", ["공종_대분류", "공종_소분류"], ">")
        df = preprocess_hierarchy(df, "사고객체", ["사고객체_대분류", "사고객체_소분류"], ">")
        df = preprocess_hierarchy(df, "부위", ["부위1", "부위2"], "/")
        
        df = preprocess_id(df)
        return df
    
    def pipeline_2(df):
        df = preprocess_dt(df, "발생일시")
        df = preprocess_tr(df)
        df = preprocess_dt(df, "사고인지 시간")
        
        df = preprocess_temp_1(df)
        df = preprocess_humid_1(df)
        df = preprocess_area_1(df)
        df = preprocess_cause_1(df)
        df = preprocess_floor_1(df)
        df = preprocess_hierarchy(df, "공사종류", ["공사종류_대분류", "공사종류_중분류","공사종류_소분류"], "/")
        df = preprocess_hierarchy(df, "공종", ["공종_대분류", "공종_소분류"], ">")
        df = preprocess_hierarchy(df, "사고객체", ["사고객체_대분류", "사고객체_소분류"], ">")
        df = preprocess_hierarchy(df, "부위", ["부위1", "부위2"], "/")
        df = preprocess_part_1(df)
        df = preprocess_season(df)
        df = preprocess_daytime(df)
        df = preprocess_recogdelay(df)
        df = preprocess_temp_2(df)
        df = preprocess_humid_2(df)
        
        df = preprocess_wt_1(df)
        df = preprocess_ct2_1(df)
        df = preprocess_ct3_1(df)
        df = preprocess_areafloor(df)
        df = preprocess_hd_1(df)
        df = preprocess_md_1(df)
        df = preprocess_mt1_1(df)
        df = preprocess_mt2_1(df)
        df = preprocess_ao1_1(df)
        df = preprocess_ao2_1(df)
        df = preprocess_wp_1(df)
        df = preprocess_place_1(df)

        df = preprocess_id(df)
        return df
    
    
def install_pipeline(pipeline:str = "pipeline_0"):
    return getattr(Pipelines, pipeline)