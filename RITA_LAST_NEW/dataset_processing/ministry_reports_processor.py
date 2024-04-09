import numpy as np
import pandas as pd
from tqdm import tqdm
pd.options.mode.chained_assignment = None

MAIN_PATH = '/Users/leonidharlov/Documents/Industrial Potential/data/FINAL/'
OUTPUT_DATA_PATH = f'{MAIN_PATH}output_data/'
INPUT_DATA_PATH = f'{MAIN_PATH}input_data/'

# returns group code from a code of the programm
def extract_code_group(x):
    res = x.split('.')
    return res[0]

# returns education level code (BA/MA/SPEC/etc.) from a code of the programm
def extract_code_level(x):
    res = x.split('.')
    if len(res) > 1:
        return res[1]
    return np.nan

# returns major code from a code of the programm
def extract_code_major(x):
    res = x.split('.')
    if len(res) > 2:
        return res[2]
    return np.nan

# harmonizes region names with the ones seen in MICC data (table1)
def clean_region_names(x):
    return x.replace(
        'Республика ','').replace(
        'й','й').replace(
        'Саха (Якутия)','Якутия').replace(
        'Чеченская Республика','Чечня').replace(
        'Чувашская Республика','Чувашия').replace(
        'Кабардино-Балкарская Республика','Кабардино-Балкария').replace(
        'Карачаево-Черкесская Республика','Карачаево-Черкесия').replace(
        'Удмуртская Республика','Удмуртия').replace(
        'Северная Осетия - Алания','Северная Осетия').replace(
        'автономный округ','АО').replace(
        'автономная область','АО').lstrip('г.')

# cleans city names
def fix_city_names(x):
    return x.replace(
        'п. свх. ','').replace(
        'м. ', '').replace(
        'г. ','').replace(
        'п.','').replace(
        'г.т.','').replace(
        'ст. ','').replace(
        'с. ', '').replace(
        'т. ','').replace(
        'д. ','').replace(
        'х. ','').replace(
        'р. ','').strip().replace(
        'город ','').replace(
        'село ','').strip().lstrip('с')

# returns estimated programme duration
# estimated programme duration is the latest study year with share of students 
# exceeding set treshold (defaults to 1%)
def get_programme_duration(x,treshold = 0.01):
    for i in range(6,0,-1):
        if x[i-1] > treshold:
            return i
    return 0

# returns dataframe with duration estimates for all programmes in df
def get_programme_duration_df(df):
    res = df.copy()

    study_years_count = 6
    if 'year_7' in res.columns:
        study_years_count = 7
    column_names = []

    for i in np.arange(1,study_years_count+1):
        res[f'year_{i}_share'] = res.apply(lambda x: x[f'year_{i}']/x['students_total'],axis=1)
        column_names.append(f'year_{i}_share')
    
    res = res.groupby(['code','region'])[column_names].mean().reset_index()

    res['programme_duration'] = res.loc[:,'year_1_share':f'year_{study_years_count}_share'].apply(
        lambda x: get_programme_duration(x),axis=1)

    res = res[['code','region','programme_duration']]
    
    return res

# returns retention rate for a given major in a given year
# x is a series with major's code, report year, region,
# programme_duration, and education type
# df is a dataframe with the parsed ministry reports
def get_retention(x, df):
    max_year = df['year'].max()
    code = x['code']
    year = x['year']
    region = x['region']
    edu_type = x['type']
    programme_duration = x['programme_duration']
    
    filtered_df = df.query(
        f'code == "{code}" and region == "{region}" and type == "{edu_type}"')
    
    if year+programme_duration-1 > max_year:
        return -100 # not enough data to compute retention – too recent
    
    total_students = x['students_total']
    if total_students == 0:
        return -200 # no students this year
    
    graduate_counts = filtered_df[filtered_df['year'].between(year, year + programme_duration - 1)]['graduates']
    if len(graduate_counts)<programme_duration:
        return -300  # major vanished from the reports
    
    graduate_count = graduate_counts.sum()
    
    if graduate_count == 0:

        if programme_duration == 1:
            return 1  # programme_duration is 1 and graduate count is 0
        
        if programme_duration < 1:
            return -500  # programme duration is zero
        
        last_year = filtered_df.query(f'year == {year + programme_duration - 1}')[f'year_{programme_duration - 1}'].iloc[0]
        first_year = filtered_df.query(f'year == {year + programme_duration - 1}')['year_1'].iloc[0]
        
        if first_year == 0:
            return -400  # zero graduates and zero freshmen
        
        retention = last_year / first_year
        return retention
    
    retention = graduate_count / total_students

    return retention

# add information about education domain
def add_edu_domain_info(df):
    code_group_to_edu_domain = pd.read_csv(f'{INPUT_DATA_PATH}code_group_to_edu_domain.csv')
    df = df.merge(code_group_to_edu_domain,how='left')
    # fill missing education domains
    domain_replace_dict = df.dropna(subset=['edu_domain']).groupby('major')['edu_domain'].first().to_dict()
    df['edu_domain'] = df.apply(
        lambda x: domain_replace_dict[x.major] if pd.isna(x.edu_domain) and x.major in domain_replace_dict else x.edu_domain,axis=1)
    
    return df

# processes ministry data
def process_education_df(df):
    res = df.copy()
    print('calculating total students')
    if 'year_7' in res.columns:
        res['students_total'] = res.loc[:,'year_1':'year_7'].sum(axis=1)
    else:
        res['students_total'] = res.loc[:,'year_1':'year_6'].sum(axis=1)
    
    res = res.query('students_total > 0')

    print('extracting group, level, and major codes')
    res['code_group'] = pd.to_numeric(res['code'].map(lambda x: extract_code_group(x)))
    res['code_level'] = pd.to_numeric(res['code'].map(lambda x: extract_code_level(x)))
    res['code_major'] = pd.to_numeric(res['code'].map(lambda x: extract_code_major(x)))

    print('fixing region names')
    res['region'] = res['region'].map(lambda x: clean_region_names(x))

    print('assessing programme durations')
    res = res.merge(get_programme_duration_df(res),on=['code','region'],how='left')

    print('calculating retention')
    retention_list = []
    for index, row in tqdm(res.iterrows(), total=res.shape[0]):
        retention_list.append(get_retention(row,res))    
    res['retention'] = retention_list

    res = add_edu_domain_info(res)
    
    return res

# removes entries with missing retention
# removes entires with extremely high retention (top 5%)
# returns filtered dataframe
def filter_education_df(df,save_file=True):
    df = df.query('retention > 0')
    retention_higher_treshold = df['retention'].quantile(0.95)
    df = df.query('retention < @retention_higher_treshold')
    
    if save_file:
        df.to_csv(f'{OUTPUT_DATA_PATH}table0_filtered.csv')

    return df