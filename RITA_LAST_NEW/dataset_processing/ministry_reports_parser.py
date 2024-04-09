import numpy as np
import pandas as pd
from functools import reduce
import os
from tqdm import tqdm
pd.options.mode.chained_assignment = None

MAIN_PATH = ''
OUTPUT_DATA_PATH = f'{MAIN_PATH}output_data/'
INPUT_DATA_PATH = f'{MAIN_PATH}input_data/'

# returns a list of files in a given folder
def list_files(folder, show_hidden=False):
    files_list = os.listdir(folder)
    if show_hidden:
        return files_list
    return [x for x in files_list if not x.startswith('.')]

# returns a dictionary, keys are the regions available in the reports, values are the table filenames
def get_region_filenames(year, edu_type, fulltime_only=True):
    folder_name = 'middle' if edu_type == 'СПО' else 'higher'
    reports_path = f'{INPUT_DATA_PATH}ministry_reports/{folder_name}/Своды {edu_type}-1 {year}/'
    path_state = reports_path+'Государственные/'
    path_private = reports_path+'Негосударственные/'

    filenames_state = sorted(list_files(path_state))
    filenames_private = sorted(list_files(path_private))
    
    if fulltime_only:
        filenames_state = [x for x in filenames_state if x.split('_')[-1].split('.')[0].lower() == 'очная']
        filenames_private = [x for x in filenames_private if x.split('_')[-1].split('.')[0].lower() == 'очная']
    
    region_filenames_state = pd.DataFrame([(x.split('_')[0],"Государственные/"+x) for x in filenames_state]).groupby(0)[1].apply(list)
    region_filenames_private = pd.DataFrame([(x.split('_')[0],"Негосударственные/"+x) for x in filenames_private]).groupby(0)[1].apply(list)
    region_filenames = dict(pd.concat([region_filenames_state,region_filenames_private]).reset_index().groupby(0)[1].apply(sum))
    
    return region_filenames

# returns index where a table begins
def get_start_index(df):
    for index,rows in df.iterrows():
        if np.array_equal(rows.values[:3], [1,2,3]):
            return index+1

# returns a table with the data on the number of applicants and the number of admitted students 
# for a given region in a given year
def get_admitted_table(region,year,edu_type,save_file=True):
    folder_name = 'middle' if edu_type == 'СПО' else 'higher'
    reports_path = f'{INPUT_DATA_PATH}ministry_reports/{folder_name}/Своды {edu_type}-1 {year}/'
    region_filenames = get_region_filenames(year,edu_type)
    
    df_list = []
    
    for study_mode in region_filenames[region]:
        path_file = reports_path+study_mode

        df = pd.read_excel(path_file,sheet_name='Р2_1_1',skiprows=6)
        df.columns = np.arange(df.shape[1])+1
        df = df[get_start_index(df):]

        if year > 2015:
            if edu_type == 'СПО':
                res = df[[1,3,4,5,6]].replace(0,np.nan).dropna(subset=[3,6])
                res = res.rename(columns={
                    1:'major',
                    3:'code',
                    4:'applied_budget',
                    5:'applied_paid',
                    6:'entered'})

            elif edu_type == 'ВПО':
                res = df[[1,3,4,7,8]].replace(0,np.nan).dropna(subset=[3,8])
                res = res.rename(columns={
                    1:'major',
                    3:'code',
                    4:'applied_budget',
                    7:'applied_paid',
                    8:'entered'})
            
            res['applied_budget'] = pd.to_numeric(res['applied_budget'])
            res['applied_paid'] = pd.to_numeric(res['applied_paid'])
            res['entered'] = pd.to_numeric(res['entered'])
            res['applied'] = res[['applied_budget','applied_paid']].sum(axis=1)
            res = res.drop(['applied_budget','applied_paid'],axis=1)

        else:
            res = df[[1,3,4,5]].replace(0,np.nan).dropna(subset=[3])
            res = res.rename(columns={
                1:'major',
                3:'code',
                4:'applied',
                5:'entered'})
            res['applied'] = pd.to_numeric(res['applied'])
            res['entered'] = pd.to_numeric(res['entered'])

        if len(res) == 0: 
            continue

        df_list.append(res)
    
    if len(df_list) > 0:
        admitted_table = pd.concat(df_list)
        admitted_table = admitted_table.groupby(
            ['major','code']).agg({'applied':'sum','entered':'sum'}
                                  ).sort_values('code').reset_index()
        if save_file:
            admitted_table.to_csv(f'{OUTPUT_DATA_PATH}admitted_table_{region}_{edu_type}_{year}.csv')

        return admitted_table
    
    return pd.DataFrame()

# returns a table with the data on the number of graduates 
# for a given region in a given year
def get_graduates_table(region,year,edu_type,save_file=True):
    folder_name = 'middle' if edu_type == 'СПО' else 'higher'
    reports_path = f'{INPUT_DATA_PATH}ministry_reports/{folder_name}/Своды {edu_type}-1 {year}/'
    region_filenames = get_region_filenames(year,edu_type)
    
    df_list = []
    
    for study_mode in region_filenames[region]:
        path_file = reports_path+study_mode
        
        if year > 2015:
            sheet_name = 'Р2_1_3(1)'
        else:
            sheet_name = 'Р2_1_4_П'

        df = pd.read_excel(path_file,sheet_name=sheet_name,skiprows=6)
        df.columns = np.arange(df.shape[1])+1
        df = df[get_start_index(df):]

        if edu_type == 'СПО':
            if year > 2015:
                res = df[[1,3,4]]
            else:
                res = df[[1,3,6]]
        elif edu_type == 'ВПО':
            if year > 2015:
                res = df[[1,4,5]]
            else:
                res = df[[1,4,6]]

        res.columns = ['major','code','graduates']
        res['graduates'] = res['graduates'].replace('человек',np.nan)
        res['code'] = res['code'].replace(0,np.nan)
        res = res.dropna(subset=['code','graduates'])
        res['code'] = res['code'].astype(str)
        res['graduates'] = pd.to_numeric(res['graduates'])

        df_list.append(res)
        
    if len(df_list) > 0:
        graduates_table = pd.concat(df_list)
        graduates_table = graduates_table.groupby(
            ['major','code']).agg({'graduates':'sum'}).sort_values('code').reset_index()
        
        if save_file:
            graduates_table.to_csv(f'{OUTPUT_DATA_PATH}graduates_table_{region}_{edu_type}_{year}.csv')

        return graduates_table
    
    return pd.DataFrame()

# returns a table with the data on the number of students in each study year 
# for a given region in a given year
def get_year_table(region,year,edu_type,save_file=True):
    folder_name = 'middle' if edu_type == 'СПО' else 'higher'
    reports_path = f'{INPUT_DATA_PATH}ministry_reports/{folder_name}/Своды {edu_type}-1 {year}/'
    region_filenames = get_region_filenames(year,edu_type)

    df_list = []

    for study_mode in region_filenames[region]:
        path_file = reports_path+study_mode

        if year > 2015:
            sheets = ['Р2_1_2(1)','Р2_1_2 (2)','Р2_1_2 (3)','Р2_1_2 (4)']
            years = np.array([1,2])
            sheets_merged = []

            for sheet in sheets:
                if edu_type == 'СПО' and sheet == 'Р2_1_2 (4)':
                    continue
                df = pd.read_excel(path_file,sheet_name=sheet,skiprows=6)
                df.columns = np.arange(df.shape[1])+1
                df = df[get_start_index(df):]

                if edu_type == 'СПО':
                    res = df[[1,3,4,11]]
                    res[3] = res[3].replace(0,np.nan)
                    res = res.dropna(subset=[3])
                    res.columns = ['major','code',f'year_{years[0]}',f'year_{years[1]}']

                    if years[0] == 1:
                        res['source'] = study_mode.split('_')[-1]
                        sheets_merged.append(res)
                    else:
                        sheets_merged.append(res.drop(['major', 'code'], axis=1))

                    years += 2
                elif edu_type == 'ВПО':
                    if year == 2021 and sheet == 'Р2_1_2 (4)':
                        continue
                    if sheet == 'Р2_1_2 (3)' and year == 2021:
                        res = df[[1,4,5,12,19]]
                    else:
                        res = df[[1,4,5,12]]
                        
                    res[4] = res[4].replace(0,np.nan)
                    res = res.dropna(subset=[4])
                    res[4] = res[4].astype(str)
                    
                    if sheet == 'Р2_1_2 (3)' and year == 2021:
                        res.columns = ['major','code',f'year_{years[0]}',f'year_{years[1]}',f'year_{years[1]+1}']
                    else:
                        res.columns = ['major','code',f'year_{years[0]}',f'year_{years[1]}']

                    if years[0] == 1:
                        res['source'] = study_mode.split('_')[-1]
                        sheets_merged.append(res)
                    else:
                        sheets_merged.append(res.drop(['major', 'code'], axis=1))
                        
                    years += 2

            if len(sheets_merged) > 0:
                df_list.append(pd.concat(sheets_merged, axis=1))
        else:
            df = pd.read_excel(path_file,sheet_name='Р2_1_4',skiprows=6)

            df.columns = np.arange(df.shape[1])+1
            df = df[get_start_index(df):]

            if edu_type == 'СПО':
                res = df[[1,3,4,6,8,10,12,14]]
                res.columns = ['major','code','year_1','year_2',
                               'year_3','year_4','year_5','year_6']
                res['code'] = res['code'].replace(0,np.nan)
                res = res.dropna(subset=['code'])

            elif edu_type == 'ВПО':
                res = df[[1,4,5,7,9,11,13,15,17]]
                res.columns = ['major','code','year_1','year_2',
                               'year_3','year_4','year_5','year_6','year_7']
                res['code'] = res['code'].replace(0,np.nan)
                res = res.dropna(subset=['code'])
                res['code'] = res['code'].astype(str)

            df_list.append(res)

    if len(df_list) > 0:
        if edu_type == 'ВПО':
            year_list = ['year_1','year_2','year_3','year_4','year_5','year_6','year_7']
        else:
            year_list = ['year_1','year_2','year_3','year_4','year_5','year_6']
            
        res = pd.concat(df_list).groupby(
            ['major','code'])[year_list].sum().sort_values('code').reset_index()
        
        if save_file:
            res.to_csv(f'{OUTPUT_DATA_PATH}year_table_{region}_{edu_type}_{year}.csv')

        return res

    return pd.DataFrame()

# combines tables with the data on admitted students, graduates, and yearly breakdown
# for a given region in a given year
def get_region_table(region,year,edu_type,save_file=True):
    admitted_table = get_admitted_table(region,year,edu_type,save_file=False)
    graduates_table = get_graduates_table(region,year,edu_type,save_file=False)
    year_table = get_year_table(region,year,edu_type,save_file=False)
    
    non_empty_dfs = [x for x in [admitted_table,graduates_table,year_table] if len(x) > 0]

    res = reduce(
        lambda left,right: pd.merge(left,right,on=['major','code'],how='outer'), non_empty_dfs)
    
    if save_file:
        res.to_csv(f'{OUTPUT_DATA_PATH}region_table_{region}_{edu_type}_{year}.csv')

    return res

# combines all region tables with data between year_min and year_max
def get_all_regions_table(edu_type='', year_min = 2015, year_max = 2021,save_file=True):
    years = np.arange(year_min,year_max+1)
    if edu_type != '':
        regions = list(get_region_filenames(year_max,edu_type).keys())
    else:
        regions_middle = list(get_region_filenames(year_max,'СПО').keys())
        regions_higher = list(get_region_filenames(year_max,'ВПО').keys())
        regions = list(set(regions_middle+regions_higher))

    df_list = []

    for region in tqdm(regions,leave=True):
        for year in tqdm(years,leave=False):
            if edu_type != '':
                res = get_region_table(region,year,edu_type,save_file=False)
                res['type'] = edu_type
            else: 
                res_middle = pd.DataFrame()
                if region in regions_middle:
                    res_middle = get_region_table(region,year,'СПО',save_file=False)
                    res_middle['type'] = 'СПО'
                    
                res_higher = pd.DataFrame()
                if region in regions_higher:
                    res_higher = get_region_table(region,year,'ВПО',save_file=False)
                    res_higher['type'] = 'ВПО'
                    
                res = pd.concat([res_middle,res_higher])

            res['region'] = region
            res['year'] = year

            df_list.append(res)

    final_df = pd.concat(df_list)

    if edu_type == '' or edu_type == 'ВПО':
        final_df = pd.concat([final_df.drop(['type','region','year'],axis=1),final_df[['type','region','year']]],axis=1)

    if save_file:
        filename = f'table0_raw_{edu_type}.csv' if edu_type != '' else 'table0_raw.csv'
        final_df.to_csv(f'{OUTPUT_DATA_PATH}{filename}')

    return final_df