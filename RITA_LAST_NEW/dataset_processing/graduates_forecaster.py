import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

MAIN_PATH = '/Users/leonidharlov/Documents/Industrial Potential/data/FINAL/'
OUTPUT_DATA_PATH = f'{MAIN_PATH}output_data/'

def forecast_graduates(students_total,retention):
    return np.round(students_total * retention)

def get_forecast_df(table0,table1,save_file=False):
    retention_df = table0.groupby(['region','code_group','type']).agg(
        {'retention':'mean','programme_duration':'first'}).reset_index()

    table1 = table1.merge(retention_df,on=[
        'region','code_group','type'],how='left')
    # fill missing retention values with median retention of the code_group
    median_retention_per_group = table1.groupby('code_group')[
        'retention'].median().to_dict()
    table1['retention'] = table1.apply(
        lambda x: median_retention_per_group[x['code_group']] if pd.isna(x['retention']) and x['code_group'] in median_retention_per_group else x['retention'],axis=1)

    median_duration_per_group = table1.groupby(
        'code_group')['programme_duration'].median().dropna().to_dict()
    table1['programme_duration'] = table1.apply(
        lambda x: median_duration_per_group[x['code_group']] if pd.isna(x['programme_duration']) and x['code_group'] in median_retention_per_group else x['programme_duration'],
        axis=1).astype(int)

    table1['graduates_forecast'] = table1.apply(
        lambda x: forecast_graduates(x['students_total'],x['retention']),axis=1).astype(int)

    table1 = table1.dropna(subset=['graduates_forecast'])

    table1['graduates_per_year_forecast'] = (
        table1['graduates_forecast']/table1[
            'programme_duration']).round().astype(int)
    
    if save_file:
        table1.to_csv(f'{OUTPUT_DATA_PATH}table1.csv')

    return table1