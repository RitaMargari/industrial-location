import numpy as np
import pandas as pd
import geopandas as gpd
import ast
import os
pd.options.mode.chained_assignment = None

MAIN_PATH = ''
OUTPUT_DATA_PATH = f'{MAIN_PATH}output_data/'
INPUT_DATA_PATH = f'{MAIN_PATH}input_data/'

# one group might belong to a few professional domains
# so to better approximate the number of students per domain
# we compute coefficients for every group-domain pair
def get_domain_coeff_table(ontology):
    ontology = ontology.drop('speciality',axis=1).drop_duplicates()
    total_majors_per_domain = ontology.groupby(
        ['code_group','prof_domain','type'])['code'].count().reset_index().rename(
        columns={'code':'total_in_domain'})

    total_majors_per_group = ontology.groupby(
        ['code_group','type'])['code'].nunique().reset_index().rename(
        columns={'code':'total_in_group'})

    domain_coeff_table = total_majors_per_domain.merge(total_majors_per_group,how='left')
    domain_coeff_table['coeff'] = domain_coeff_table['total_in_domain'] / domain_coeff_table['total_in_group']
    domain_coeff_table = domain_coeff_table.drop(['total_in_domain','total_in_group'],axis=1)
    
    return domain_coeff_table

# converts coordinate column from str or list to POINT type
# returns geodataframe
def df_to_gdf(df,coordinate_column_name='coordinates'):
    df = df.dropna(subset=[coordinate_column_name])
    if type(df[coordinate_column_name].iloc[0]) == str:
        df[coordinate_column_name] = df[coordinate_column_name].map(
            lambda x: ast.literal_eval(x))

    df[coordinate_column_name] = df[coordinate_column_name].map(
        lambda x: f'POINT({x[0]} {x[1]})')
    df[coordinate_column_name] = gpd.GeoSeries.from_wkt(df[coordinate_column_name])
    gdf = gpd.GeoDataFrame(df, geometry=coordinate_column_name)
    return gdf

# returns pivot table – number of students per each prof_domain in each city
# edu_type is either "ВПО" or "СПО" or ""
def generate_table2(table1,ontology,cities_coords,edu_type='',save_csv=False,save_geojson=False):
    domain_coeff_table = get_domain_coeff_table(ontology)

    if edu_type != '':
        domain_coeff_table = domain_coeff_table.query('type == @edu_type')
    
    table2 = domain_coeff_table.merge(
        table1.groupby(['code_group','region','city'])['graduates_forecast'].sum().reset_index())

    table2['graduates_forecast_adjusted'] = table2['graduates_forecast'] * table2['coeff']

    table2 = table2.groupby(
        ['city','prof_domain'])['graduates_forecast_adjusted'].sum().reset_index().drop_duplicates().pivot(
        index='city',columns='prof_domain',values='graduates_forecast_adjusted').sort_values('city')

    table2 = table2.merge(cities_coords,on='city')

    filename = f'table2_{edu_type}' if edu_type != '' else 'table2'

    if save_csv:
        table2.to_csv(f'{OUTPUT_DATA_PATH}{filename}.csv')
    if save_geojson:
        df_to_gdf(table2).to_file(f'{OUTPUT_DATA_PATH}{filename}.geojson', driver='GeoJSON')
    
    return table2

# returns pivot table – number of students per each code_group in prof_domain in each city
# edu_type is either "ВПО" or "СПО" or ""
def generate_table3(table1,ontology,cities_coords,edu_type='',save_csv=False,save_geojson=False):
    if edu_type != '':
        ontology = ontology.query('type == @edu_type')
    ontology_domain = ontology.groupby(['prof_domain'])['code_group'].unique()

    table3_dict = {}
    for domain in ontology_domain.keys():
        allowed_codes = ontology_domain.loc[domain]
        if edu_type != '':
            table3 = table1.query(
                f'code_group in @allowed_codes and type == @edu_type').groupby(
                ['city','code_group'])['graduates_forecast'].sum().reset_index().pivot(
                index='city', columns='code_group', values='graduates_forecast')
        else:
            table3_middle = table1.query(
                f'code_group in @allowed_codes and type == "СПО"').groupby(
                ['city','code_group'])['graduates_forecast'].sum().reset_index().pivot(
                index='city', columns='code_group', values='graduates_forecast')
            
            table3_higher = table1.query(
                f'code_group in @allowed_codes and type == "ВПО"').groupby(
                ['city','code_group'])['graduates_forecast'].sum().reset_index().pivot(
                index='city', columns='code_group', values='graduates_forecast')
            
            table3 = pd.concat([table3_middle,table3_higher]).groupby('city').sum()

        table3 = table3.replace(0,np.nan)
        table3 = table3.merge(cities_coords,on='city')
        table3_dict[domain] = table3

        filename = domain
        folder_name = f'table3_{edu_type}' if edu_type != '' else 'table3'

        if not os.path.exists(f'{OUTPUT_DATA_PATH}{folder_name}'):
            os.makedirs(f'{OUTPUT_DATA_PATH}{folder_name}')

        if save_csv:
            table3.to_csv(f'{OUTPUT_DATA_PATH}{folder_name}/{filename}.csv')
        if save_geojson:
            table3.columns = [str(x) for x in table3.columns]
            df_to_gdf(table3).to_file(
                f'{OUTPUT_DATA_PATH}{folder_name}/{filename}.geojson', driver='GeoJSON')

    return table3_dict