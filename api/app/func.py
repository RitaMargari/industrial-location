import numpy as np
import pandas as pd
import geopandas as gpd
import json

from typing import Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from catboost import Pool
from pandas.core.frame import DataFrame
from geopandas.geodataframe import GeoDataFrame
from shapely.geometry import LineString
from shapely import wkt


'''Returns json containing the ontology of industries (full [industry_code = None] or for specified industry)'''
def get_ontology_industry(ontology: DataFrame, industry_code: Optional[str] = None):

    industry = ontology[ontology["industry_code"] == industry_code] if industry_code else ontology
    return industry.reset_index(drop=True)

'''Returns dict with key-value pairs speciality_id-speciality for all industries [industry_code = None] or for specified one'''
def get_ontology_specialities(ontology: DataFrame, industry_code: Optional[str] = None):

    industry = ontology[ontology["industry_code"] == industry_code] if industry_code else ontology
    specialities = industry.set_index("speciality_id")["speciality"].to_dict()
    return specialities


'''Returns dict with key-value pairs edu_group_id-edu_group for all industries [industry_code = None] or for specified one'''
def get_ontology_edu_groups(ontology: DataFrame, industry_code: Optional[str] = None):

    industry = ontology[ontology["industry_code"] == industry_code] if industry_code else ontology
    edu_groups = dict(((
        i, industry[industry["type"] == i].set_index("edu_group_id")["edu_group"].to_dict()
        ) for i in industry["type"].unique()))
    return edu_groups


'''
Returns:
    (1) a point-based geojson containing:
      the estimates that reflect the potential of industry development in cities of Russian Federation and cities' statistics.
    (2) a linestring-based geojson containing:
      the number of responses from job-seekers in one city to vacancies in another city.
    
   Parameters:
        workforce_type (str) - specifies the type of workforce that is used in estimation ('all', "specialists", 'graduates')
        specialities (dict) - specifies specialisties ids (as keys) and their weights (as values)
        edu_groups (dict) - specifies edu groups ids (as keys) and their weights (as values)
        links_output (bool) - specifies if the migration links are the part of the output
'''
def get_potential_estimates(ontology: DataFrame, cv: DataFrame, graduates: DataFrame, cities: GeoDataFrame,
                            responses: DataFrame, vacancy: DataFrame, 
                            workforce_type: str = 'all', specialities: Optional[dict] = None, 
                            edu_groups: Optional[dict] = None, links_output:bool=False):

    for var, var_name in zip([specialities, edu_groups], ["specialities", "edu_groups"]):
        if var is None: break
        if not all(isinstance(eval(x), int) for x in var.keys()): 
            raise TypeError(f"The keys in {var_name} must be inetegers.")
        if not all(isinstance(x, float) or isinstance(x, int) for x in var.values()): 
            raise TypeError(f"The values in {var_name} must be floats from 0 to 1.")
        if not all((x >= 0 and x <=1) for x in var.values()): 
            raise ValueError(f"The values in {var_name} must be between 0 and 1.")
    
    if workforce_type not in ["all", "graduates", "specialists"]: 
        raise ValueError(f"The workforce type '{workforce_type}'' is not supported.")

    YEAR = 2021

    cv_loc = cv[cv["year"] == YEAR]
    vacancy_loc = vacancy[vacancy["year"] == YEAR]
    responses_loc = responses[responses["year"] == YEAR]

    all_specialities = ontology.set_index("speciality_id")["speciality"].drop_duplicates()
    all_edu_groups = ontology.set_index("edu_group_id")[["type", "edu_group_code", "edu_group"]].drop_duplicates()

    cities = cities.set_index("region_city")
    cities["estimate"] = 0

    if workforce_type == "graduates" or workforce_type == "all":

        if not edu_groups: raise TypeError(f"With workforce_type == '{workforce_type}' edu_groups can't be None")

        edu_groups = {int(k):float(v) for k,v in edu_groups.items()}
        if not all(x in all_edu_groups.index for x in edu_groups.keys()): 
            fail_ids = [x for x in edu_groups.keys() if x not in all_edu_groups.index]
            raise ValueError(f"Educational groups' ids {fail_ids} don't present in the current ontology")
        
        edu_groups = pd.concat((all_edu_groups.loc[edu_groups.keys()], pd.Series(edu_groups).rename("weights")), axis=1)
        edu_groups = edu_groups.set_index(["type", "edu_group_code"])

        # get estimates based on the number of potential graduates
        cities = estimate_graduates(graduates, cities, edu_groups)

    if  workforce_type == "specialists" or workforce_type == "all":

        if not specialities: raise TypeError(f"With workforce_type == '{workforce_type}' specialities can't be None")

        specialities = {int(k):float(v) for k,v in specialities.items()}
        if not all(x in all_specialities.index for x in specialities.keys()): 
            fail_ids = [x for x in specialities.keys() if x not in all_specialities.index]
            raise ValueError(f"Specialities' ids {fail_ids} don't present in the current ontology")

        specialities = pd.DataFrame(
            list(zip(all_specialities.loc[specialities.keys()].to_list(), specialities.values())),
            columns=["speciality", "weights"]
        )

        # get estimates based on the number of open resumes of relevant specialists
        cities = estimate_cv(cv_loc, cities, specialities)

    
    # get estimates based on observed pseudo migration (responses stat)
    cities = estimate_migration(cities)

    # if links_output is True, prepare the strongest (migration) connections between cities
    links_json = get_migration_links(responses_loc, cities) if links_output else None

    # scale estimates
    scaler = MinMaxScaler()
    cities["estimate"] = pd.Series(
        scaler.fit_transform(np.expand_dims(cities["estimate"].to_numpy(), 1)).squeeze(),
        index=cities.index
        )
        
    cities = cities.sort_values(by="estimate", ascending=False)
    cities["estimate"] = cities["estimate"].round(3)
    cities = cities.reset_index()
        
    return  {"estimates": cities, "links": links_json}


'''
Returns GeoDataFrame that contains the estimates of industry development potential for cities of Russian Federation
based on the information about graduates
'''
def estimate_graduates(graduates: DataFrame, cities: GeoDataFrame, edu_groups: DataFrame):
            
        graduates = pd.DataFrame(graduates.groupby(["region_city", "type", "edu_group_code"])["graduates_forecast"].sum())
        graduates = graduates.join(edu_groups, on=["type", "edu_group_code"]).dropna(subset=["weights"])
        graduates = graduates.reset_index().set_index(["region_city", "edu_group"])
        graduates["graduates_weighted"] = graduates["graduates_forecast"] * graduates["weights"]

        graduates_cities_gr = graduates.groupby(["region_city"])
        graduates_cities = graduates_cities_gr[["graduates_forecast", "graduates_weighted"]].sum().add_suffix("_sum")

        graduates_cities["graduates_forecast"] = graduates_cities_gr.apply(
            lambda x: x["graduates_forecast"].droplevel("region_city").to_dict()
            )

        graduates_cities["graduates_forecast"] = graduates_cities["graduates_forecast"].apply(lambda x: str(x) if x else x)
        cities = cities.join(graduates_cities, how="left")
        cities[["graduates_weighted_sum", "graduates_forecast_sum"]] = cities[["graduates_weighted_sum", "graduates_forecast_sum"]].fillna(0)

        scaler = MinMaxScaler()
        graduates_estimate = pd.Series(
            scaler.fit_transform(np.expand_dims(np.log(cities["graduates_weighted_sum"].to_numpy() + 10e-06), 1)).squeeze(),
            index=cities.index
            )
        cities["estimate"] = cities["estimate"].add(graduates_estimate).fillna(cities["estimate"]).dropna()
        cities = cities.rename(
            columns={"graduates_forecast": "graduates_forecast_number", "graduates_forecast_sum": "graduates_forecast_sum_number"}
            )

        return cities


'''
Returns GeoDataFrame that contains the estimates of industry development potential for cities of Russian Federation
based on the information about open CVs
'''
def estimate_cv(cv: DataFrame, cities: GeoDataFrame, specialities: DataFrame):

    cv_select = cv[cv["hh_name"].isin(specialities["speciality"])]
    if len(cv_select) == 0:
        cities["specialists_sum_number"] = 0
        cities["specialists_number"] = None
        cities["estimate"] = cities["estimate"].add(0).fillna(cities["estimate"]).dropna()
        return cities

    cv_select = cv_select.groupby(["cluster_center", "hh_name"])["id_cv"].count().rename("cv_count").reset_index()
    cv_select = cv_select.join(specialities.set_index("speciality")["weights"], on="hh_name")
    cv_select["cv_count_weighted"] = cv_select["cv_count"] * cv_select["weights"]

    cv_cities_gr = cv_select.groupby(["cluster_center"])
    cv_cities = cv_cities_gr[["cv_count", "cv_count_weighted"]].sum().add_suffix("_sum")
    cv_cities["cv_count"] = cv_cities_gr[["hh_name", "cv_count"]].apply(
        lambda x: x.set_index("hh_name")["cv_count"].to_dict()
        )
    cv_cities["cv_count"] = cv_cities["cv_count"].apply(lambda x: str(x) if x else x)
    cities = cities.join(cv_cities, how="left")
    cities[["cv_count_weighted_sum", "cv_count_sum"]] = cities[["cv_count_weighted_sum", "cv_count_sum"]].fillna(0)

    scaler = MinMaxScaler()
    cv_estimate = pd.Series(
        scaler.fit_transform(np.expand_dims(np.log(cities["cv_count_weighted_sum"].to_numpy() + 10e-06), 1)).squeeze(),
        index=cities.index
        )
    cities["estimate"] = cities["estimate"].add(cv_estimate).fillna(cities["estimate"]).dropna()
    cities = cities.rename(columns={"cv_count": "specialists_number", "cv_count_sum": "specialists_sum_number"})

    return cities


'''
Returns GeoDataFrame that contains cities' information about the number of open CVs and vacancies, min/max/median salary offered in industry
'''
def get_cities_stat(vacancy: DataFrame, cities: GeoDataFrame):

    cities = cities.join(vacancy.groupby(["region_city"]).agg(
        {"min_salary": "min", "max_salary": "max", "median_salary": "median", "id_vacancy": "count"}
        ).rename(columns={"id_vacancy": "vacancies_count_in_industries", 
                          "min_salary": "min_salary_in_industries",
                          "max_salary": "max_salary_in_industries",
                          "median_salary": "median_salary_in_industries"})
                          )
    
    return cities


'''
Returns GeoDataFrame that contains the estimates of industry development potential for cities of Russian Federation
based on the megration (responses) stat
'''
def estimate_migration(cities: GeoDataFrame): 

    # based on precalculated one_vacancy_out_response and probability_to_move

    scaler = MinMaxScaler()
    column_norm = ["one_vacancy_out_response", "probability_to_move"]
    migration_estinate = pd.DataFrame(
        scaler.fit_transform(np.log(cities[column_norm].abs().to_numpy() + 10e-06)),
        index=cities.index, columns=column_norm
    )
    cities["estimate"] = cities["estimate"].add(
        migration_estinate["one_vacancy_out_response"] 
        - migration_estinate["probability_to_move"]
        )

    return cities


'''
Returns FeatureCollection that contains the number of job seekers' responses from city_source 
to a job vacancies from city_destination 
'''
def get_migration_links(responses: DataFrame, cities: GeoDataFrame):

    links = responses.groupby(["cluster_center_cv", "cluster_center_vacancy"])["year"].count()
    links = links.rename("num_responses").reset_index()
    links = links.rename(columns={"cluster_center_cv": "city_source", "cluster_center_vacancy": "city_destination"})
    links = links[links["city_source"] != links["city_destination"]]
    links = links[links["num_responses"] >= links["num_responses"].quantile(0.95)]
    links["geometry"] = links.apply(
        lambda x: LineString((cities.loc[x.city_source]["geometry"], cities.loc[x.city_destination]["geometry"])), 
        axis=1)
    links = gpd.GeoDataFrame(links).sort_values(by=["num_responses"], ascending=True)

    return links


'''
Returns GeoDataFrame that contains migration (responses) connections of chosen city with other ones

Parameters:
        city_selected (str) - name of the selected city

'''
def get_city_migration_links(responses: DataFrame, cities: GeoDataFrame, city_selected: str):
        
    if city_selected not in list(cities["region_city"]):
        raise ValueError(f"The city {city_selected} is not supported")

    YEAR = 2021
    cities = cities.set_index("region_city")
    cities_geometry = cities["geometry"]

    responses_loc = responses[responses['year'] == YEAR]
    responses_loc = responses[responses["cluster_center_cv"] != responses["cluster_center_vacancy"]]
    responses_loc = responses_loc[
        (responses_loc["cluster_center_cv"] == city_selected) | 
        (responses_loc["cluster_center_vacancy"]== city_selected)
        ]

    responses_aggr = responses_loc.groupby(["cluster_center_cv", "cluster_center_vacancy"])
    responses_aggr = responses_aggr.count()["id_cv"].rename("responses").reset_index()
    responses_aggr["geometry"] = responses_aggr.apply(lambda x: LineString(
        (cities_geometry.loc[x.cluster_center_cv], cities_geometry.loc[x.cluster_center_vacancy])
        ), axis=1)
    
    responses_aggr["direction"] = None
    responses_aggr.loc[responses_aggr["cluster_center_cv"] == city_selected, "direction"] = "out"
    responses_aggr.loc[responses_aggr["cluster_center_vacancy"] == city_selected, "direction"] = "in"

    responses_aggr = gpd.GeoDataFrame(responses_aggr)
    return responses_aggr


'''
Returns GeoDataFrame that contains agglomeration connections of chosen city with other ones

Parameters:
        city_selected (str) - name of the selected city
'''
def get_city_agglomeration_links(agglomerations: DataFrame, cities: DataFrame, city_selected: str):

    if city_selected not in list(cities["region_city"]):
        raise ValueError(f"The city {city_selected} is not supported")

    cities = cities.set_index("region_city")
    cities_geometry = cities["geometry"]    

    aggl_loc = agglomerations[agglomerations["cluster_center"] == city_selected]
    aggl_loc = aggl_loc[aggl_loc["cluster_city"] != aggl_loc["cluster_center"]]

    city_coord = cities_geometry.loc[city_selected]
    aggl_loc["coordinates"] = aggl_loc["coordinates"].apply(wkt.loads)
    aggl_loc["geometry"] = aggl_loc["coordinates"].apply(lambda x: LineString((x, city_coord)))

    aggl_links = gpd.GeoDataFrame(aggl_loc[["cluster_city", "cluster_center", "geometry"]])
    aggl_nodes = gpd.GeoDataFrame(aggl_loc[["coordinates", "cluster_city"]].rename(columns={"coordinates": "geometry"}))

    return {"links": aggl_links, "nodes": aggl_nodes}



'''
Returns:
    (1) a point-based geojson containing features of the selected city, in particular:
      recalculate estimate that reflect the potential of industry development in cities of Russian Federation (based on changed city's stats)
    (2) a linestring-based geojson containing:
      recalculated number of responses from job-seekers over the country to vacancies in the selected city
    (3) dict containing changes in estimate and num_in_migration occured due to the recalculation
    
   Parameters:
        cities (GeoDataFrame) - GeoDataFrame obtained as a result of function get_potential_estimates
        city_name (str) - name of the selected city
'''

def predict_migration(cities_compare: GeoDataFrame, responses: DataFrame, DM: DataFrame, model, 
                      cities: GeoDataFrame, city_name: str):

    cities = cities.set_index("region_city")
    cities_compare = cities_compare.set_index("region_city")

    # features used in model for prediction
    columns = [
        "city_category", 
        "harsh_climate", 
        "ueqi_residential", 
        "ueqi_street_networks", 
        "ueqi_green_spaces", 
        "ueqi_public_and_business_infrastructure", 
        "ueqi_social_and_leisure_infrastructure",
        "ueqi_citywide_space",
        "cvs_count_all",
        "vacancies_count_all",
        "factories_total",
        'median_salary_all'
        ]

    # check if there is any changes in cities' stats
    if cities_compare[columns].loc[cities.index].equals(cities[columns]):
        # if there is no changes, return initial table 
        city_update, update_dict, migration = get_response_no_changes(cities, cities_compare, city_name, responses)
    else:
        # if there are some changes, recalculate num_in_migration and estimate for the selected city 
        city_update, update_dict, migration = recalculate(cities, columns, city_name, DM, model)

    return {
        'city_features': city_update, 
        'update_dict': update_dict, 
        'new_links': migration
        }


'''
Returns GeoDataFrame with recalculated estimate, GeoDataFrame with recalculated migration links and dict with changes
'''
def recalculate(cities, columns, city_name, DM, model): 

    # prepare features for the model
    cities_features = decode_features(cities, columns)

    # create df for predictions
    cities_index = cities_features.index
    responses_predict = pd.DataFrame({
        'cluster_center_cv': [i for i in cities_index], 
        'cluster_center_vacancy': [city_name for i in range(len(cities_index))],
        'distance': None})

    # add distance from distance matrix
    responses_predict = responses_predict.set_index('cluster_center_cv')
    responses_predict['distance'] = DM.loc[city_name]
    responses_predict = responses_predict.reset_index()

    # form an input vecor x
    responses_predict['x'] = list(np.concatenate((
        cities_features.loc[list(responses_predict['cluster_center_cv'])].drop(['vacancies_count_all'], axis=1).to_numpy(), 
        cities_features.loc[list(responses_predict['cluster_center_vacancy'])].drop(['cvs_count_all'], axis=1).to_numpy(), 
        responses_predict['distance'].astype('float').round(3).to_numpy().reshape(len(responses_predict), 1)
        ), axis=1))

    # predict migration and process the response
    responses_predict['responses'] = model.predict(Pool(data=responses_predict['x']))
    responses_predict.loc[responses_predict['responses'] < 0, 'responses'] = 0
    responses_predict['responses'] = responses_predict['responses'].apply(lambda x: 0 if x < 0.6 else x)
    responses_predict['responses'] = responses_predict['responses'].round()

    # create new migration links
    migration = responses_predict[responses_predict["cluster_center_cv"] != responses_predict["cluster_center_vacancy"]]
    migration = gpd.GeoDataFrame(migration)
    migration['distance'] = migration['distance'].round(2)

    cities_geometry = cities["geometry"]
    migration["geometry"] = migration.apply(lambda x: LineString(
        (cities_geometry.loc[x.cluster_center_cv], cities_geometry.loc[x.cluster_center_vacancy])
        ), axis=1)

    # recalculate num_in_migration
    num_vacancy = cities["vacancies_count_all"] # the total number of relevant vacancies in a city
    num_in_migration = migration[migration["cluster_center_vacancy"] == city_name].set_index("cluster_center_cv")["responses"].sum() 

    # num_responses = cities["num_responses"] # the total number of responses on vacancies in a city
    # num_out_migration = migration[migration["cluster_center_cv"] == city_name]["responses"].sum()

    cities_update = cities.copy()
    cities_update.loc[city_name, 'num_in_migration'] = num_in_migration
    cities_update.loc[city_name, 'one_vacancy_out_response'] = (num_in_migration / num_vacancy.loc[city_name]).round(3)
    # cities.loc[city_name, 'probability_to_move'] = num_out_migration / num_responses
    
    # rescale estimates
    scaler = MinMaxScaler()
    column_norm = ["cv_count_weighted_sum", "graduates_weighted_sum", "probability_to_move", "one_vacancy_out_response"]
    migration_estinate = pd.DataFrame(
        scaler.fit_transform(np.log(cities_update[column_norm].abs().to_numpy() + 10e-06)),
        index=cities_update.index, columns=column_norm
    )

    cities_update["estimate"] = 0
    cities_update["estimate"] = migration_estinate['cv_count_weighted_sum'] \
                        + migration_estinate['graduates_weighted_sum'] \
                        + migration_estinate["one_vacancy_out_response"] \
                        - migration_estinate["probability_to_move"]

    scaler = MinMaxScaler()
    cities_update["estimate"] = pd.Series(
        scaler.fit_transform(np.expand_dims(cities_update["estimate"].to_numpy(), 1)).squeeze(),
        index=cities_update.index
        ).fillna(0).round(3)
    
    city_update = cities_update.loc[[city_name]]
    update_dict = {
        'before': {'estimate': float(cities['estimate'][city_name]), 'in_migration': int(cities['num_in_migration'][city_name])},
        'after': {'estimate': float(city_update['estimate'][city_name]), 'in_migration': int(city_update['num_in_migration'][city_name])}
        }
    
    return city_update, update_dict, migration.drop(['x'], axis=1)


'''
Returns GeoDataFrame with initial estimate, GeoDataFrame with calculated migration links and dict with no changes
'''
def get_response_no_changes(cities, cities_compare, city_name, responses):

    YEAR = 2021

    city_update = cities.loc[[city_name]]
    estimate = float(cities['estimate'][city_name])
    in_migration = int(cities['num_in_migration'][city_name])

    update_dict = {
            'before': {'estimate': estimate, 'in_migration': in_migration},
            'after': {'estimate': estimate, 'in_migration': in_migration}
        }
    
    responses_loc = responses[responses['year'] == YEAR]
    migration = get_city_migration_links(responses_loc, cities_compare.reset_index(), city_name)
    
    return city_update, update_dict, migration


'''
Returns GeoDataFrame with cities' features transformed for the model input
'''
def decode_features(df, columns):

    df = df.dropna(subset=columns)
    df_features = df[columns]
    one_hot = OneHotEncoder(drop=['Большой город'])
    encoded_category = one_hot.fit_transform(np.expand_dims(df["city_category"].to_numpy(), 1)).toarray()
    encoded_category_names = one_hot.get_feature_names_out(["category"])
    df_features.loc[:, encoded_category_names] = encoded_category
    df_features = df_features.drop(["city_category"], axis=1)
    df_features["harsh_climate"] = df_features["harsh_climate"].astype(int)
    return df_features
