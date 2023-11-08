import numpy as np
import pandas as pd
import geopandas as gpd
import json

from typing import Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from catboost import Pool
from pandas.core.frame import DataFrame
from shapely.geometry import LineString
from shapely import wkt


'''Returns json containing the ontology of industries (full [industry_code = None] or for specified industry)'''
def get_ontology_industry(ontology: DataFrame, industry_code: Optional[str] = None):

    industry = ontology[ontology["industry_code"] == industry_code] if industry_code else ontology
    return eval(industry.reset_index(drop=True).to_json())


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
Returns geojson containing (1) the estimates that reflect the potential of industry development in cities of Russian Federation
                       and (2) cities' statistics
   
   Parameters:
        workforce_type (str) - specifies the type of workforce that is used in estimation ('all', "specialists", 'graduates')
        specialities (dict) - specifies specialisties ids (as keys) and their weights (as values)
        edu_groups (dict) - specifies edu groups ids (as keys) and their weights (as values)
        links_output (bool) - specifies if the migration links are the part of the output
'''
def get_potential_estimates(ontology: DataFrame, cv: DataFrame, graduates: DataFrame, cities: DataFrame,
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
        
        edu_groups_relevant_prof = ontology[ontology["edu_group_id"].isin(edu_groups.keys())]["speciality"]
        edu_groups = pd.concat((all_edu_groups.loc[edu_groups.keys()], pd.Series(edu_groups).rename("weights")), axis=1)
        edu_groups = edu_groups.set_index(["type", "edu_group_code"])

        # get estimates based on the number of potential graduates
        cities = estimate_graduates(graduates, cities, edu_groups)

    else: 
        edu_groups_relevant_prof = None

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
        specialities_names = specialities["speciality"]

    else: 
        specialities_names = None


    # frame the complete list of the specialists based on the 'specialities' and 'edu_groups' variables
    specialities_names = list(pd.concat((specialities_names, edu_groups_relevant_prof)).drop_duplicates())

    cv_loc = cv_loc[cv_loc["hh_name"].isin(specialities_names)]
    vacancy_loc = vacancy_loc[vacancy_loc["title_hh"].isin(specialities_names)]
    responses_loc = responses_loc[responses_loc["hh_name"].isin(specialities_names)]

    # get descriptive characteristics of cities 
    # (city type, quality of urban environment, number of open resume/vacancies, salary offered)
    cities = get_cities_stat(cv, vacancy, cities)
    
    # get estimates based on observed pseudo migration (responses stat)
    cities = estimate_migration(responses, cities)

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
        
    return  {"estimates": json.loads(cities.to_json()), "links": links_json}


'''
Returns GeoDataFrame that contains the estimates of industry development potential for cities of Russian Federation
based on the information about graduates
'''
def estimate_graduates(graduates: DataFrame, cities: DataFrame, edu_groups: DataFrame):
            
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

        scaler = MinMaxScaler()
        graduates_estimate = pd.Series(
            scaler.fit_transform(np.expand_dims(graduates_cities["graduates_weighted_sum"].to_numpy(), 1)).squeeze(),
            index=graduates_cities.index
            )
        cities["estimate"] = cities["estimate"].add(graduates_estimate).fillna(cities["estimate"]).dropna()

        graduates_cities = graduates_cities.drop(["graduates_weighted_sum"], axis=1)
        graduates_cities = graduates_cities.rename(
            columns={"graduates_forecast": "graduates_forecast_number", "graduates_forecast_sum": "graduates_forecast_sum_number"}
            )
        cities = cities.join(graduates_cities, how="left")
        cities = cities.fillna({"graduates_forecast_sum_number": 0})

        return cities


'''
Returns GeoDataFrame that contains the estimates of industry development potential for cities of Russian Federation
based on the information about open CVs
'''
def estimate_cv(cv: DataFrame, cities: DataFrame, specialities: DataFrame):

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

    scaler = MinMaxScaler()
    cv_estimate = pd.Series(
        scaler.fit_transform(np.expand_dims(cv_cities["cv_count_weighted_sum"].to_numpy(), 1)).squeeze(),
        index=cv_cities.index
        )
    cities["estimate"] = cities["estimate"].add(cv_estimate).fillna(cities["estimate"]).dropna()

    cv_cities = cv_cities.drop(["cv_count_weighted_sum"], axis=1)
    cv_cities = cv_cities.rename(columns={"cv_count": "specialists_number", "cv_count_sum": "specialists_sum_number"})
    cities = cities.join(cv_cities, how="left")
    cities = cities.fillna({"specialists_sum_number": 0})

    return cities


'''
Returns GeoDataFrame that contains cities' information about the number of open resumes and vacancies, min/max/median salary offered
'''
def get_cities_stat(cv, vacancy, cities):
    
    # the number of cvs is counted over city's agglomeration (by 'cluster_center' field) since we assume people will be 
    # willing to move there if the new fabric is open
    cities = cities.join(cv.groupby(["cluster_center"])["id_cv"].count().rename("cv_count"))

    # however, for vacancies we use feature 'region_city' which points out exact city as the characteristics of this city
    # will influence the desire of other people (outside of agglomeration) to move there 
    cities = cities.join(vacancy.groupby(["region_city"]).agg(
        {"min_salary": "min", "max_salary": "max", "median_salary": "median", "id_vacancy": "count"}
        ).rename(columns={"id_vacancy": "vacancy_count"}))
    
    return cities


'''
Returns GeoDataFrame that contains the estimates of industry development potential for cities of Russian Federation
based on the megration (responses) stat
'''
def estimate_migration(responses, cities): 

    num_vacancy = cities["vacancy_count"] # the total number of relevan vacancies in a city
    num_responses = responses.groupby(["cluster_center_cv"])["id_cv"].count() # the total number of responses on vacancies in a city

    migration = responses[responses["cluster_center_cv"] != responses["cluster_center_vacancy"]]

    num_in_migration = migration.groupby(["cluster_center_vacancy"])["id_cv"].count()
    num_out_migration = migration.groupby(["cluster_center_cv"])["id_cv"].count() # the number of responses on vacancies in other cities
    cities = cities.join(pd.concat((
        num_in_migration.rename("num_in_migration"), num_out_migration.rename("num_out_migration")
        ), axis=1), on="region_city").fillna(0)

    in_coef = (num_in_migration / num_vacancy).fillna(0)
    out_coef = (num_out_migration / num_responses).fillna(0)

    migration_stat = pd.concat([in_coef.rename("one_vacancy_out_response"), out_coef.rename("probability_to_move")], axis=1)
    cities = cities.join(migration_stat.round(3), on="region_city").fillna(0)

    scaler = MinMaxScaler()
    column_norm = ["one_vacancy_out_response", "probability_to_move"]
    migration_estinate = pd.DataFrame(
        scaler.fit_transform(cities[column_norm].to_numpy()),
        index=cities.index, columns=column_norm
    )
    cities["estimate"] = cities["estimate"].add(
        migration_estinate["one_vacancy_out_response"] - migration_estinate["probability_to_move"]
        )
    
    return cities


'''
Returns FeatureCollection that contains the number of job seekers' responses from city_source 
to a job vacancies from city_destination 
'''
def get_migration_links(responses, cities):

    links = responses.groupby(["cluster_center_cv", "cluster_center_vacancy"])["year"].count()
    links = links.rename("num_responses").reset_index()
    links = links.rename(columns={"cluster_center_cv": "city_source", "cluster_center_vacancy": "city_destination"})
    links = links[links["city_source"] != links["city_destination"]]
    links = links[links["num_responses"] >= links["num_responses"].quantile(0.95)]
    links["geometry"] = links.apply(
        lambda x: LineString((cities.loc[x.city_source]["geometry"], cities.loc[x.city_destination]["geometry"])), 
        axis=1)
    links = gpd.GeoDataFrame(links).sort_values(by=["num_responses"], ascending=True)

    return json.loads(links.to_json())


'''
Returns GeoDataFrame that contains migration connections of chosen city with other ones
'''
def get_city_migration_links(responses: DataFrame, cities:DataFrame, specialists: list, city_selected:str):

    if not all(isinstance(x, int) for x in specialists): 
        raise TypeError(f"The keys in specialists must be inetegers.")
        
    if city_selected not in list(cities["region_city"]):
        raise ValueError(f"The city {city_selected} is not supported")

    cities = cities.set_index("region_city")
    cities_geometry = cities["geometry"]

    year = 2021
    responses_loc = responses[responses["year"] == year]
    responses_loc = responses_loc[responses_loc["cluster_center_cv"] != responses_loc["cluster_center_vacancy"]]
    responses_loc = responses_loc[responses_loc["speciality_id"].isin(specialists)]
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
    return json.loads(responses_aggr.to_json())


'''
Returns GeoDataFrame that contains agglomeration connections of chosen city with other ones
'''
def get_city_agglomeration_links(agglomerations:DataFrame, cities:DataFrame, city_selected:str):

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

    return {"links": json.loads(aggl_links.to_json()), "nodes": json.loads(aggl_nodes.to_json())}


def predict_links(cv: pd.DataFrame, vacancy: pd.DataFrame, DM: pd.DataFrame, cities: gpd.GeoDataFrame, 
                  model, city_name: str, responses):
    
    city = cities[cities["region_city"] == city_name]

    cities = cities.set_index("region_city")
    # cities = get_cities_stat(cv, vacancy, cities)
    cities = cities[cities["city_category"] != 0]

    # cities = pd.concat((cities[cities.index != city_name], city.set_index("region_city")))
    cities_features = decode_features(cities)

    other_cities = cities_features.index
    responses = pd.DataFrame({
        'cluster_center_cv': [city_name for i in range(len(other_cities))] + [i for i in other_cities], 
        'cluster_center_vacancy': [i for i in other_cities] + [city_name for i in range(len(other_cities))],
        'distance': None})

    responses = responses.set_index('cluster_center_vacancy')
    responses['distance'] = DM.loc[city_name]

    responses = responses.reset_index().set_index('cluster_center_cv')
    responses.loc[responses.index != city_name, 'distance'] = DM.loc[other_cities, city_name]
    responses = responses.reset_index().drop_duplicates()

    responses['x'] = list(np.concatenate((
        cities_features.loc[list(responses['cluster_center_cv'])].to_numpy(), 
        cities_features.loc[list(responses['cluster_center_vacancy'])].to_numpy(), 
        responses['distance'].astype('float').round(3).to_numpy().reshape(len(responses), 1)
        ), axis=1))

    responses['responses'] = model.predict(Pool(data=responses['x']))
    responses.loc[responses['responses'] < 0, 'responses'] = 0
    responses['responses'] = responses['responses'].apply(lambda x: 0 if x < 1 else x)
    responses['responses'] = responses['responses'].round()


    num_vacancy = cities.loc[city_name, "vacancy_count"] # the total number of relevan vacancies in a city
    num_responses = responses[responses["cluster_center_cv"] == city_name]['responses'].sum() # the total number of responses on vacancies in a city

    migration = responses[responses["cluster_center_cv"] != responses["cluster_center_vacancy"]]

    num_out_migration = migration[migration["cluster_center_cv"] == city_name]["responses"].sum()
    num_in_migration = migration[migration["cluster_center_vacancy"] == city_name]["responses"].sum() # the number of responses on vacancies in other cities

    cities.loc[city_name, 'one_vacancy_out_response'] = num_in_migration / num_vacancy
    cities.loc[city_name, 'probability_to_move'] = num_out_migration / num_responses

    scaler = MinMaxScaler()
    column_norm = ["specialists_sum_number", "graduates_forecast_sum_number", "probability_to_move", "one_vacancy_out_response"]
    migration_estinate = pd.DataFrame(
        scaler.fit_transform(cities[column_norm].to_numpy()),
        index=cities.index, columns=column_norm
    )
    cities["estimate"] = migration_estinate['specialists_sum_number'] \
                        + migration_estinate['graduates_forecast_sum_number'] \
                        + migration_estinate["one_vacancy_out_response"] \
                        - migration_estinate["probability_to_move"]

    scaler = MinMaxScaler()
    cities["estimate"] = pd.Series(
        scaler.fit_transform(np.expand_dims(cities["estimate"].to_numpy(), 1)).squeeze(),
        index=cities.index
        )
    
    return cities.loc[[city_name]]

def decode_features(df):

    df_features = df[[
        # 'population',
        "city_category", 
        "harsh_climate", 
        "ueqi_residential", 
        "ueqi_street_networks", 
        "ueqi_green_spaces", 
        "ueqi_public_and_business_infrastructure", 
        "ueqi_social_and_leisure_infrastructure",
        "ueqi_citywide_space",
        "cv_count",
        "vacancy_count",
        "factories_total",
        'min_salary',
        'max_salary',
        'median_salary'
        ]]
    
    # df_features['cv_count'] = df_features['cv_count'] / df_features['population']
    # df_features['vacancy_count'] = df_features['vacancy_count'] / df_features['population']
    # df_features = df_features.drop(['population'], axis=1)

    one_hot = OneHotEncoder(drop='first')
    encoded_category = one_hot.fit_transform(np.expand_dims(df["city_category"].to_numpy(), 1)).toarray()
    encoded_category_names = one_hot.get_feature_names_out(["category"])
    df_features.loc[:, encoded_category_names] = encoded_category
    df_features = df_features.drop(["city_category"], axis=1)
    df_features["harsh_climate"] = df_features["harsh_climate"].astype(int).fillna(0)

    return df_features
