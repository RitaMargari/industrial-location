import numpy as np
import pandas as pd
import geopandas as gpd
import json

from typing import Optional
from sklearn.preprocessing import MinMaxScaler
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
Returns geojson containing the estimates that reflect the potential of industry development in cities of Russian Federation
   
   Parameters:
        workforce_type (str) - specifies the type of workforce that is used in estimation ('all', "specialists", 'graduates')
        specialities (dict) - specifies specialisties ids (as keys) and their weights (as values)
        edu_groups (dict) - specifies edu groups ids (as keys) and their weights (as values)
'''
def get_potential_estimates(ontology: DataFrame, cv: DataFrame, graduates: DataFrame, cities: DataFrame,
                            responses: DataFrame, vacancy: DataFrame, workforce_type: str = 'all', 
                            specialities: Optional[dict] = None, edu_groups: Optional[dict] = None, 
                            links_output:bool=False):

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

    year = 2021
    cv = cv[cv["year"] == year]
    vacancy = vacancy[vacancy["year"] == year]
    responses = responses[responses["year"] == year]

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
        cities = estimate_cv(cv, cities, specialities)

        cities = estimate_migration(responses, cv, vacancy, cities)

        scaler = MinMaxScaler()
        cities["estimate"] = pd.Series(
            scaler.fit_transform(np.expand_dims(cities["estimate"].to_numpy(), 1)).squeeze(),
            index=cities.index
            )

    links_json = get_links(cities, responses, all_specialities) if links_output else None
        
    cities = cities.sort_values(by="estimate", ascending=False)
    cities["estimate"] = cities["estimate"].round(3)
    cities = cities.reset_index().reindex([
        "region", "city", "region_city", "population", "estimate", "graduates_forecast_number", 
        "graduates_forecast_sum_number", "specialists_number", "specialists_sum_number", "one_vacancy_out_response", 
        "probability_to_move", "geometry"
        ], axis=1)
        
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

    cv_select = cv_select.groupby(["region_city", "hh_name"])["id_cv"].count().rename("cv_count").reset_index()
    cv_select = cv_select.join(specialities.set_index("speciality")["weights"], on="hh_name")
    cv_select["cv_count_weighted"] = cv_select["cv_count"] * cv_select["weights"]

    cv_cities_gr = cv_select.groupby(["region_city"])
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

def estimate_migration(responses, cv, vacancy, cities): 
    
    migration = responses[responses["cluster_center_cv"] != responses["cluster_center_vacancy"]]

    num_in_migration = migration.groupby(["cluster_center_vacancy"])["id_cv"].nunique()
    num_out_migration = migration.groupby(["cluster_center_cv"])["id_cv"].nunique()

    num_vacancy = vacancy.groupby(["cluster_center"])["id_vacancy"].nunique()
    num_resume = cv.groupby(["cluster_center"])["id_cv"].nunique()

    in_coef = (num_in_migration / num_vacancy).fillna(0)
    out_coef = (num_out_migration / num_resume).fillna(0)

    migration_stat = pd.concat([in_coef.rename("one_vacancy_out_response"), out_coef.rename("probability_to_move")], axis=1)
    cities = cities.join(migration_stat, on="region_city").fillna(0)

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
Returns FeatureCollection that contains the number of responses of job seekers from city_source 
to a job vacancies from city_destination 
'''
def get_links(cities, responses, all_specialities):

    responses_match = responses[responses["hh_name"].isin(all_specialities)]
    links = responses_match.groupby(["cluster_center_cv", "cluster_center_vacancy"])["year"].count()
    links = links.rename("num_responses").reset_index()
    links = links.rename(columns={"cluster_center_cv": "city_source", "cluster_center_vacancy": "city_destination"})
    links = links[links["city_source"] != links["city_destination"]]
    links = links[links["num_responses"] >= links["num_responses"].quantile(0.95)]
    links["geometry"] = links.apply(
        lambda x: LineString((cities.loc[x.city_source]["geometry"], cities.loc[x.city_destination]["geometry"])), 
        axis=1)
    links = gpd.GeoDataFrame(links).sort_values(by=["num_responses"], ascending=False)

    return json.loads(links.to_json())


def get_migration_links(responses: DataFrame, cities:DataFrame, specialists: list, city_selected:str):

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


def get_agglomeration_links(agglomerations:DataFrame, cities:DataFrame, city_selected:str):

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