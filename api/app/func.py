import numpy as np
import pandas as pd
import json

from typing import Optional
from sklearn.preprocessing import MinMaxScaler
from pandas.core.frame import DataFrame


'''Returns json containing the ontology of industries (full [idustry_code = None] or for specified industry)'''
def get_ontology_industry(ontology: DataFrame, idustry_code: Optional[str] = None):

    industry = ontology[ontology["idustry_code"] == idustry_code] if idustry_code else ontology
    return eval(industry.reset_index(drop=True).to_json())


'''Returns dict with key-value pairs speciality_id-speciality for all industries [idustry_code = None] or for specified one'''
def get_ontology_specialities(ontology: DataFrame, idustry_code: Optional[str] = None):

    industry = ontology[ontology["idustry_code"] == idustry_code] if idustry_code else ontology
    specialities = industry.set_index("speciality_id")["speciality"].to_dict()
    return specialities


'''Returns dict with key-value pairs edu_group_id-edu_group for all industries [idustry_code = None] or for specified one'''
def get_ontology_edu_groups(ontology: DataFrame, idustry_code: Optional[str] = None):

    industry = ontology[ontology["idustry_code"] == idustry_code] if idustry_code else ontology
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
                            workforce_type: str = 'all', specialities: Optional[dict] = None, 
                            edu_groups: Optional[dict] = None, raw_data=False):

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

    all_specialities = ontology.set_index("speciality_id")["speciality"].drop_duplicates()
    all_edu_groups = ontology.set_index("edu_group_id")[["type", "edu_group_code"]].drop_duplicates()

    cities = cities.set_index("city")
    cities["estimate"] = 0

    if workforce_type == "graduates" or workforce_type == "all":

        if not edu_groups: raise TypeError(f"With workforce_type == '{workforce_type}' edu_groups can't be None")

        edu_groups = {int(k):float(v) for k,v in edu_groups.items()}
        if not all(x in all_edu_groups.index for x in edu_groups.keys()): 
            fail_ids = [x for x in edu_groups.keys() if x not in all_edu_groups.index]
            raise ValueError(f"Educational groups' ids {fail_ids} don't present in the current ontology")

        edu_groups = all_edu_groups.loc[edu_groups.keys()].join(pd.Series(edu_groups).rename("weights"))
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

    if workforce_type == "all":
        scaler = MinMaxScaler()
        cities["estimate"] = pd.Series(
            scaler.fit_transform(np.expand_dims(cities["estimate"].to_numpy(), 1)).squeeze(),
            index=cities.index
            )
    
    cities = cities.sort_values(by="estimate", ascending=False)
    cities["estimate"] = cities["estimate"].round(3)
    return  json.loads(cities.reset_index().to_json())


'''
Returns GeoDataFrame that contains the estimates of industry development potential for cities of Russian Federation
based on the information about graduates
'''
def estimate_graduates(graduates: DataFrame, cities: DataFrame, edu_groups: DataFrame):
            
        graduates = pd.DataFrame(graduates.groupby(["city", "type", "edu_group_code"])["graduates_forecast"].sum())
        graduates = graduates.join(edu_groups, on=["type", "edu_group_code"]).dropna(subset=["weights"])
        graduates["graduates_weighted"] = graduates["graduates_forecast"] * graduates["weights"]

        graduates_cities_gr = graduates.groupby(["city"])
        graduates_cities = graduates_cities_gr[["graduates_forecast", "graduates_weighted"]].sum().add_suffix("_sum")
        graduates_cities["graduates_forecast"] = graduates_cities_gr.apply(
            lambda x: x["graduates_forecast"].droplevel("city").to_dict()
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

    cv_select = cv[cv["hh_names"].isin(specialities["speciality"])]
    cv_select = cv_select.groupby(["city", "hh_names"])["id_cv"].count().rename("cv_count").reset_index()
    cv_select = cv_select.join(specialities.set_index("speciality")["weights"], on="hh_names")
    cv_select["cv_count_weighted"] = cv_select["cv_count"] * cv_select["weights"]

    cv_cities_gr = cv_select.groupby(["city"])
    cv_cities = cv_cities_gr[["cv_count", "cv_count_weighted"]].sum().add_suffix("_sum")
    cv_cities["cv_count"] = cv_cities_gr[["hh_names", "cv_count"]].apply(
        lambda x: x.set_index("hh_names")["cv_count"].to_dict()
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