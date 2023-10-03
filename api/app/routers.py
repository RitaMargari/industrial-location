import faulthandler
import pandas as pd
import geopandas as gpd
import app.func as func
from shapely.geometry import Point
import pyproj
from app.jhm_metric_calcs import jhm_metric

from fastapi import APIRouter, HTTPException, status, Body, Depends
from fastapi.responses import StreamingResponse
from geojson_pydantic import FeatureCollection

from enum import auto
from app import enums, schemas
import networkx as nx
from typing import Optional
import statistics


router = APIRouter()
faulthandler.enable()

ontology = pd.read_csv("app/data/ontology.csv", index_col=0)
cv = pd.read_csv("app/data/cv.csv", index_col=0)
graduates = pd.read_csv("app/data/graduates.csv", index_col=0)
# solve the problem with duplicates
cities = gpd.read_file("app/data/cities.geojson", index_col=0).drop_duplicates(
    "city", keep=False
)

house_prices = gpd.read_parquet("app/data/houses_price_demo.parquet")
G_d = nx.read_graphml("app/data/G_drive.graphml")
# G_t = nx.read_graphml("app/data/G_transport.graphml")
G_t = None


class Tags(str, enums.AutoName):
    def _generate_next_value_(name, start, count, last_values):
        return name

    industry = auto()
    specialities = auto()
    edu_groups = auto()
    estimates = auto()
    jhm_metric = auto()


@router.get("/")
async def read_root():
    return {"Hello": "World"}


@router.get("/ontology/get_industry", response_model=dict, tags=[Tags.industry])
def get_ontology_industry(query_params: schemas.OntologyQueryParams = Depends()):
    result = func.get_ontology_industry(
        ontology=ontology, industry_code=query_params.industry_code
    )
    return result


@router.get("/ontology/get_specialities", response_model=dict, tags=[Tags.specialities])
def get_ontology_specialities(query_params: schemas.OntologyQueryParams = Depends()):
    result = func.get_ontology_specialities(
        ontology=ontology, industry_code=query_params.industry_code
    )
    return result


@router.get("/ontology/get_edu_groups", response_model=dict, tags=[Tags.edu_groups])
def get_ontology_edu_groups(query_params: schemas.OntologyQueryParams = Depends()):
    result = func.get_ontology_edu_groups(
        ontology=ontology, industry_code=query_params.industry_code
    )
    return result


@router.post(
    "/calculation/estimates", response_model=FeatureCollection, tags=[Tags.edu_groups]
)
def get_potential_estimates(query_params: schemas.EstimatesIn):
    result = func.get_potential_estimates(
        ontology=ontology,
        cv=cv,
        graduates=graduates,
        cities=cities,
        workforce_type=query_params.workforce_type,
        specialities=query_params.specialities,
        edu_groups=query_params.edu_groups,
    )
    return result


@router.post("/metrics/get_jhm_metric", response_model=dict, tags=[Tags.jhm_metric])
def get_jhm_metric(query_params: schemas.JhmQueryParams = Depends()):
    DEFAULT_ROOM_AREA = 35
    # DEFAULT_IF_DEBUG_MODE = True

    room_area_m2 = DEFAULT_ROOM_AREA
    filter_coef = query_params.filter_coef
    # debug_mode = query_params.debug_mode or DEFAULT_IF_DEBUG_MODE

    graph_type = {
        "public_transport": G_t,
        "private_car": G_d,
    }

    result = {}
    mean_coef = []

    for element in query_params.worker_and_salary:
        coef_gdf = jhm_metric.main(
            G=graph_type[query_params.transportation_type],
            house_prices=house_prices,
            company_location=query_params.company_location,
            salary=element.salary,
            # TODO: constant value, change to some average value for rent price
            room_area_m2=room_area_m2,
            # debug params
            filter_coef=filter_coef,
            # debug_mode=debug_mode
        )
        
        mean_coef.append(coef_gdf['coef'].mean())
        
        result[element.speciality] = coef_gdf.to_dict(orient="records")

        # .to_dict(orient="records")
    
    return {"mean_coef": statistics.mean(mean_coef), "res": str(result)}
