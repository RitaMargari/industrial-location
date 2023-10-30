import faulthandler
import pandas as pd
import geopandas as gpd
import app.func as func
from app.jhm_metric_calcs import main
from app.jhm_metric_calcs.utils import read_intermodal_G_from_gdrive

from fastapi import APIRouter, HTTPException, status, Body, Depends
from fastapi.responses import StreamingResponse
from geojson_pydantic import FeatureCollection

from enum import auto
from app import enums, schemas
import networkx as nx
from collections import defaultdict
import numpy as np


router = APIRouter()
faulthandler.enable()

ontology = pd.read_csv("app/data/ontology.csv", index_col=0)
graduates = pd.read_csv("app/data/graduates.csv", index_col=0)
cities = gpd.read_file("app/data/cities.geojson", index_col=0)
vacancy = pd.read_parquet("app/data/vacancy.gzip")
responses = pd.read_parquet("app/data/responses.gzip")
cv = pd.read_parquet("app/data/cv.gzip")
agglomerations = pd.read_parquet("app/data/agglomerations.gzip")

gdf_houses = gpd.read_parquet("app/data/houses_price_demo.parquet")
G_drive = nx.read_graphml("app/data/G_drive.graphml")
G_intermodal = read_intermodal_G_from_gdrive()


class Tags(str, enums.AutoName):
    def _generate_next_value_(name, start, count, last_values):
        return name

    industry = auto()
    specialities = auto()
    edu_groups = auto()
    estimates = auto()
    connections = auto()
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
    '/calculation/estimates',
    response_model=schemas.EstimatesOut, tags=[Tags.estimates]
)
def get_potential_estimates(query_params: schemas.EstimatesIn):
    result = func.get_potential_estimates(
        ontology=ontology, cv=cv, graduates=graduates, cities=cities, responses=responses, vacancy=vacancy,
        workforce_type = query_params.workforce_type, specialities=query_params.specialities, 
        edu_groups=query_params.edu_groups, links_output=query_params.links_output
        )
    return result

@router.post(
    '/calculation/connection',
    response_model=schemas.ConnectionsOut, tags=[Tags.connections]
)
def get_potential_estimates(query_params: schemas.ConnectionsIn):
    migration = func.get_city_migration_links(responses, cities, query_params.specialists, query_params.city)
    agglomeration = func.get_city_agglomeration_links(agglomerations, cities, query_params.city)

    return {
        "migration_link": migration, 
        "agglomeration_links": agglomeration["links"], 
        "agglomeration_nodes": agglomeration["nodes"]
        }

@router.post("/metrics/get_jhm_metric", response_model=dict, tags=[Tags.jhm_metric])
def get_jhm_metric(query_params: schemas.JhmQueryParams):
    graph_type = {
        "public_transport": G_intermodal,
        "private_car": G_drive,
    }

    return main(
        gdf_houses,
        query_params.worker_and_salary,
        graph_type[query_params.transportation_type],
        query_params.company_location,
    )