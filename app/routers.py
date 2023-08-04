import faulthandler
import pandas as pd
import geopandas as gpd
import app.func as func

from fastapi import APIRouter, HTTPException, status, Body, Depends
from fastapi.responses import StreamingResponse
from geojson_pydantic import FeatureCollection

from enum import auto
from app import enums, schemas
from typing import Optional


router = APIRouter()
faulthandler.enable()

ontology = pd.read_csv("app/data/ontology.csv", index_col=0)
cv = pd.read_csv("app/data/cv.csv", index_col=0)
graduates = pd.read_csv("app/data/graduates.csv", index_col=0)
# solve the problem with duplicates
cities = gpd.read_file("app/data/cities.geojson", index_col=0).drop_duplicates("city", keep=False)


class Tags(str, enums.AutoName):
    def _generate_next_value_(name, start, count, last_values):
        return name

    industry = auto()
    specialities = auto()
    edu_groups = auto()


@router.get("/")
async def read_root():
    return {"Hello": "World"}


@router.get(
    '/ontology/get_industry',
    response_model=dict, tags=[Tags.industry]
)
def get_ontology_industry(query_params: schemas.OntologyQueryParams = Depends()):
    result = func.get_ontology_industry(ontology=ontology, idustry_code=query_params.idustry_code)
    return result

@router.get(
    '/ontology/get_specialities',
    response_model=dict, tags=[Tags.specialities]
)
def get_ontology_specialities(query_params: schemas.OntologyQueryParams = Depends()):
    result = func.get_ontology_specialities(ontology=ontology, idustry_code=query_params.idustry_code)
    return result

@router.get(
    '/ontology/get_edu_groups',
    response_model=dict, tags=[Tags.edu_groups]
)
def get_ontology_specialities(query_params: schemas.OntologyQueryParams = Depends()):
    result = func.get_ontology_edu_groups(ontology=ontology, idustry_code=query_params.idustry_code)
    return result