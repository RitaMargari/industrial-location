import faulthandler
import app.func as func

from fastapi import APIRouter, HTTPException, status, Body, Depends
from fastapi.responses import StreamingResponse
from geojson_pydantic import FeatureCollection

from enum import auto
from app import enums, schemas
from typing import Optional


router = APIRouter()
faulthandler.enable()

class Tags(str, enums.AutoName):
    def _generate_next_value_(name, start, count, last_values):
        return name

    ontology = auto()


@router.get("/")
async def read_root():
    return {"Hello": "World"}


@router.get(
    '/ontology/get_ontology',
    response_model=dict, tags=[Tags.ontology]
)
def get_ontology(query_params: schemas.OntologyQueryParams = Depends()):
    result = func.get_ontology(idustry_code=query_params.idustry_code)
    print(type(result))
    return result