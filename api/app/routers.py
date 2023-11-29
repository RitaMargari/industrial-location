import faulthandler
import pandas as pd
import geopandas as gpd
import json
import app.func as func
from app.jhm_metric_calcs.jhm_metric import main
from app.routers_utils import validate_company_location, validate_workers_salary, download_intermodal_g_spb

from fastapi import APIRouter,  Depends
from catboost import CatBoostRegressor

from enum import auto
from app import enums, schemas
import networkx as nx


router = APIRouter()
faulthandler.enable()

ontology = pd.read_csv("app/data/ontology.csv", index_col=0)
graduates = pd.read_csv("app/data/graduates.csv", index_col=0)
cities = gpd.read_file("app/data/cities.geojson", index_col=0)
vacancy = pd.read_parquet("app/data/vacancy.gzip")
responses = pd.read_parquet("app/data/responses.gzip")
cv = pd.read_parquet("app/data/cv.gzip")
agglomerations = pd.read_parquet("app/data/agglomerations.gzip") # TODO: replace to a new file
DM = pd.read_parquet("app/data/DM.gzip")
model = CatBoostRegressor().load_model(f"app/data/cat_model_dummies_40")
agglomerations = pd.read_parquet("app/data/agglomerations.gzip")
download_intermodal_g_spb()



class Tags(str, enums.AutoName):
    def _generate_next_value_(name, start, count, last_values):
        return name

    industry = auto()
    specialities = auto()
    edu_groups = auto()
    estimates = auto()
    connections = auto()
    prediction = auto()
    jhm_metric = auto()


@router.get("/")
async def read_root():
    return {"Hello": "World"}


@router.get("/ontology/get_industry", response_model=dict, tags=[Tags.industry])
def get_ontology_industry(query_params: schemas.OntologyQueryParams = Depends()):
    result = func.get_ontology_industry(
        ontology=ontology, industry_code=query_params.industry_code
    )
    return json.loads(result.to_json())


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
    "/calculation/estimates", response_model=schemas.EstimatesOut, tags=[Tags.estimates]
)
def get_potential_estimates(query_params: schemas.EstimatesIn):
    result = func.get_potential_estimates(
        ontology=ontology, cv=cv, graduates=graduates, cities=cities, responses=responses, vacancy=vacancy,
        workforce_type = query_params.workforce_type, specialities=query_params.specialities, 
        edu_groups=query_params.edu_groups, links_output=query_params.links_output
        )
    return {
        'estimates': json.loads(result['estimates'].to_json()),
        'links': json.loads(result['links'].to_json()) if not None else links_json
        }

@router.post(
    "/calculation/connection",
    response_model=schemas.ConnectionsOut,
    tags=[Tags.connections],
)
def get_connections(query_params: schemas.ConnectionsIn):
    migration = func.get_city_migration_links(responses, cities, query_params.city)
    agglomeration = func.get_city_agglomeration_links(agglomerations, cities, query_params.city)

    return {
        "migration_link": json.loads(migration.to_json()), 
        "agglomeration_links": json.loads(agglomeration["links"].to_json()), 
        "agglomeration_nodes": json.loads(agglomeration["nodes"].to_json())
        }


@router.post(
    '/calculation/prediction',
    response_model=schemas.PredictionOut, tags=[Tags.prediction]
)
def predict_migration(query_params: schemas.PredictionIn):

    estimates_table = query_params.estimates_table.dict()
    estimates_table = gpd.GeoDataFrame.from_features(estimates_table['features'])
    result = func.predict_migration(
        cities, responses, DM, model, estimates_table, query_params.city_name
        )

    return {
        "city_features": json.loads(result['city_features'].to_json()), 
        "update_dict": result["update_dict"], 
        "new_links": json.loads(result["new_links"].to_json())
        }


@router.post("/metrics/get_jhm_metric", response_model=dict, tags=[Tags.jhm_metric])
def get_jhm_metric(query_params: schemas.JhmQueryParams):

    validate_company_location(query_params.company_location, query_params.city_name.value)
    validate_workers_salary(query_params.worker_and_salary)

    path = f"app/provisions_data/{query_params.city_name.value}_prov/"
    gdf_houses = gpd.read_parquet(path + "houses_price_demo_prov.parquet")
    graph_type = {
        "public_transport": nx.read_graphml(path + "G_intermodal.graphml"),
        # "private_car":  nx.read_graphml(path + "G_drive.graphml"),
    }

    result = main(
        gdf_houses=gdf_houses,
        worker_and_salary=query_params.worker_and_salary,
        graph=graph_type[query_params.transportation_type],
        company_location=query_params.company_location,
        cell_size_meters=query_params.cell_size_meters,
    )

    return result
