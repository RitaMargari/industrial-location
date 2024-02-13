import pandas as pd
import geopandas as gpd
from catboost import CatBoostRegressor
import networkx as nx
import sys
from app.routers_utils import download_intermodal_g_spb

from loguru import logger
logger.remove(0)
logger.add(sys.stderr, format="{level} | {message}")

logger.info('loading data...')

ontology = pd.read_csv("app/data/ontology.csv", index_col=0)
graduates = pd.read_csv("app/data/graduates.csv", index_col=0)
cities = gpd.read_file("app/data/cities.geojson", index_col=0)
vacancy = pd.read_parquet("app/data/vacancy.gzip")
responses = pd.read_parquet("app/data/responses.gzip")
cv = pd.read_parquet("app/data/cv.gzip")
agglomerations = pd.read_parquet(
    "app/data/agglomerations.gzip"
)  # TODO: replace to a new file
DM = pd.read_parquet("app/data/DM.gzip")
model = CatBoostRegressor().load_model(f"app/data/cat_model_dummies_40")
agglomerations = pd.read_parquet("app/data/agglomerations.gzip")
download_intermodal_g_spb()

data_provisions = {
    "perm": {
        "private_car": nx.read_graphml("app/provisions_data/perm_prov/G_drive.graphml"),
        "public_transport": nx.read_graphml(
            "app/provisions_data/perm_prov/G_intermodal.graphml"
        ),
        "gdf_houses": gpd.read_parquet(
            "app/provisions_data/perm_prov/houses_price_demo_prov.parquet"
        ),
    },
    "tomsk": {
        "private_car": nx.read_graphml("app/provisions_data/tomsk_prov/G_intermodal.graphml"),
        "public_transport": nx.read_graphml(
            "app/provisions_data/tomsk_prov/G_intermodal.graphml"
        ),
        "gdf_houses": gpd.read_parquet(
            "app/provisions_data/tomsk_prov/houses_price_demo_prov.parquet"
        ),
    },
    "saint-petersburg": {
        "private_car": nx.read_graphml(
            "app/provisions_data/saint-petersburg_prov/G_drive.graphml"
        ),
        "public_transport": nx.read_graphml(
            "app/provisions_data/saint-petersburg_prov/G_intermodal.graphml"
        ),
        "gdf_houses": gpd.read_parquet(
            "app/provisions_data/saint-petersburg_prov/houses_price_demo_prov.parquet"
        ),
    },
    "shakhty": {
        "private_car": nx.read_graphml("app/provisions_data/shakhty_prov/G_drive.graphml"),
        "public_transport": nx.read_graphml(
            "app/provisions_data/shakhty_prov/G_intermodal.graphml"
        ),
        "gdf_houses": gpd.read_parquet(
            "app/provisions_data/shakhty_prov/houses_price_demo_prov.parquet"
        ),
    },
}


cities = cities.rename(
    columns={
        "vacancies_count_all": "vacancy_count",
        "max_salary_all": "max_salary",
        "median_salary_all": "median_salary",
        "min_salary_all": "min_salary",
    }
)

logger.info('input data successfully loaded')