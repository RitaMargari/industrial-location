# app/services/data_service.py

import os
import io

import geopandas as gpd
import pandas as pd
from networkx.classes.multidigraph import MultiDiGraph
import networkx as nx
import requests
import psycopg2 as pg
from loguru import logger


class DataGetter:

    GENERAL_REQ_TIMEOUT = 120

    postgis_engine = pg.connect(
        "dbname='city_db_final' user='postgres' host='10.32.1.107' port='5432' password='postgres'"
    )
    mongo_address = "http://10.32.1.65:5100"
    fileserver_upload_address = "http://172.17.0.1:5000/upload"
    fileserver_get_address = "http://172.17.0.1:5000/"

    @staticmethod
    def get_nx_graph_mongo(city_name: str, graph_type: str, node_type: type=int) -> MultiDiGraph:

        logger.info('loading graph')

        file_name = city_name.lower() + "_" + graph_type
        response = requests.get(
            DataGetter.mongo_address + "/uploads/city_graphs/" + file_name,
            timeout=DataGetter.GENERAL_REQ_TIMEOUT,
        )
        if response.ok:
            graph = nx.readwrite.graphml.parse_graphml(
                response.text, node_type=node_type
            )
            logger.info(f"SUCCESS: Graph loaded: {graph_type, city_name}")
            return graph
        else:
            logger.warning(f"No existing graph: {graph_type, city_name}")
            return None

    @staticmethod
    def get_buildings(service_name: str, year: int, city_name: int) -> gpd.GeoDataFrame:

        logger.info('loading buildings')

        query_buildings = (
            f"SELECT pr.building_id ,pr.{service_name}_service_demand_value_model as demand, b.geometry "
            f"FROM provision.buildings_load_future pr "
            f"JOIN all_buildings b on b.building_id = pr.building_id "
            f"WHERE pr.year = {year} and b.city_id=(select id from cities where code='{city_name}') "
            f"AND b.is_living IS TRUE AND b.population_balanced > 0"
        )

        buildings = (
            gpd.read_postgis(query_buildings, con=DataGetter.postgis_engine, geom_col='geometry')
        )

        assert buildings.shape[0] > 0

        logger.info('SUCCESS: buildings loaded')

        return buildings
    
    @staticmethod
    def get_city_utm_crs(city_name: str) -> int:
        
        logger.info('loading city utm crs')

        query_crs = (
            f"SELECT local_crs "
            f"FROM cities "
            f"WHERE code = '{city_name}'"
        )

        utm_crs = pd.read_sql(query_crs, con=DataGetter.postgis_engine).loc[:, 'local_crs'].item()

        assert utm_crs > 0

        logger.info('SUCCESS: city utm crs loaded')

        return utm_crs
    
    @staticmethod
    def get_service_normative(service_name:str) -> int:

        logger.info('loading service normative')

        query_service_normative = (
            f"SELECT public_transport_time_normative, walking_radius_normative "
            f"FROM city_service_types "
            f"WHERE code = '{service_name}'"
        )

        both_normatives = pd.read_sql(query_service_normative, con=DataGetter.postgis_engine).fillna(0)
        
        # since there either one or other
        normative = both_normatives['public_transport_time_normative'] + both_normatives['walking_radius_normative']
        normative = normative.item()

        assert normative > 0

        logger.info('SUCCESS: service normative loaded')        

        return normative



    @staticmethod
    def get_services(service_name: str, city_name: int):

        logger.info('loading services')

        query_services = (
            f"SELECT functional_object_id,capacity,geometry, city_service_type_code "
            f"FROM public.all_services "
            f"WHERE city_service_type_code='{service_name}' and city_id=(select id from cities where code='{city_name}')"
        )
        services = gpd.read_postgis(query_services, con=DataGetter.postgis_engine, geom_col='geometry')

        assert services.shape[0] > 0
        logger.info('SUCCESS: services loaded')

        return services

    @staticmethod
    def get_nk_graph():
        pass

    @staticmethod
    def get_matrix_fileserver(city_name: int, service_name: str, folder: str):
        filename = f"{city_name}_{service_name}.pkl"

        logger.info(f'\n\nGettimg matrix: {DataGetter.fileserver_get_address}{folder}{filename}\n\n')
        
        m = DataGetter.get_pickle(
            filename=filename,
            folder=folder,
            url=DataGetter.fileserver_get_address,
        )

        m.index = m.index.astype(int)
        m.columns = m.columns.map(int)

        return m

    @staticmethod
    def get_pre_calculated_provision_fileserver(
        service_name: str, year: int, city_name: int, valuation_type: str
    ):
        filename = f"{city_name}_{year}_{service_name}_{valuation_type}.pkl"
        return DataGetter.get_pickle(
            filename=filename,
            folder="provision_data/",
            url=DataGetter.fileserver_get_address,
        )

    @staticmethod
    def post(initial_file_df:pd.DataFrame, filename, folder, url) -> None:
        data_bytes_io = io.BytesIO()
        initial_file_df.to_pickle(data_bytes_io)
        file_to_post = data_bytes_io.getvalue()

        files = {"files": (filename, file_to_post)}
        data = {"newdir": folder}

        logger.info(f"{url}{filename}{data}")

        response = requests.post(
            url, files=files, data=data, timeout=DataGetter.GENERAL_REQ_TIMEOUT
        )

        logger.info(f"{response.status_code}, {filename}, {response}")

    @staticmethod
    def get_pickle(filename, folder, url):

        print(url + folder + filename)
        
        r = requests.get(
            url + folder + filename, timeout=DataGetter.GENERAL_REQ_TIMEOUT
        )
        received_bytes_io = io.BytesIO(r.content)
        received_df = pd.read_pickle(received_bytes_io)
        logger.info(f"File successfully loaded: {filename}")
        return received_df
