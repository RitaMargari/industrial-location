import os

import pandas as pd
from loguru import logger

from provisio import get_service_provision, demands_from_buildings_by_normative
from dongraphio.base_models import BuildsMatrixer
from app.db.data_service import DataGetter


def get_provision(
    city_name: str, service_type_name: str, valuation_type_name: str, year: int
):

    city_crs = DataGetter.get_city_utm_crs(city_name=city_name)
    services = DataGetter.get_services(
        service_name=service_type_name, city_name=city_name
    ).to_crs(city_crs)
    buildings = DataGetter.get_buildings(
        service_name=service_type_name, year=year, city_name=city_name
    ).to_crs(city_crs)
    service_normative_value = DataGetter.get_service_normative(
        service_name=service_type_name
    )

    matrix = DataGetter.get_matrix_fileserver(
        city_name=city_name, service_name=service_type_name, folder="matrix/"
    )

    if not isinstance(matrix, pd.DataFrame):
        graph_nx = DataGetter.get_nx_graph_mongo(
            city_name=city_name, graph_type="intermodal_graph"
        )
        matrix = BuildsMatrixer(
            buildings_from=buildings,
            services_to=services,
            nx_intermodal_graph=graph_nx,
            city_crs=city_crs,
            weight="time_min",
        ).get_adjacency_matrix()

    logger.info("Starting provision calculations")
    res = get_service_provision(
        services=services,
        adjacency_matrix=matrix,
        demanded_buildings=buildings,
        threshold=service_normative_value,
    )

    DataGetter.post(
        initial_file_df=matrix,
        filename="matrix.pkl",
        folder="matrix",
        url=DataGetter.fileserver_upload_address,
    )
    DataGetter.post(
        initial_file_df=res[0],
        filename="builds.pkl",
        folder="calculated_provision",
        url=DataGetter.fileserver_upload_address,
    )
    DataGetter.post(
        initial_file_df=res[1],
        filename="servs.pkl",
        folder="calculated_provision",
        url=DataGetter.fileserver_upload_address,
    )
    DataGetter.post(
        initial_file_df=res[2],
        filename="links.pkl",
        folder="calculated_provision",
        url=DataGetter.fileserver_upload_address,
    )

    return res
