from typing import Dict
import osmnx as ox
from fastapi import HTTPException
from shapely.geometry import Point
from app.constants import city_osmid_dict, LEAST_POSSIBLE_SALARY
import os
import gdown


class UserInputError(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def raise_http_400_exception(detail):
    raise HTTPException(
        status_code=400,
        detail=detail,
    )


# Create the Overpass query to get the boundary of the city
def validate_company_location(coords_dict: Dict[str, float], city_name: str):
    boundary_gdf = ox.geocode_to_gdf(
        city_osmid_dict[city_name], by_osmid=True, which_result=1
    )
    # Check if the coordinates are within the city boundary
    if not boundary_gdf.intersects(
        Point(coords_dict["lon"], coords_dict["lat"])
    ).item():
        # Raise a custom exception to indicate user's fault
        detail = "Invalid coordinates provided. Please provide valid x and y coordinates within the city boundary."
        raise_http_400_exception(detail)


def validate_workers_salary(worker_and_salary):
    detail = "Invalid salary provided. Please provide valid salary value >10_000 rub."
    for worker in worker_and_salary:
        if worker.salary < LEAST_POSSIBLE_SALARY:
            raise_http_400_exception(detail)


def download_intermodal_g_spb():
    file_path = "app/provisions_data/saint-petersburg_prov/G_intermodal.graphml"
    
    if not os.path.isfile(file_path):
        print('DOWNLOADING SPB INTERMODAL G')
        url = "https://drive.google.com/uc?export=download&id=1vGGh1s7EIjxgGEF_0Dylb6XNBnCpApEQ"
        gdown.download(url, file_path, quiet=False)