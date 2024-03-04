# app/routers/spatial_data.py
import os

from fastapi import APIRouter
from app.scripts.calculate_provision import get_provision
from app.db.data_service import DataGetter
from app.api.schemes.schemes import ProvisionGetProvisionIn

router = APIRouter()


@router.post("/get_provision/")
def get_spatial_data_cached(query_params: ProvisionGetProvisionIn):

    pre_calc = os.path.isfile(DataGetter.fileserver_upload_address + "servs.pkl")

    if not pre_calc:
        prov = get_provision(
            city_name=query_params.city,
            service_type_name=query_params.service_types,
            valuation_type_name=query_params.valuation_type,
            year=query_params.year,
        )
    else:
        pre_calc_builds = DataGetter.get_pickle(
            filename="builds.pkl",
            folder="calculated_provision",
            url=DataGetter.fileserver_upload_address,
        )
        pre_calc_services = DataGetter.get_pickle(
            filename="servs.pkl",
            folder="calculated_provision",
            url=DataGetter.fileserver_upload_address,
        )
        pre_calc_links = DataGetter.get_pickle(
            filename="links.pkl",
            folder="calculated_provision",
            url=DataGetter.fileserver_upload_address,
        )

        prov = [pre_calc_builds, pre_calc_services, pre_calc_links]

    return {
        "houses": eval(
            prov[0]
            .to_json()
            .replace("true", "True")
            .replace("null", "None")
            .replace("false", "False")
        ),
        "services": eval(
            prov[1]
            .to_json()
            .replace("true", "True")
            .replace("null", "None")
            .replace("false", "False")
        ),
        "provisions": {
            query_params.service_types: eval(prov[2]
            .to_json()
            .replace("true", "True")
            .replace("null", "None")
            .replace("false", "False")
            )
        },
    }
