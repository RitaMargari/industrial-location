import app.enums as enums
import json

from pydantic import BaseModel, root_validator, conint
from geojson_pydantic import FeatureCollection
from typing import Optional, List, Dict


class OntologyQueryParams:
    def __init__(
        self,
        industry_code: Optional[enums.IndustryEnums] = None,
    ):
        self.industry_code = industry_code

    class Config:
        schema_extra = {
            "example": [
                {
                    "industry_code": "pharma",
                }
            ]
        }


class EstimatesIn(BaseModel):
    workforce_type: enums.WorkForce
    specialities: Optional[dict]
    edu_groups: Optional[dict]
    links_output: bool = True


    @root_validator(pre=False, skip_on_failure=True)
    def check_user_dict_format(cls, values):
        for var, var_name in zip(
            [values["specialities"], values["edu_groups"]],
            ["specialities", "edu_groups"],
        ):
            if var is None:
                break
            if not all(isinstance(eval(x), int) for x in list(var.keys())):
                raise TypeError(f"The keys in {var_name} must be inetegers.")
            if not all(
                isinstance(x, float) or isinstance(x, int) for x in list(var.values())
            ):
                raise TypeError(f"The values in {var_name} must be floats from 0 to 1.")
            if not all((x >= 0 and x <= 1) for x in list(var.values())):
                raise ValueError(f"The values in {var_name} must be between 0 and 1.")

        return values

    @root_validator(pre=False, skip_on_failure=True)
    def check_user_workforce_option(cls, values):
        if "workforce_type" in values:
            workforce_type = values["workforce_type"]

            graduates_state = workforce_type == "all" or workforce_type == "graduates"
            if graduates_state and values["edu_groups"] is None:
                raise ValueError(
                    f"With workforce_type == '{workforce_type}' edu_groups can't be None"
                )

            specialists_state = (
                workforce_type == "all" or workforce_type == "specialists"
            )
            if specialists_state and values["specialities"] is None:
                raise ValueError(
                    f"With workforce_type == '{workforce_type}' specialities can't be None"
                )

        return values

    class Config:
        schema_extra = {
            "example": {
                "workforce_type": "all",
                "specialities": {
                    18: 0.5,
                    1: 1,
                    14: 1,
                    0: 0.9,
                    11: 0.2,
                    22: 1,
                    4: 0.1,
                    10: 0.9,
                    16: 1,
                    8: 0.5,
                    17: 0.6,
                },
                "edu_groups": {20: 0.5, 12: 0.7, 3: 0.1, 21: 0.6, 5: 0.5},
            }
        }


class EstimatesOut(BaseModel):
    estimates: FeatureCollection    
    links: Optional[FeatureCollection]


class ConnectionsIn(BaseModel):
    city: str

    class Config:
        schema_extra = {
            "example": {
                "city": "Тюменская область, Тюмень"
            }
        }


class ConnectionsOut(BaseModel):
    migration_link: FeatureCollection
    agglomeration_links: FeatureCollection
    agglomeration_nodes: FeatureCollection



with open("app/data/estimates_sample.geojson") as f:
    estimates_sample = json.load(f)

# TODO: check the list of features and city names in estimates_table
class PredictionIn(BaseModel):
    city_name: str
    estimates_table: FeatureCollection

    class Config:
        schema_extra = {
            "example": {
                "city_name": "Тюменская область, Тюмень",
                "estimates_table": estimates_sample
            }
        }


class PredictionOut(BaseModel):
    city_features: FeatureCollection
    update_dict: dict
    new_links: FeatureCollection


class Workers(BaseModel):
    speciality: str
    salary: conint(ge=10_000, lt=500_000)


class JhmQueryParams(BaseModel):
    worker_and_salary: List[Workers]
    transportation_type: enums.Transportation
    company_location: Dict[str, float]
    cell_size_meters: conint(ge=300, lt=5_000) = 500

    class Config:
        schema_extra = {
            "example": {
                "company_location": {"lon": 59.860510, "lat": 30.211518},
                "transportation_type": "private_car",
                "worker_and_salary": [
                    {"speciality": "worker_1", "salary": 45000},
                    {"speciality": "worker_2", "salary": 70000},
                ],
                "cell_size_meters": 500
            }
        }
