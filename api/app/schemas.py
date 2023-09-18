import app.enums as enums

from pydantic import BaseModel, root_validator
from typing import Optional, List

class OntologyQueryParams:
    def __init__(self,
                 industry_code: Optional[enums.IndustryEnums] = None,
                 ):
        self.industry_code = industry_code

    class Config:
        schema_extra = {
            'example': [
                {'industry_code': 'pharma',}
            ]
        }

class EstimatesIn(BaseModel):
    workforce_type: enums.WorkForce
    specialities: Optional[dict]
    edu_groups: Optional[dict]


    @root_validator(pre=False, skip_on_failure=True)
    def check_user_dict_format(cls, values):
        for var, var_name in zip([eval(values["specialities"]), eval(values["edu_groups"])], ["specialities", "edu_groups"]):
            if var is None: break
            if not all(isinstance(eval(x), int) for x in list(var.keys())): 
                raise TypeError(f"The keys in {var_name} must be inetegers.")
            if not all(isinstance(x, float) or isinstance(x, int) for x in list(var.values())): 
                raise TypeError(f"The values in {var_name} must be floats from 0 to 1.")
            if not all((x >= 0 and x <=1) for x in list(var.values())): 
                raise ValueError(f"The values in {var_name} must be between 0 and 1.")
            
        return values
        

    @root_validator(pre=False, skip_on_failure=True)
    def check_user_workforce_option(cls, values):
        
        if "workforce_type" in values: 
            workforce_type = values["workforce_type"]

            graduates_state = workforce_type == 'all' or workforce_type == 'graduates'
            if graduates_state and values["edu_groups"] is None:
                raise ValueError(f"With workforce_type == '{workforce_type}' edu_groups can't be None")
            
            specialists_state = workforce_type == 'all' or workforce_type == 'specialists'
            if specialists_state and values["specialities"] is None:
                raise ValueError(f"With workforce_type == '{workforce_type}' specialities can't be None")

        return values

    class Config:
        schema_extra = {
            "example": {
                "workforce_type": "all",
                "specialities": {18: 0.5, 1: 1, 14: 1, 0: 0.9, 11: 0.2, 22: 1, 4: 0.1, 10: 0.9, 16: 1, 8: 0.5, 17: 0.6},
                "edu_groups": {20: 0.5, 12: 0.7, 3: 0.1, 21: 0.6, 5: 0.5}
            }
        }