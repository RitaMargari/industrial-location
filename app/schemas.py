import app.enums as enums

from pydantic import BaseModel


# /mobility_analysis/isochrones
class OntologyQueryParams:
    def __init__(self,
                 idustry_code: enums.IndustryEnums,
                 ):
        self.idustry_code = idustry_code

    class Config:
        schema_extra = {
            'example': [
                {'idustry_code': 'pharma',}
            ]
        }
