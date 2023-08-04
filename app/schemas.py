import app.enums as enums

from pydantic import BaseModel
from typing import Optional, List


# /mobility_analysis/isochrones
class OntologyQueryParams:
    def __init__(self,
                 idustry_code: Optional[enums.IndustryEnums] = None,
                 ):
        self.idustry_code = idustry_code

    class Config:
        schema_extra = {
            'example': [
                {'idustry_code': 'pharma',}
            ]
        }
