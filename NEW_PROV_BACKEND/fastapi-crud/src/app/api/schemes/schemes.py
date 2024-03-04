from pydantic import BaseModel

class ProvisionInBase(BaseModel):
    """Базовый класс схемы входных параметров для обеспеченности. """
    city: str
    service_types: str
    valuation_type: str
    year: int
    # user_selection_zone: Optional[Polygon] = None
    # service_impotancy: Optional[list] = None
    # valuation_type: str = 'gravity'



class ProvisionGetProvisionIn(ProvisionInBase):
    class Config:
        schema_extra = {
            "example": {
                "city": "tara",
                "service_types": "kindergartens",
                "valuation_type": "normative",
                "year": 2023,
            }
        }