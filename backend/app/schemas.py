from pydantic import BaseModel


class PredictionResponse(BaseModel):
    food_name: str
    weight: float
    cost: float
    co2: float
    confidence: float
    image_path: str