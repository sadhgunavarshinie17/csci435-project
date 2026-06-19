from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from .database import Base


class FoodWaste(Base):
    __tablename__ = "food_waste"

    id = Column(Integer, primary_key=True, index=True)

    food_name = Column(String)
    weight = Column(Float)
    cost = Column(Float)
    co2 = Column(Float)
    confidence = Column(Float)

    image_path = Column(String)

    timestamp = Column(DateTime, default=datetime.utcnow)