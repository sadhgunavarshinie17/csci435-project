from sqlalchemy.orm import Session
from . import models


def create_record(db: Session, food: str, weight: float, cost: float, co2: float):
    db_record = models.FoodWaste(
        food=food,
        weight=weight,
        cost=cost,
        co2=co2
    )

    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return db_record