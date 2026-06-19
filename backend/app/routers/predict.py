from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session

from ..deps import get_db
from ..ai.detector import detect_food
from ..services.calculator import calculate_impact
from .. import crud

router = APIRouter()


@router.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    image_bytes = await file.read()

    # 1. AI detection (fake for now)
    result = detect_food(image_bytes)

    food_name = result["food_name"]
    weight = result["weight"]

    # 2. compute impact
    impact = calculate_impact(food_name, weight)

    # 3. SAVE TO DATABASE
    record = crud.create_record(
        db=db,
        food=food_name,
        weight=weight,
        cost=impact["cost"],
        co2=impact["co2"]
    )

    # 4. return response
    return {
        "id": record.id,
        "food": food_name,
        "weight": weight,
        "confidence": result["confidence"],
        "cost": impact["cost"],
        "co2": impact["co2"]
    }