from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..deps import get_db
from .. import crud

router = APIRouter()


@router.get("/history")
def history(db: Session = Depends(get_db)):
    return crud.get_all_records(db)

@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    records = crud.get_all_records(db)

    total_cost = sum(r.cost for r in records)
    total_co2 = sum(r.co2 for r in records)

    return {
        "total_waste_items": len(records),
        "total_cost": round(total_cost, 2),
        "total_co2": round(total_co2, 3)
    }