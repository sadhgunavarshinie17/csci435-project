from app.services.constants import FOOD_DB


def calculate_impact(food_name: str, weight: float):
    food_name = food_name.lower()

    if food_name not in FOOD_DB:
        return {
            "cost": 0,
            "co2": 0
        }

    data = FOOD_DB[food_name]

    cost = weight * data["price_per_kg"]
    co2 = weight * data["co2_per_kg"]

    return {
        "cost": round(cost, 2),
        "co2": round(co2, 3)
    }