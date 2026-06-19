import random

# Fake food classes for demo
FOOD_CLASSES = ["rice", "burger", "pizza", "salad", "fries", "pasta"]

def detect_food(image_bytes: bytes):
    """
    Fake AI model:
    - returns random food label
    - returns fake weight estimation
    """

    food = random.choice(FOOD_CLASSES)
    weight = round(random.uniform(0.1, 1.5), 2)

    confidence = round(random.uniform(0.7, 0.99), 2)

    return {
        "food_name": food,
        "weight": weight,
        "confidence": confidence
    }