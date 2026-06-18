from fastapi import FastAPI

app = FastAPI(title="Food Waste AI Backend")


@app.get("/")
def root():
    return {"message": "Food Waste AI Backend Running"}