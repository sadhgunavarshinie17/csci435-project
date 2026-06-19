from fastapi import FastAPI
from .database import Base, engine
from .routers import predict, history

app = FastAPI(title="Food Waste AI System")

Base.metadata.create_all(bind=engine)

app.include_router(predict.router)
app.include_router(history.router)


@app.get("/")
def root():
    return {"status": "running"}