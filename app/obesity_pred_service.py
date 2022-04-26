# Run this script with cmd : python -m uvicorn obesity_pred_service:app 
from typing import Optional
from pydantic import BaseModel, Json
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from model import ObesityPredictor


class ObesityPredRequest(BaseModel):
    data: Json
    # api_key: str


class ObesityPrediction(BaseModel):
    result: str
    probability: Optional[float]


classif = ObesityPredictor()
app = FastAPI()

@app.get("/", tags=["health"])
def health_check():
    return 200

@app.post("/predict", response_model=ObesityPrediction)
def predict_obesity(req: ObesityPredRequest) -> ObesityPrediction:
    print(req.data)
    data = jsonable_encoder(req.data)
    print(data)
    result = classif.predict(data)
    return result
