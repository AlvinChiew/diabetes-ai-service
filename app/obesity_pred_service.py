# Run this script with cmd : API_KEY=string python -m uvicorn obesity_pred_service:app 
from typing import Optional
from pydantic import BaseModel, Json
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder

from model import ObesityPredictor
from api_auth import auth_api_key

class ObesityPredRequest(BaseModel):
    data: Json
    api_key: str


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
    if not auth_api_key(req.api_key):
        raise HTTPException(status_code=401, detail="ERROR: invalid API key.")
    data = jsonable_encoder(req.data)
    return {"result": classif.predict(data)}  # TBD: add prediction probability 
