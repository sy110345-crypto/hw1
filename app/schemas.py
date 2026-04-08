# app/schemas.py 예시
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    gender: str  # 이 줄이 꼭 있어야 합니다!