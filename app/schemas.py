# app/schemas.py 예시
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    gender: str  # <-- 이 필드가 선언되어 있는지 확인하세요!