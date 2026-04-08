from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

# 기존 모델 로직에 성별 추론이 추가되었다고 가정합니다.
from app.model import predict_image 
from app.schemas import PredictionResponse

app = FastAPI(
    title="Cat & Dog Gender Classifier API",
    description="개/고양이의 종과 성별을 모두 식별하는 업그레이드된 MLOps API입니다.",
    version="2.0.0" # 성별 식별 기능 추가로 버전 업
)

@app.get("/")
def read_root():
    return {
        "message": "Cat & Dog Gender Classifier API (v2.0.0) is running.",
        "features": ["Species Classification", "Gender Identification"]
    }

@app.get("/health")
def health_check():
    # 배포 성공 확인을 위해 버전을 명시합니다.
    return {
        "status": "ok",
        "version": "2.0.0",
        "model_type": "Multi-task (Species + Gender)"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # 1. 파일 형식 검사
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    try:
        # 2. 이미지 로드
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="유효하지 않은 이미지 파일입니다.")

    # 3. 통합 모델 추론 (Species: Cat/Dog, Gender: Male/Female)
    # predict_image 내부 로직이 성별까지 반환하도록 수정되어 있어야 합니다.
    prediction = predict_image(image)
    
    return prediction