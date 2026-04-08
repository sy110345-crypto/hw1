from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

from app.model import predict_image
from app.schemas import PredictionResponse

app = FastAPI(
    title="Cat vs Dog Classifier API",
    description="A simple FastAPI server for classifying cats and dogs.",
    version="1.0.0"
)


@app.get("/")
def read_root():
    return {
        "message": "Cat vs Dog Classifier API is running."
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    prediction = predict_image(image)
    return prediction