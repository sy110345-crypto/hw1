import random
from PIL import Image

def preprocess_image(image: Image.Image):
    return image.resize((128, 128))

def predict_image(image: Image.Image):
    processed_image = preprocess_image(image)
    
    # 랜덤하게 결과 생성 (테스트용)
    score = random.random()
    gender_score = random.random()

    if score >= 0.5:
        label = "dog"
        confidence = score
    else:
        label = "cat"
        confidence = 1.0 - score

    # 성별 로직 추가 (이게 있어야 테스트가 통과됨)
    gender = "male" if gender_score >= 0.5 else "female"

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "gender": gender
    }