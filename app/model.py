import random
from PIL import Image

def preprocess_image(image: Image.Image):
    return image.resize((128, 128))

def predict_image(image: Image.Image):
    processed_image = preprocess_image(image)
    
    score = random.random()
    gender_score = random.random()

    if score >= 0.5:
        label = "dog"
        confidence = score
    else:
        label = "cat"
        confidence = 1.0 - score

    gender = "male" if gender_score >= 0.5 else "female"

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "gender": gender
    }