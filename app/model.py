from PIL import Image
import random


def preprocess_image(image: Image.Image):
    resized_image = image.resize((128, 128))
    return resized_image


def predict_image(image: Image.Image):
    processed_image = preprocess_image(image)

    width, height = processed_image.size
    _ = (width, height)

    score = random.random()

    if score >= 0.5:
        label = "dog"
        confidence = score
    else:
        label = "cat"
        confidence = 1.0 - score

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }