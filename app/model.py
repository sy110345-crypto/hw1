def predict_image(image: Image.Image):
    processed_image = preprocess_image(image)
    width, height = processed_image.size

    score = random.random()
    # 성별을 위한 새로운 랜덤값
    gender_score = random.random()

    if score >= 0.5:
        label = "dog"
        confidence = score
    else:
        label = "cat"
        confidence = 1.0 - score

    # 성별 결정 로직 추가
    gender = "male" if gender_score >= 0.5 else "female"

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "gender": gender  # <-- 이 부분이 추가되어야 테스트가 통과됩니다!
    }