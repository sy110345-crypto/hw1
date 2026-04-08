import pytest
import httpx
import os

# 로컬 서버 주소
BASE_URL = "http://localhost:8080"

# test.py와 같은 위치에 있는 dog.jpg 경로 설정
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(CUR_DIR, "dog.jpg")

@pytest.mark.asyncio
async def test_predict_with_dog_image():
    """실제 tests 폴더 안의 dog.jpg로 테스트"""
    
    # 1. 파일 존재 확인
    if not os.path.exists(IMAGE_PATH):
        pytest.fail(f"파일이 없습니다! 경로 확인: {IMAGE_PATH}")

    # 2. 이미지 전송 및 요청
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
        with open(IMAGE_PATH, "rb") as f:
            files = {'file': ('dog.jpg', f, 'image/jpeg')}
            response = await client.post("/predict", files=files)

    # 3. 검증
    assert response.status_code == 200
    data = response.json()
    
    # 서버 결과값 확인
    assert "label" in data
    assert "gender" in data
    assert data["gender"] in ["male", "female"]
    assert isinstance(data["confidence"], (float, int))
    
    print(f"\n[결과] Label: {data['label']}, Gender: {data['gender']}")

@pytest.mark.asyncio
async def test_read_root():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        response = await client.get("/")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_health_check():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        response = await client.get("/health")
    assert response.status_code == 200