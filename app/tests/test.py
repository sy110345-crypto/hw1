import pytest
import httpx
import io
from PIL import Image

# 로컬 서버 주소 설정
BASE_URL = "http://localhost:8080"

@pytest.mark.asyncio
async def test_read_root():
    """1. 루트 접속 및 v2.0.0 버전 확인"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/")
    
    assert response.status_code == 200
    data = response.json()
    assert "v2.0.0" in data["message"]
    assert "Gender Identification" in data["features"]

@pytest.mark.asyncio
async def test_health_check():
    """2. 헬스체크 및 모델 타입(Gender 포함) 확인"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "2.0.0"
    assert "Gender" in data["model_type"]

@pytest.mark.asyncio
async def test_predict_with_gender():
    """3. 이미지 업로드 시 성별 정보가 포함되는지 확인"""
    # 테스트용 빨간색 이미지 생성
    img = Image.new('RGB', (128, 128), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    
    # 성별(gender) 필드 검증 (핵심!)
    assert "label" in data
    assert "gender" in data
    assert data["gender"] in ["male", "female"]
    assert isinstance(data["confidence"], float)

@pytest.mark.asyncio
async def test_invalid_file_upload():
    """4. 잘못된 파일 형식 업로드 시 400 에러 확인"""
    files = {'file': ('test.txt', b'not an image', 'text/plain')}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/predict", files=files)
        
    assert response.status_code == 400
    assert "이미지 파일만 업로드 가능합니다" in response.json()["detail"]