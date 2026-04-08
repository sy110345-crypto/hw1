import pytest
import httpx
import io
from PIL import Image

BASE_URL = "http://localhost:8080"

@pytest.mark.asyncio
async def test_read_root():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert "v2.0.0" in data["message"]
    assert "Gender Identification" in data["features"]

@pytest.mark.asyncio
async def test_health_check():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "2.0.0"
    assert "Gender" in data["model_type"]

@pytest.mark.asyncio
async def test_predict_with_gender():
    # 1. 메모리 상에서 테스트용 이미지 생성
    img = Image.new('RGB', (128, 128), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # 2. 파일 전송 준비
    files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}

    async with httpx.AsyncClient() as client:
        # timeout을 줘서 서버 응답이 늦을 때를 대비합니다.
        response = await client.post(f"{BASE_URL}/predict", files=files, timeout=10.0)

    assert response.status_code == 200
    data = response.json()
    
    # 3. 데이터 검증
    assert "label" in data
    assert data["label"] in ["dog", "cat"]
    assert "gender" in data
    assert data["gender"] in ["male", "female"]
    assert isinstance(data["confidence"], (float, int))

@pytest.mark.asyncio
async def test_invalid_file_upload():
    # 이미지가 아닌 텍스트 파일 전송 테스트
    files = {'file': ('test.txt', b'not an image', 'text/plain')}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/predict", files=files)
        
    assert response.status_code == 400
    # 서버 코드의 에러 메시지와 일치하는지 확인
    assert "이미지 파일만 업로드 가능합니다" in response.json()["detail"]