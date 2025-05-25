from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import io
from inference.pointnet import load_pointnet, predict
from inference.landmarker import load_mediapipe, get_landmark

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 데이터 디렉토리 생성
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POINTNET_MODEL_PATH = Path("model/best_model.pth")  # 저장된 모델 경로
MEDIAPIPE_MODEL_PATH = Path("model/face_landmarker.task")

# 모델 로드
try:
    POINTNET_MODEL = load_pointnet(POINTNET_MODEL_PATH, DEVICE)
    MEDIAPIPE_MODEL = load_mediapipe(MEDIAPIPE_MODEL_PATH)
except Exception as e:
    print(f"모델 로드 중 오류 발생: {str(e)}")
    raise

# API 라우팅
@app.get("/")
def read_root():
    return {"message": "AI backend 서버입니다."}

@app.post("/image")
async def image_upload(file: UploadFile = File(...)):
    # 파일이 이미지인지 확인
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    try:
        # 이미지 데이터 읽기
        contents = await file.read()
        
        # 파일 저장
        file_path = data_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(contents)
        
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(contents),
            "message": "이미지 업로드 성공",
            "saved_path": str(file_path)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/face")
async def predict_face(file: UploadFile = File(...)):
    try:
        # 이미지 데이터 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # 랜드마크 추출
        landmarks = get_landmark(MEDIAPIPE_MODEL, image_np)
        
        # 예측
        result = predict(POINTNET_MODEL, landmarks, DEVICE)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")