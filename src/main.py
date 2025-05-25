from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import io
from inference.pointnet import load_pointnet, predict
from inference.landmarker import load_mediapipe, get_landmark

# Pydantic 모델 정의
class ImageUploadResponse(BaseModel):
    filename: str
    content_type: str
    size: int
    message: str
    saved_path: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    landmarks_count: int
    processing_time: float

class ErrorResponse(BaseModel):
    detail: str

# FastAPI 앱 생성 (메타데이터 추가)
app = FastAPI(
    title="FocusSu AI API",
    description="얼굴 이미지 분석 및 예측을 위한 AI 백엔드 API",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI 경로
    redoc_url="/redoc",  # ReDoc 경로
    openapi_url="/openapi.json"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 프로젝트 루트 디렉토리 설정 (src 디렉토리의 상위 디렉토리)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"

# 데이터 디렉토리 생성
DATA_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POINTNET_MODEL_PATH = MODEL_DIR / "best_model.pth"
MEDIAPIPE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"

# 모델 로드
try:
    POINTNET_MODEL = load_pointnet(POINTNET_MODEL_PATH, DEVICE)
    MEDIAPIPE_MODEL = load_mediapipe(MEDIAPIPE_MODEL_PATH)
except Exception as e:
    print(f"모델 로드 중 오류 발생: {str(e)}")
    raise

# API 라우팅
@app.get("/", 
         summary="서버 상태 확인",
         description="API 서버의 상태를 확인합니다.",
         response_model=Dict[str, str])
def read_root():
    return {"message": "AI backend 서버입니다."}

@app.post("/image",
          summary="이미지 업로드",
          description="이미지 파일을 서버에 업로드합니다. 지원 형식: JPG, PNG, GIF 등",
          response_model=ImageUploadResponse,
          responses={
              400: {"model": ErrorResponse, "description": "잘못된 파일 형식"},
              500: {"model": ErrorResponse, "description": "서버 내부 오류"}
          })
async def image_upload(file: UploadFile = File(..., description="업로드할 이미지 파일")):
    # 파일이 이미지인지 확인
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    try:
        # 이미지 데이터 읽기
        contents = await file.read()
        
        # 파일 저장
        file_path = DATA_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(contents)
        
        return ImageUploadResponse(
            filename=file.filename,
            content_type=file.content_type,
            size=len(contents),
            message="이미지 업로드 성공",
            saved_path=str(file_path)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/face",
          summary="얼굴 분석 및 예측",
          description="업로드된 얼굴 이미지를 분석하여 랜드마크를 추출하고 AI 모델로 예측을 수행합니다.",
          response_model=PredictionResponse,
          responses={
              400: {"model": ErrorResponse, "description": "잘못된 이미지 파일"},
              500: {"model": ErrorResponse, "description": "예측 처리 중 오류"}
          })
async def predict_face(file: UploadFile = File(..., description="분석할 얼굴 이미지 파일")):
    try:
        import time
        start_time = time.time()
        
        # 이미지 데이터 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # 랜드마크 추출
        landmarks = get_landmark(MEDIAPIPE_MODEL, image_np)
        
        # 예측
        result = predict(POINTNET_MODEL, landmarks, DEVICE)
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            prediction=result.get("label", "unknown"),
            confidence=result.get("confidence", 0.0),
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

@app.get("/health",
         summary="헬스 체크",
         description="서버와 모델의 상태를 확인합니다.",
         response_model=Dict[str, Any])
def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "models_loaded": {
            "pointnet": POINTNET_MODEL is not None,
            "mediapipe": MEDIAPIPE_MODEL is not None
        }
    }