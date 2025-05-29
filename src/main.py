from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image


import io
from inference.pointnet import load_pointnet, predict
from inference.random_forest import load_random_forest, predict_rf

# Pydantic 모델 정의
class ImageUploadResponse(BaseModel):
    filename: str
    content_type: str
    size: int
    message: str
    saved_path: str

class LandmarkData(BaseModel):
    x: float
    y: float
    z: float

class BlendshapeData(BaseModel):
    categoryName: str
    score: float

class ScorePredictionRequest(BaseModel):
    landmarks: Any  # MediaPipe 구조에 맞게 Any로 변경
    blendshapes: Optional[Any] = None  # MediaPipe 구조에 맞게 Any로 변경
    timestamp: int

class ScoreResponse(BaseModel):
    landmark_score: float
    blendshape_score: float
    confidence: float
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
POINTNET_MODEL_PATH = MODEL_DIR / "best_binary_model.pth"
RF_MODEL_PATH = MODEL_DIR / "random_forest_blendshape.pkl"

# 모델 로드
try:
    POINTNET_MODEL = load_pointnet(POINTNET_MODEL_PATH, DEVICE)
    RF_MODEL = load_random_forest(RF_MODEL_PATH, DEVICE)
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



@app.post("/ai/score",
          summary="얼굴 분석 및 예측",
          description="프론트엔드에서 전송한 랜드마크와 블렌드셰이프 데이터를 받아서 AI 모델로 예측을 수행합니다.",
          response_model=ScoreResponse,
          responses={
              400: {"model": ErrorResponse, "description": "잘못된 데이터 형식"},
              500: {"model": ErrorResponse, "description": "예측 처리 중 오류"}
          })
async def get_score(request: ScorePredictionRequest):
    try:
        import time
        start_time = time.time()
        
        # 랜드마크 데이터 처리
        if not request.landmarks:
            raise HTTPException(status_code=400, detail="랜드마크 데이터가 없습니다")
        
        landmarks_tensor = _process_landmarks(request.landmarks)
        print(f"처리된 랜드마크 형태: {landmarks_tensor.shape}")
        
        # 랜드마크 기반 예측 (PointNet)
        landmark_result = predict(POINTNET_MODEL, landmarks_tensor, DEVICE)
        landmark_score = float(landmark_result.get("confidence", 0.0)) if isinstance(landmark_result, dict) else float(landmark_result)
        
        # 블렌드셰이프 기반 예측 (선택적)
        blendshape_score = 0.0
        if request.blendshapes:
            try:
                blendshapes_tensor = _process_blendshapes(request.blendshapes)
                if blendshapes_tensor is not None:
                    blendshape_score = predict_rf(RF_MODEL, blendshapes_tensor)
                    blendshape_score = float(blendshape_score)
                    print(f"블렌드셰이프 점수: {blendshape_score}")
            except Exception as e:
                print(f"블렌드셰이프 처리 중 오류: {str(e)}")
                blendshape_score = 0.0
        
        # 최종 confidence 계산
        final_confidence = (0.7 * landmark_score + 0.3 * blendshape_score) if blendshape_score > 0 else landmark_score
        processing_time = time.time() - start_time
        
        return ScoreResponse(
            landmark_score=landmark_score,
            blendshape_score=blendshape_score,
            confidence=float(final_confidence),
            processing_time=round(processing_time, 3)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"예측 중 상세 오류: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

def _process_landmarks(landmarks_data):
    """랜드마크 데이터를 처리하여 텐서로 변환"""
    # landmarks가 리스트인 경우 (여러 얼굴)
    if isinstance(landmarks_data, list) and len(landmarks_data) > 0:
        # 첫 번째 얼굴의 랜드마크 사용
        face_landmarks = landmarks_data[0]
        
        # 각 랜드마크가 dict 형태인지 확인
        if isinstance(face_landmarks, list):
            landmarks_array = []
            for lm in face_landmarks:
                if isinstance(lm, dict):
                    landmarks_array.append([lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0)])
                else:
                    # 객체 형태라면 속성으로 접근
                    landmarks_array.append([getattr(lm, 'x', 0.0), getattr(lm, 'y', 0.0), getattr(lm, 'z', 0.0)])
            
            landmarks_tensor = torch.tensor(landmarks_array, dtype=torch.float32)

            return landmarks_tensor
        else:
            raise HTTPException(status_code=400, detail="랜드마크 데이터 형식이 올바르지 않습니다")
    else:
        raise HTTPException(status_code=400, detail="랜드마크 데이터가 비어있습니다")

def _process_blendshapes(blendshapes_data):
    """블렌드셰이프 데이터를 처리하여 텐서로 변환"""
    try:
        print(f"블렌드셰이프 원본 데이터 구조: {type(blendshapes_data)}")
        
        # MediaPipe 블렌드셰이프 구조: [{ categories: [{ score: float, categoryName: string }, ...] }]
        if isinstance(blendshapes_data, list) and len(blendshapes_data) > 0:
            # 첫 번째 얼굴의 블렌드셰이프 사용
            face_blendshapes = blendshapes_data[0]

            
            # categories 키가 있는지 확인
            if isinstance(face_blendshapes, dict) and 'categories' in face_blendshapes:
                categories = face_blendshapes['categories']
        
                blendshape_scores = []
                print(f"categories 개수: {len(categories)}")
                for i, category in enumerate(categories): 
                    if isinstance(category, dict):
                        score = category.get('score', 0.0)
                        category_name = category.get('categoryName', f'unknown_{i}')
                        blendshape_scores.append(score)
                        if i < 5:  # 처음 5개만 로깅
                            print(f"  {i}: {category_name} = {score:.6f}")
                    else:
                        # 객체 형태라면 속성으로 접근
                        score = getattr(category, 'score', 0.0)
                        blendshape_scores.append(score)

                return torch.tensor(blendshape_scores[:52], dtype=torch.float32)
                
            else:
                print("블렌드셰이프 데이터에 'categories' 키가 없습니다")
                print(f"사용 가능한 키: {list(face_blendshapes.keys()) if isinstance(face_blendshapes, dict) else 'not dict'}")
                return None
        else:
            print("블렌드셰이프 데이터가 비어있거나 올바르지 않습니다")
            return None
    except Exception as e:
        print(f"블렌드셰이프 처리 중 예외 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

@app.post("/predict/landmarks",
          summary="랜드마크/블렌드셰이프 데이터 예측",
          description="/ai/score와 동일한 기능을 제공하는 대체 엔드포인트입니다.",
          response_model=ScoreResponse,
          responses={
              400: {"model": ErrorResponse, "description": "잘못된 데이터 형식"},
              500: {"model": ErrorResponse, "description": "예측 처리 중 오류"}
          })
async def predict_from_frontend(request: ScorePredictionRequest):
    """프론트엔드에서 호출하는 엔드포인트 - /ai/score와 동일한 기능"""
    return await get_score(request)

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
            "random_forest": RF_MODEL is not None
        }
    }