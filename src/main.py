from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import requests
import io
from inference.pointnet import load_pointnet, predict
from inference.random_forest import load_random_forest, predict_rf
from datetime import datetime

load_dotenv()

API_SERVER_URL = os.getenv('API_SERVER_URL')


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
    ticketNumber: int
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
POINTNET_MODEL_PATH = MODEL_DIR / "best_multi_model.pth"
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
          description="프론트엔드에서 전송한 랜드마크와 블렌드셰이프 데이터를 받아서 AI 모델로 예측을 수행합니다. 배치 처리를 지원합니다.",
          response_model=ScoreResponse,
          responses={
              400: {"model": ErrorResponse, "description": "잘못된 데이터 형식"},
              500: {"model": ErrorResponse, "description": "예측 처리 중 오류"}
          })
async def get_score(request: ScorePredictionRequest):
    try:
        import time
        start_time = time.time()
        ticketNumber = request.ticketNumber if request.ticketNumber else 0
        # 랜드마크 데이터 처리
        if not request.landmarks:
            raise HTTPException(status_code=400, detail="랜드마크 데이터가 없습니다")
        
        landmarks_tensor = _process_landmarks(request.landmarks)

        
        # 블렌드셰이프 기반 예측 (선택적) - 첫 번째 얼굴만 사용
        if not request.blendshapes:
            raise HTTPException(status_code=400, detail="블렌드셰이프 데이터가 없습니다")
        
        blendshapes_tensor = _process_blendshapes(request.blendshapes)
        if blendshapes_tensor is None:
            raise HTTPException(status_code=400, detail="블렌드셰이프 데이터 처리 실패")
              
        # 랜드마크 기반 예측 (PointNet) - 배치 처리
        landmark_result = predict(POINTNET_MODEL, landmarks_tensor, DEVICE)
        
        # 블렌드셰이프 기반 예측 (Random Forest) - 배치 처리
        blendshape_result = predict_rf(RF_MODEL, blendshapes_tensor)
        
        

        # PyTorch 텐서를 NumPy 배열로 변환 후 평균 계산
        if isinstance(landmark_result, torch.Tensor):
            landmark_result = landmark_result.detach().cpu().numpy()
        
        landmark_result = np.mean(landmark_result, axis=0)
        pos = landmark_result[0] + landmark_result[1]
        neg = landmark_result[2] + landmark_result[3]
        #  0: 집중(흥미) 1: 집중(차분) 2: 비집중(차분) 3: 비집중(졸음) 4: 졸음
        if landmark_result[4] > 0.5:
            landmark_score = 0.0
        
        landmark_score = pos if pos > neg else neg
        print("[0]:", landmark_result[0])
        print("[1]:", landmark_result[1])
        print("[2]:", landmark_result[2])
        print("[3]:", landmark_result[3])
        print("[4]:", landmark_result[4])
        print("landmark_score:", landmark_score)
        
        if isinstance(blendshape_result, (list, tuple, np.ndarray)):
            blendshape_score = float(np.mean(blendshape_result))
        else:
            blendshape_score = float(blendshape_result)
                
    
        # 최종 confidence 계산
        final_confidence = (0.7 * landmark_score + 0.3 * blendshape_score) if blendshape_score > 0 else landmark_score
        processing_time = time.time() - start_time
        end_time = time.time()
        
        # 시간을 ISO 8601 형식으로 변환
        start_time_iso = datetime.fromtimestamp(start_time).isoformat() + 'Z'
        end_time_iso = datetime.fromtimestamp(end_time).isoformat() + 'Z'
        
        # API 서버로 분석 결과 전송 (헤더 추가)
        headers = {
            'Content-Type': 'application/json',
            #'Authorization': f'Bearer {API_TOKEN}'  # Bearer 토큰 추가
        }
        
        payload = {
            "ticketNumber": ticketNumber,
            "startTime": start_time_iso,
            "endTime": end_time_iso,
            "score": final_confidence
        }
        
        # response = requests.post(
        #     f"{API_SERVER_URL}/ai-analysis", 
        #     json=payload,  # json 파라미터 사용
        #     headers=headers
        # )
        # print(response)
        
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
    """랜드마크 데이터를 처리하여 3차원 텐서로 변환 (B, 478, 3)"""
    
    # landmarks_data가 배치로 전달되는 경우를 가정
    # 입력 형태: [batch_of_faces] 또는 [face1, face2, ...] 
    if isinstance(landmarks_data, list) and len(landmarks_data) > 0:
        batch_landmarks = []
        
        # 각 배치 항목 처리
        for batch_item in landmarks_data:
            if isinstance(batch_item, list) and len(batch_item) > 0:
                # 단일 얼굴의 랜드마크 리스트
                face_landmarks_array = []
                for lm in batch_item[0]:
                    if isinstance(lm, dict):
                        face_landmarks_array.append([lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0)])
                    else:
                        # 객체 형태라면 속성으로 접근
                        face_landmarks_array.append([getattr(lm, 'x', 0.0), getattr(lm, 'y', 0.0), getattr(lm, 'z', 0.0)])
                
                batch_landmarks.append(face_landmarks_array)
            else:
                print("랜드마크 배치 항목이 올바르지 않습니다. 건너뛰기.")
                # 올바르지 않은 데이터는 건너뛰기
                continue
        
        if not batch_landmarks:
            raise HTTPException(status_code=400, detail="처리할 수 있는 랜드마크 데이터가 없습니다")
        
        # (B, 478, 3) 형태의 텐서 생성
        landmarks_tensor = torch.tensor(batch_landmarks, dtype=torch.float32)
        print(f"배치 처리된 랜드마크 형태: {landmarks_tensor.shape}")
        return landmarks_tensor
    
    else:
        raise HTTPException(status_code=400, detail="랜드마크 데이터가 비어있거나 올바르지 않은 형식입니다")

def _process_blendshapes(blendshapes_data):
    """블렌드셰이프 데이터를 처리하여 배치 텐서로 변환 (B, 52)"""
    try:
        #type(blendshapes_data): class 'list'
        
        # MediaPipe 블렌드셰이프 구조: [{ categories: [{ score: float, categoryName: string }, ...] }]
        batch_blendshapes = []
        
        if isinstance(blendshapes_data, list) and len(blendshapes_data) > 0:
            for batch_item in blendshapes_data:
                if isinstance(batch_item, list) and len(batch_item) > 0:
                    # 첫 번째 얼굴의 블렌드셰이프 사용
                    face_blendshapes = batch_item[0]

                    # categories 키가 있는지 확인
                    if isinstance(face_blendshapes, dict) and 'categories' in face_blendshapes:
                        categories = face_blendshapes['categories']
                
                        blendshape_scores = []
           
                        for i, category in enumerate(categories): 
                            if isinstance(category, dict):
                                score = category.get('score', 0.0)
                                category_name = category.get('categoryName', f'unknown_{i}')
                                blendshape_scores.append(score)
                                # if i < 5:  # 처음 5개만 로깅
                                #     print(f"  {i}: {category_name} = {score:.6f}")
                            else:
                                # 객체 형태라면 속성으로 접근
                                score = getattr(category, 'score', 0.0)
                                blendshape_scores.append(score)
                        
                        batch_blendshapes.append(blendshape_scores)
                    else:
                        print(f"블렌드셰이프 데이터에 'categories' 키가 없습니다. 배치 항목 건너뛰기.")
                        print(f"사용 가능한 키: {list(face_blendshapes.keys()) if isinstance(face_blendshapes, dict) else 'not dict'}")
                        # 올바르지 않은 데이터는 건너뛰기 (빈 데이터 추가하지 않음)
                        continue
                else:
                    print("배치 항목이 올바르지 않습니다. 건너뛰기.")
                    # 올바르지 않은 데이터는 건너뛰기 (빈 데이터 추가하지 않음)
                    continue
        
        if not batch_blendshapes:
            print("처리할 수 있는 블렌드셰이프 데이터가 없습니다")
            return None
        
        # (B, 52) 형태의 텐서 생성
        blendshapes_tensor = torch.tensor(batch_blendshapes, dtype=torch.float32)
        
        return blendshapes_tensor
        
    except Exception as e:
        print(f"블렌드셰이프 처리 중 예외 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None


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