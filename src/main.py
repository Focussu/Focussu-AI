from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from tqdm import tqdm

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

# LLaVA 라이브러리 임포트
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.constants import IMAGE_TOKEN_INDEX
from LLaVA.llava.conversation import conv_templates
from LLaVA.llava.mm_utils import process_images, tokenizer_image_token
from sentence_transformers import SentenceTransformer, util
import openai
import shutil

from collections import defaultdict


load_dotenv()

BATCH_SIZE = 4  # 배치 크기 설정

API_SERVER_URL = os.getenv('API_SERVER_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class UserScore():
    def __init__(self):
        self.userScore = defaultdict(list)

    def addScore(self, ticketNumber, score):
        scores = self.userScore[ticketNumber]
        
        # 최근 5개만 유지
        if len(scores) >= 5:
            scores.pop(0)
        scores.append(score)

        # 급격한 변화 감지 (상승/하락 통합)
        if len(scores) >= 3:
            delta = abs(scores[-1] - scores[-2])
            if delta >= 0.25:
                print("📈 급격한 변화 감지")
                return True
            delta2 = abs(scores[-1]-scores[0])
            if delta2 >= 0.25:
                print("📈 급격한 변화 감지")
                return True
        else:
            return False



        # 저점 (집중 매우 낮음)
        if scores[-1] < 0.3 and scores[-2] < 0.3:
            print("🟠 집중도 저점")
            return True
        if all(score > 0.8 for score in scores):
            print("🟢 집중도 최고점")
            return True
        return False


userScore = UserScore()

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
    flag: bool

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
        
        # 파일명에서 ticketNumber와 timestamp 추출 시도
        try:
            file_name = file.filename.split(".")[0]
            ticketNumber, timestamp = file_name.split("_")
            ticketNumber = int(ticketNumber)
        except (ValueError, AttributeError, IndexError):
            # 파일명이 예상 형식이 아닌 경우 기본값 사용
            ticketNumber = 0
            timestamp = "unknown"
            print(f"파일명 '{file.filename}'이 예상 형식(ticketNumber_timestamp.확장자)이 아닙니다. 기본값을 사용합니다.")
        
        # 파일 저장
        # ticketNumber별 디렉토리 생성
        ticket_dir = DATA_DIR / str(ticketNumber)
        ticket_dir.mkdir(exist_ok=True)  # 디렉토리가 없으면 생성
        
        file_path = ticket_dir / file.filename
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
    
openai.api_key = OPENAI_API_KEY

# ====== 모델 로드 ======
print("Loading LLaVA model...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="/home/focussu/minji/Focussu-AI/src/LLaVA/checkpoints/llava-v1.5-7b",
    model_base=None,
    model_name="llava-v1.5-7b",
    device_map="cuda",
    load_8bit=True,
    load_4bit=False
)
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype
print(f"Model loaded on device: {device}, dtype: {dtype}")

#app = FastAPI()

# ====== 전역 템플릿 및 상수 ======
conv_template = conv_templates["llava_v1"].copy()
IMAGE_TOKEN_INDEX = tokenizer.convert_tokens_to_ids(["<image>"])[0]  # 필요한 경우 직접 설정
base_prompt = (
    "This is a frame taken from a video recorded every 10 seconds. "
    "Please analyze the person's posture, gaze direction, and activity. "
    "Evaluate their level of concentration and explain why you think so. "
    "If the person seems distracted, suggest possible causes."
)

# 이미지 불러오기
def load_images_from_folder(folder_path: str) -> List[Image.Image]:
    image_list = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path).convert("RGB")
            image_list.append(img)
    return image_list

# 프롬프트 생성
def make_prompts(start_idx: int, end_idx: int) -> List[str]:
    return [f"{base_prompt}\n\nThis is frame {i+1} in the sequence." for i in range(start_idx, end_idx)]


@app.post("/analyze")
async def analyze_concentration(ticketNumber: int, userID: int):
# async def analyze_concentration(files: List[UploadFile] = File(...)):
    print(f"집중도 분석 시작 - ticketNumber: {ticketNumber}, userID: {userID}")
    try:
        folder_path = f"/home/focussu/minji/Focussu-AI/data/{ticketNumber}"
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"폴더가 존재하지 않음: {folder_path}")

        # 1. 이미지 로딩 및 전처리
        image_list = load_images_from_folder(folder_path)
        if len(image_list) == 0:
            raise ValueError("이미지가 없습니다.")
        image_tensor = process_pil_images(image_list)
        
        # 2. 배치로 나눠 처리
        all_responses = []
        num_images = image_tensor.shape[0]
        for i in tqdm(range(0, num_images, BATCH_SIZE)):
            image_batch = image_tensor[i:i+BATCH_SIZE]
            prompts = make_prompts(i, min(i + BATCH_SIZE, num_images))
            batch_responses = generate_responses(image_batch, prompts)
            all_responses.extend(batch_responses)
        
        # 3. 중복 제거 및 정리
        unique_responses = deduplicate_responses(all_responses)
        translated = translate_with_gpt(unique_responses)


        # Post (ticketNumber, content)
        
        #    API 서버로 분석 결과 전송 (헤더 추가)
        headers = {
            'Content-Type': 'application/json',
            #'Authorization': f'Bearer {API_TOKEN}'  # Bearer 토큰 추가
        }
        
        payload = {
            "ticketNumber": ticketNumber,
            # "userID": userID,
            "content": translated,
        }
        
        response = requests.post(
            f"{API_SERVER_URL}/analysis-document", 
            json=payload,  # json 파라미터 사용
            headers=headers
        )
        print(response)
        
        return {
            "status": "success",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====== 유틸 함수들 ======
def process_uploaded_images(files: List[UploadFile]) -> torch.Tensor:
    images = [Image.open(io.BytesIO(file.file.read())).convert("RGB") for file in files]
    image_tensor = process_images(images, image_processor, model.config)
    return image_tensor.to(device=device, dtype=dtype)

def process_pil_images(images: List[Image.Image]) -> torch.Tensor:
    """PIL Image 리스트를 처리하여 텐서로 변환"""
    image_tensor = process_images(images, image_processor, model.config)
    return image_tensor.to(device=device, dtype=dtype)

def generate_responses(image_tensor_batch: torch.Tensor, prompts: List[str]) -> List[str]:
    conv_batch = [conv_template.copy() for _ in prompts]
    for conv, prompt in zip(conv_batch, prompts):
        conv.messages = []
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)

    input_ids_batch = [
        tokenizer_image_token(c.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        for c in conv_batch
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor_batch,
            do_sample=False,
            temperature=0.2,
            max_new_tokens=256,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # 응답 디코딩
    return [
        tokenizer.decode(output_ids[i][input_ids.shape[1]:], skip_special_tokens=True).strip()
        for i in range(output_ids.shape[0])
    ]


def deduplicate_responses(paragraphs: List[str], threshold: float = 0.85) -> List[str]:
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(paragraphs, convert_to_tensor=True)
    unique_paragraphs = []

    for i, emb in enumerate(embeddings):
        if all(util.cos_sim(emb, model.encode([p], convert_to_tensor=True)).item() < threshold for p in unique_paragraphs):
            unique_paragraphs.append(paragraphs[i])
    return unique_paragraphs


def translate_with_gpt(sentences: List[str]) -> str:
    prompt = (
        "다음 문장들은 영상 속 사람 한 명의 시간에 따른 집중도, 자세, 시선을 분석한 내용입니다.\n"
        "이 문장들을 중복 없이 자연스럽게 한국어로 번역하고, 문맥이 어색한 부분이나 말이 이어지지 않는 부분은 자연스럽게 다듬어 주세요. 문장 앞에 넘버링은 매기지 말아주세요.\n\n"
    )
    joined = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt + joined}],
        temperature=0.2,
    )
    return response.choices[0].message["content"].strip()

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
        print(f"ticketNumber: {ticketNumber}")
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
        else:
            landmark_score = pos if pos > neg else (1-neg)
        
        if isinstance(blendshape_result, (list, tuple, np.ndarray)):
            blendshape_score = float(np.mean(blendshape_result))
        else:
            blendshape_score = float(blendshape_result)
                
    
        # 최종 confidence 계산
        final_confidence = (0.8 * landmark_score + 0.2 * blendshape_score) if blendshape_score > 0 else landmark_score
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
        
        response = requests.post(
            f"{API_SERVER_URL}/ai-analysis", 
            json=payload,  # json 파라미터 사용
            headers=headers
        )
        print(response)

        # 점수 리스트 추가 및 이미지 저장 여부 설정
        flag = userScore.addScore(ticketNumber, final_confidence)

        return ScoreResponse(
            landmark_score=landmark_score,
            blendshape_score=blendshape_score,
            confidence=float(final_confidence),
            processing_time=round(processing_time, 3),
            flag=flag
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