from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

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
