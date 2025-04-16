# CUDA 12.2 및 cuDNN 포함된 공식 이미지 사용
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# 기본 도구 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev curl git \
    && apt-get clean

# 심볼릭 링크 생성 (python → python3, pip → pip3)
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# 작업 디렉토리 생성
WORKDIR /workspace

# requirements 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 앱 복사
COPY ./app /workspace/app

# 기본 실행 명령 (개발 서버용)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

