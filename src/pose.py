import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

def load_model():
    """PoseNet 모델을 로드합니다."""
    model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    model = hub.load(model_url)
    return model

def process_image(image):
    """이미지를 PoseNet 모델에 맞게 전처리합니다."""
    # 이미지를 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 이미지 크기 조정 (192x192)
    image = cv2.resize(image, (192, 192))
    # 정규화
    image = image / 255.0
    # 배치 차원 추가
    image = np.expand_dims(image, axis=0)
    return image

def detect_pose(model, image):
    """이미지에서 포즈를 감지합니다."""
    # 이미지 전처리
    input_image = process_image(image)
    
    # 모델 추론
    results = model(input_image)
    
    # 결과 처리
    keypoints = results['output_0'].numpy()
    return keypoints[0]  # 첫 번째 배치의 결과만 반환

def draw_keypoints(image, keypoints):
    """감지된 키포인트를 이미지에 그립니다."""
    height, width = image.shape[:2]
    
    # 키포인트 연결 정보 (COCO 포맷)
    connections = [
        (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
        (5, 6), (5, 11), (6, 12),  # 어깨
        (11, 13), (13, 15), (12, 14), (14, 16),  # 다리
        (11, 12),  # 엉덩이
        (0, 1), (1, 2), (2, 3), (3, 4),  # 얼굴
    ]
    
    # 키포인트 그리기
    for i, (y, x, score) in enumerate(keypoints):
        if score > 0.3:  # 신뢰도 임계값
            cv2.circle(image, 
                      (int(x * width), int(y * height)), 
                      4, (0, 255, 0), -1)
    
    # 연결선 그리기
    for connection in connections:
        start_idx, end_idx = connection
        start_point = keypoints[start_idx]
        end_point = keypoints[end_idx]
        
        if start_point[2] > 0.3 and end_point[2] > 0.3:
            start_pos = (int(start_point[1] * width), int(start_point[0] * height))
            end_pos = (int(end_point[1] * width), int(end_point[0] * height))
            cv2.line(image, start_pos, end_pos, (0, 0, 255), 2)
    
    return image

def main():
    # 모델 로드
    model = load_model()
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 포즈 감지
        keypoints = detect_pose(model, frame)
        
        # 결과 시각화
        result_image = draw_keypoints(frame.copy(), keypoints)
        
        # 결과 표시
        cv2.imshow('Pose Detection', result_image)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()