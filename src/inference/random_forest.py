import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

def load_random_forest(model_path, device):
    """Random Forest 모델을 로드합니다."""
    try:
        model = joblib.load(model_path)
        print(f"Random Forest 모델 로드 완료: {model_path}")
        return model
    except Exception as e:
        print(f"Random Forest 모델 로드 실패: {str(e)}")
        raise

def predict_rf(model, blendshapes, device=None):
    """
    Random Forest 모델을 사용하여 집중도 confidence를 예측합니다.
    
    Args:
        model: 학습된 Random Forest 모델
        blendshapes: 블렌드셰이프 데이터 (torch.Tensor 또는 numpy.ndarray) - (B, 52) 형태
        device: 디바이스 (Random Forest는 CPU에서만 동작하므로 무시됨)
    
    Returns:
        float 또는 list: 집중도 confidence 값 (0.0 ~ 1.0)
    """
    try:
        # 입력 데이터 전처리
        if isinstance(blendshapes, torch.Tensor):
            blendshapes_np = blendshapes.detach().cpu().numpy()
        else:
            blendshapes_np = np.array(blendshapes)

        
        # 배치 처리를 위한 형태 확인
        if blendshapes_np.ndim == 1:
            # 1차원인 경우 (52,) -> (1, 52)로 변환
            blendshapes_np = blendshapes_np.reshape(1, -1)
        elif blendshapes_np.ndim == 2:
            # 이미 2차원인 경우 (B, 52) 형태 유지
            pass
        else:
            raise ValueError(f"지원하지 않는 입력 차원: {blendshapes_np.ndim}")
        
        # 특성 개수 확인
        expected_features = 52
        if blendshapes_np.shape[1] != expected_features:
            raise ValueError(f"블렌드셰이프 특성 개수가 맞지 않습니다. 예상: {expected_features}, 실제: {blendshapes_np.shape[1]}")
        
        print(f"Random Forest 최종 입력 형태: {blendshapes_np.shape}")
        print(f"블렌드셰이프 값 범위: min={blendshapes_np.min():.4f}, max={blendshapes_np.max():.4f}")
        
        # 배치 예측
        probabilities = model.predict_proba(blendshapes_np)  # (B, n_classes)
        print(f"Random Forest 예측 확률 형태: {probabilities.shape}")
        
        # 클래스 라벨 확인
        classes = model.classes_
        batch_size = blendshapes_np.shape[0]
        
        confidences = []
        for i in range(batch_size):
            batch_probs = probabilities[i]
            
            # 집중(focused) 클래스의 확률을 confidence로 사용
            # 클래스 0: 비집중, 클래스 1: 집중
            if len(classes) == 2:
                if 0 in classes:
                    focused_idx = np.where(classes == 0)[0][0]
                    confidence = float(batch_probs[focused_idx])
                else:
                    # 클래스 0이 없다면 가장 높은 확률을 사용
                    confidence = float(np.max(batch_probs))
            else:
                # 이진 분류가 아닌 경우 가장 높은 확률을 사용
                confidence = float(np.max(batch_probs))
            
            confidences.append(confidence)
        
        print(f"Random Forest 배치 결과: {confidences}")
        
        # 단일 배치인 경우 float 반환, 여러 배치인 경우 list 반환
        if batch_size == 1:
            return confidences[0]
        else:
            return confidences
        
    except Exception as e:
        print(f"Random Forest 예측 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # 오류 발생 시 기본값 반환
        return 0.5

def get_feature_importance(model, top_n=10):
    """
    Random Forest 모델의 특성 중요도를 반환합니다.
    
    Args:
        model: 학습된 Random Forest 모델
        top_n: 상위 몇 개의 특성을 반환할지
    
    Returns:
        dict: 특성 인덱스와 중요도를 포함한 딕셔너리
    """
    try:
        feature_importance = model.feature_importances_
        top_features = np.argsort(feature_importance)[-top_n:][::-1]
        
        importance_dict = {}
        for i, idx in enumerate(top_features):
            importance_dict[f"feature_{idx}"] = float(feature_importance[idx])
        
        return importance_dict
        
    except Exception as e:
        print(f"특성 중요도 추출 중 오류 발생: {str(e)}")
        return {}

def predict_with_explanation(model, blendshapes, device=None, top_features=5):
    """
    예측과 함께 상위 중요 특성들의 기여도를 반환합니다.
    
    Args:
        model: 학습된 Random Forest 모델
        blendshapes: 블렌드셰이프 데이터
        device: 디바이스 (무시됨)
        top_features: 상위 몇 개의 특성을 반환할지
    
    Returns:
        dict: confidence와 중요한 특성들의 정보
    """
    try:
        # 기본 예측
        confidence = predict_rf(model, blendshapes, device)
        
        # 입력 데이터 전처리
        if isinstance(blendshapes, torch.Tensor):
            blendshapes_np = blendshapes.detach().cpu().numpy().flatten()
        else:
            blendshapes_np = np.array(blendshapes).flatten()
        
        # 특성 중요도 가져오기
        feature_importance = model.feature_importances_
        top_indices = np.argsort(feature_importance)[-top_features:][::-1]
        
        important_features = {}
        for idx in top_indices:
            important_features[f"blendshape_{idx}"] = {
                "importance": float(feature_importance[idx]),
                "value": float(blendshapes_np[idx]) if idx < len(blendshapes_np) else 0.0
            }
        
        return {
            "confidence": confidence,
            "important_features": important_features
        }
        
    except Exception as e:
        print(f"설명 가능한 예측 중 오류 발생: {str(e)}")
        return {
            "confidence": 0.5,
            "important_features": {}
        } 