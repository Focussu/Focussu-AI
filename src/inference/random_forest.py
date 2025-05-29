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
        blendshapes: 블렌드셰이프 데이터 (torch.Tensor 또는 numpy.ndarray)
        device: 디바이스 (Random Forest는 CPU에서만 동작하므로 무시됨)
    
    Returns:
        float: 집중도 confidence 값 (0.0 ~ 1.0)
    """
    try:
        # 입력 데이터 전처리
        if isinstance(blendshapes, torch.Tensor):
            blendshapes_np = blendshapes.detach().cpu().numpy()
        else:
            blendshapes_np = np.array(blendshapes)
        
        # 1차원으로 변환 (52,) 형태로 맞춤
        if blendshapes_np.ndim > 1:
            blendshapes_np = blendshapes_np.flatten()
        
        # 배치 차원 추가 (모델이 2D 배열을 기대하므로)
        if blendshapes_np.ndim == 1:
            blendshapes_np = blendshapes_np.reshape(1, -1)
        
        # 특성 개수 확인 (52개 블렌드셰이프)
        expected_features = 52
        if blendshapes_np.shape[1] != expected_features:
            raise ValueError(f"블렌드셰이프 특성 개수가 맞지 않습니다. 예상: {expected_features}, 실제: {blendshapes_np.shape[1]}")
        
        #print(f"Random Forest 입력 데이터 형태: {blendshapes_np.shape}")
        #print(f"블렌드셰이프 값 범위: min={blendshapes_np.min():.4f}, max={blendshapes_np.max():.4f}")
        
        # 확률 예측 (각 클래스에 대한 확률)
        probabilities = model.predict_proba(blendshapes_np)[0]  # 첫 번째 샘플의 확률
        print(f"Random Forest 예측 확률: {probabilities}")
        
        # 클래스 라벨 확인
        classes = model.classes_
        
        # 집중(focused) 클래스의 확률을 confidence로 사용
        # 클래스 0: 비집중, 클래스 1: 집중
        if len(classes) == 2:
            if 0 in classes:
                focused_idx = np.where(classes == 0)[0][0]
                confidence = float(probabilities[focused_idx])
                #print(f"집중 클래스(0) 확률: {confidence}")
            else:
                # 클래스 0이 없다면 가장 높은 확률을 사용
                confidence = float(np.max(probabilities))
                print(f"최대 확률 사용: {confidence}")
        else:
            # 이진 분류가 아닌 경우 가장 높은 확률을 사용
            confidence = float(np.max(probabilities))
            print(f"다중 클래스 최대 확률: {confidence}")
        
        return confidence
        
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