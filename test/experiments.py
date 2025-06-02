import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import defaultdict
import pandas as pd
from torch.utils.data import DataLoader
from model.PointNet import PointNetClassifier
from train.data import FocusDataset_multi
import os
from sklearn.decomposition import PCA


import matplotlib.font_manager
font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
[matplotlib.font_manager.FontProperties(fname=font).get_name() for font in font_list if 'Nanum' in font]
\
import matplotlib.pyplot as plt
plt.rc('font', family='NanumGothicCoding')
\
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = '/shared_data/focussu/109.학습태도_및_성향_관찰_데이터/3.개방데이터/1.데이터/Validation'
MODEL_PATH = 'model/best_multi_model.pth'

# experiments 폴더 생성
EXPERIMENTS_DIR = 'experiments'
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# 라벨 정의
LABEL_NAMES = {
    0: "집중(흥미로움)",
    1: "집중(차분함)", 
    2: "비집중(차분함)",
    3: "비집중(지루함)",
    4: "졸음"
}

def evaluate_model_performance(model, dataloader, device):
    """모델의 종합적인 성능 지표를 계산하고 보고서 생성"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_logits = []  # softmax 확률
    all_raw_logits = []  # raw logits (softmax 이전)
    all_features = []  # 모델의 feature 추출 부분
    all_formats = []  # format 정보 저장
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    print("모델 평가를 시작합니다...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            landmarks = batch['landmarks'].to(device)
            labels = batch['label'].long().to(device)
            formats = batch['format']  # format 정보 가져오기
            
            # 예측 수행 (feature와 raw logits 모두 추출)
            logits = model(landmarks)
            
            # 모델의 feature extraction 부분만 실행 (PointNet의 경우)
            try:
                # PointNet의 feature extraction 부분 실행
                features = model.extract_features(landmarks)
                all_features.extend(features.cpu().numpy())
            except AttributeError:
                # extract_features 메소드가 없는 경우, raw logits 사용
                all_features.extend(logits.cpu().numpy())  # raw logits를 features로 사용
            
            _, predicted = torch.max(logits.data, 1)
            
            # 결과 저장
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_raw_logits.extend(logits.cpu().numpy())  # raw logits 저장
            all_logits.extend(torch.softmax(logits, dim=1).cpu().numpy())  # softmax 확률
            all_formats.extend(formats)  # format 정보 저장
            
            # 클래스별 정확도 계산을 위한 데이터 수집
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
            
            if batch_idx % 10 == 0:
                print(f"배치 {batch_idx}/{len(dataloader)} 처리 완료")
    
    # NumPy 배열로 변환
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    all_raw_logits = np.array(all_raw_logits)
    all_features = np.array(all_features)
    
    print(f"\n총 {len(all_predictions)}개 샘플 평가 완료")
    print(f"Feature 차원: {all_features.shape}")
    print(f"Raw logits 차원: {all_raw_logits.shape}")
    print(f"Softmax 확률 차원: {all_logits.shape}")
    
    # 1. 전체 정확도
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    
    # 2. 클래스별 정확도
    class_accuracies = {}
    for class_id in range(5):
        if class_total[class_id] > 0:
            class_accuracies[class_id] = class_correct[class_id] / class_total[class_id]
        else:
            class_accuracies[class_id] = 0.0
    
    # 3. Precision, Recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # 4. Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 5. Classification Report
    class_report = classification_report(
        all_labels, all_predictions, 
        target_names=[LABEL_NAMES[i] for i in range(5)],
        zero_division=0
    )
    
    return {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': class_accuracies,
        'class_total': dict(class_total),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'predictions': all_predictions,
        'labels': all_labels,
        'logits': all_logits,  # softmax 확률
        'raw_logits': all_raw_logits,  # raw logits
        'features': all_features,  # feature vectors
        'formats': all_formats  # format 정보 추가
    }

def print_performance_report(results):
    """성능 지표 보고서를 출력"""
    print("=" * 80)
    print("모델 성능 평가 보고서")
    print("=" * 80)
    
    # 전체 정확도
    print(f"\n📊 전체 정확도: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    
    
    # 클래스별 상세 정보
    print(f"\n📋 클래스별 성능 지표:")
    print("-" * 80)
    print(f"{'클래스':<15} {'샘플수':<8} {'정확도':<8} {'정밀도':<8} {'재현율':<8} {'F1점수':<8}")
    print("-" * 80)
    
    for i in range(5):
        class_name = LABEL_NAMES[i]
        sample_count = results['class_total'].get(i, 0)
        accuracy = results['class_accuracies'][i]
        precision = results['precision'][i]
        recall = results['recall'][i]
        f1 = results['f1_score'][i]
        
        print(f"{class_name:<15} {sample_count:<8} {accuracy:<8.4f} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f}")
    
    # 평균 지표
    print("-" * 80)
    avg_precision = np.mean(results['precision'])
    avg_recall = np.mean(results['recall'])
    avg_f1 = np.mean(results['f1_score'])
    
    print(f"{'평균':<15} {'':<8} {'':<8} {avg_precision:<8.4f} {avg_recall:<8.4f} {avg_f1:<8.4f}")
    
    # 가중 평균 지표
    weighted_precision = np.average(results['precision'], weights=results['support'])
    weighted_recall = np.average(results['recall'], weights=results['support'])
    weighted_f1 = np.average(results['f1_score'], weights=results['support'])
    
    print(f"{'가중평균':<15} {'':<8} {'':<8} {weighted_precision:<8.4f} {weighted_recall:<8.4f} {weighted_f1:<8.4f}")
    
    # 상세 분류 보고서
    print(f"\n📈 상세 분류 보고서:")
    print(results['classification_report'])

def plot_confusion_matrix(cm, save_path=None):
    """Confusion Matrix 시각화"""
    if save_path is None:
        save_path = os.path.join(EXPERIMENTS_DIR, 'confusion_matrix.png')
    
    # 동적 라벨 가져오기 (이미 폰트 설정도 포함)
    
    plt.figure(figsize=(12, 10))
    
    # 정규화된 confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 라벨 설정
    class_labels = [LABEL_NAMES[i] for i in range(5)]
    
    # 히트맵 생성
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.3f', 
                cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels,
                cbar_kws={'label': 'Normalized Frequency'},
                annot_kws={'size': 12})
    
    plt.title('Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    # 여백 조정
    plt.tight_layout()
    
    # 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Confusion Matrix가 {save_path}에 저장되었습니다.")
    plt.show()

def plot_class_performance(results, save_path=None):
    """클래스별 성능 지표 시각화"""
    if save_path is None:
        save_path = os.path.join(EXPERIMENTS_DIR, 'class_performance.png')
    
    # 동적 라벨 가져오기 (이미 폰트 설정도 포함)
 
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    
    # 라벨 설정
    class_names = [LABEL_NAMES[i] for i in range(5)]
    
    # 1. 클래스별 정확도
    accuracies = [results['class_accuracies'][i] for i in range(5)]
    bars1 = axes[0, 0].bar(class_names, accuracies, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
    axes[0, 0].set_title('Class Accuracy', fontweight='bold', fontsize=16, pad=15)
    axes[0, 0].set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].tick_params(axis='x', rotation=35, labelsize=11)
    axes[0, 0].tick_params(axis='y', labelsize=11)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    # 2. Precision, Recall, F1-score 비교
    x = np.arange(len(class_names))
    width = 0.25
    
    bars2_1 = axes[0, 1].bar(x - width, results['precision'], width, label='Precision', alpha=0.8, color='lightcoral')
    bars2_2 = axes[0, 1].bar(x, results['recall'], width, label='Recall', alpha=0.8, color='lightgreen')
    bars2_3 = axes[0, 1].bar(x + width, results['f1_score'], width, label='F1-Score', alpha=0.8, color='lightsalmon')
    
    axes[0, 1].set_title('Precision, Recall, F1-Score', fontweight='bold', fontsize=16, pad=15)
    axes[0, 1].set_ylabel('Score', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(class_names, rotation=35, ha='right', fontsize=11)
    axes[0, 1].tick_params(axis='y', labelsize=11)
    axes[0, 1].legend(fontsize=12, loc='upper right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. 클래스별 샘플 수 (훈련 데이터)
    sample_counts = [results['class_total'].get(i, 0) for i in range(5)]
    bars3 = axes[1, 0].bar(class_names, sample_counts, color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)
    axes[1, 0].set_title('Total Sample Count', fontweight='bold', fontsize=16, pad=15)
    axes[1, 0].set_ylabel('Count', fontsize=13, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=35, labelsize=11)
    axes[1, 0].tick_params(axis='y', labelsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for i, v in enumerate(sample_counts):
        if v > 0:
            axes[1, 0].text(i, v + max(sample_counts)*0.02, str(v), ha='center', fontsize=11, fontweight='bold')
    
    # 4. Support (테스트 세트 클래스별 샘플 수)
    bars4 = axes[1, 1].bar(class_names, results['support'], color='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=1)
    axes[1, 1].set_title('Test Set Sample Count', fontweight='bold', fontsize=16, pad=15)
    axes[1, 1].set_ylabel('Count', fontsize=13, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=35, labelsize=11)
    axes[1, 1].tick_params(axis='y', labelsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for i, v in enumerate(results['support']):
        if v > 0:
            axes[1, 1].text(i, v + max(results['support'])*0.02, str(v), ha='center', fontsize=11, fontweight='bold')
    
    # 전체 레이아웃 조정
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 클래스 성능 차트가 {save_path}에 저장되었습니다.")
    plt.show()

def plot_tsne_visualization(results, perplexity=30, n_iter=1000, save_path=None, 
                           input_type='raw_logits', max_samples=5000):
    """t-SNE를 사용한 분류 결과 분포 시각화
    
    Args:
        input_type: 't-SNE 입력 데이터 유형
            - 'features': 모델의 feature vectors (권장)
            - 'raw_logits': raw logits (softmax 이전)
            - 'softmax': softmax 확률 (권장하지 않음)
    """
    if save_path is None:
        save_path = os.path.join(EXPERIMENTS_DIR, f'tsne_{input_type}_visualization.png')
    
    # 입력 데이터 선택
    if input_type == 'features':
        input_data = results['features']
        data_name = "Feature Vectors"
    elif input_type == 'raw_logits':
        input_data = results['raw_logits']
        data_name = "Raw Logits"
    elif input_type == 'softmax':
        input_data = results['logits']
        data_name = "Softmax Probabilities"
    else:
        raise ValueError("input_type은 'features', 'raw_logits', 'softmax' 중 하나여야 합니다.")
    
    true_labels = results['labels']
    predicted_labels = results['predictions']
    
    print(f"\n🔍 t-SNE 분석 ({data_name} 사용)")
    print(f"   • 입력 데이터 형태: {input_data.shape}")
    print(f"   • 입력 데이터 타입: {input_type}")
    
    # 데이터 특성 분석
    print(f"   • 데이터 범위: [{input_data.min():.4f}, {input_data.max():.4f}]")
    print(f"   • 데이터 평균: {input_data.mean():.4f}")
    print(f"   • 데이터 표준편차: {input_data.std():.4f}")
    
    # Softmax 확률 사용 시 경고
    if input_type == 'softmax':
        print("⚠️  경고: Softmax 확률을 t-SNE 입력으로 사용하고 있습니다.")
        print("   - Softmax는 simplex 제약으로 인해 정보가 제한됩니다.")
        print("   - Raw logits 또는 feature vectors 사용을 권장합니다.")
    
    # 메모리 절약을 위한 데이터 샘플링
    if len(input_data) > max_samples:
        print(f"⚠️  메모리 절약을 위해 {len(input_data)}개 샘플 중 {max_samples}개를 랜덤 샘플링합니다.")
        
        # 클래스별 균등 샘플링
        sampled_indices = []
        samples_per_class = max_samples // 5
        
        for class_id in range(5):
            class_indices = np.where(true_labels == class_id)[0]
            if len(class_indices) > samples_per_class:
                selected = np.random.choice(class_indices, samples_per_class, replace=False)
            else:
                selected = class_indices
            sampled_indices.extend(selected)
        
        # 남은 샘플로 부족한 부분 채우기
        if len(sampled_indices) < max_samples:
            remaining_indices = np.setdiff1d(np.arange(len(input_data)), sampled_indices)
            additional_needed = max_samples - len(sampled_indices)
            if len(remaining_indices) > 0:
                additional = np.random.choice(remaining_indices, 
                                            min(additional_needed, len(remaining_indices)), 
                                            replace=False)
                sampled_indices.extend(additional)
        
        sampled_indices = np.array(sampled_indices)
        
        # 샘플링된 데이터 사용
        input_data = input_data[sampled_indices]
        true_labels = true_labels[sampled_indices]
        predicted_labels = predicted_labels[sampled_indices]
        
        print(f"   • 샘플링 후 데이터 형태: {input_data.shape}")
        
        # 클래스별 샘플 수 확인
        print("   • 클래스별 샘플 수:")
        for i in range(5):
            count = np.sum(true_labels == i)
            print(f"     - {LABEL_NAMES[i]}: {count}개")
    
    # 메모리 효율적인 t-SNE 파라미터 조정
    if perplexity >= len(input_data) / 3:
        perplexity = max(5, len(input_data) // 4)
        print(f"⚠️  Perplexity를 {perplexity}로 조정합니다.")
    
    try:
        print(f"🔍 t-SNE 차원 축소를 수행합니다...")
        print(f"   • Perplexity: {perplexity}")
        print(f"   • 반복 횟수: {n_iter}")
        print(f"   • 샘플 수: {len(input_data)}")
        print("   • 시간이 다소 걸릴 수 있습니다...")
        
        # 메모리 효율적인 t-SNE 설정
        tsne = TSNE(
            n_components=2, 
            perplexity=perplexity, 
            n_iter=n_iter, 
            random_state=405, 
            verbose=0,
            n_jobs=1,
            learning_rate='auto'
        )
        tsne_results = tsne.fit_transform(input_data)
        
        print("✅ t-SNE 완료!")
        
    except MemoryError:
        print("❌ t-SNE 실행 중 메모리 부족 오류가 발생했습니다.")
        print("🔄 PCA로 대체하여 차원 축소를 수행합니다...")
        
        pca = PCA(n_components=2, random_state=405)
        tsne_results = pca.fit_transform(input_data)
        
        print(f"✅ PCA 차원 축소 완료 (설명 분산비: {pca.explained_variance_ratio_.sum():.3f})")
        save_path = save_path.replace('tsne_', 'pca_')
        data_name = f"PCA ({data_name})"
    
    except Exception as e:
        print(f"❌ 차원 축소 중 오류 발생: {e}")
        print("🔄 단순 PCA로 대체합니다...")
        
        pca = PCA(n_components=2, random_state=42)
        tsne_results = pca.fit_transform(input_data)
        save_path = save_path.replace('tsne_', 'pca_')
        data_name = f"PCA ({data_name})"
    
    # 시각화 - 더 명확하게 구분되는 색상 사용
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # 명확하게 구분되는 5가지 색상 (색맹 친화적)
    colors = ['#D32F2F', '#1976D2', '#388E3C', '#7B1FA2', '#F57C00']  # 빨강, 파랑, 초록, 보라, 주황
    class_names = [LABEL_NAMES[i] for i in range(5)]
    
    method_name = 'PCA' if 'pca_' in save_path else 't-SNE'
    
    # 1. 실제 라벨별 분포
    for i in range(5):
        mask = true_labels == i
        if np.sum(mask) > 0:
            axes[0].scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                           c=colors[i], label=class_names[i], alpha=0.8, s=25, edgecolors='black', linewidth=0.5)
    
    axes[0].set_title(f'{method_name}: 실제 라벨별 분포\n({data_name})', 
                      fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel(f'{method_name} 차원 1', fontsize=12)
    axes[0].set_ylabel(f'{method_name} 차원 2', fontsize=12)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. 예측 라벨별 분포
    for i in range(5):
        mask = predicted_labels == i
        if np.sum(mask) > 0:
            axes[1].scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                           c=colors[i], label=class_names[i], alpha=0.8, s=25, edgecolors='black', linewidth=0.5)
    
    axes[1].set_title(f'{method_name}: 예측 라벨별 분포\n({data_name})', 
                      fontsize=16, fontweight='bold', pad=15)
    axes[1].set_xlabel(f'{method_name} 차원 1', fontsize=12)
    axes[1].set_ylabel(f'{method_name} 차원 2', fontsize=12)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 3. 정분류/오분류 구분
    correct_mask = true_labels == predicted_labels
    incorrect_mask = ~correct_mask
    
    if np.sum(correct_mask) > 0:
        axes[2].scatter(tsne_results[correct_mask, 0], tsne_results[correct_mask, 1], 
                       c='#2E7D32', label=f'정분류 ({np.sum(correct_mask)}개)', 
                       alpha=0.8, s=25, edgecolors='black', linewidth=0.5)
    
    if np.sum(incorrect_mask) > 0:
        axes[2].scatter(tsne_results[incorrect_mask, 0], tsne_results[incorrect_mask, 1], 
                       c='#C62828', label=f'오분류 ({np.sum(incorrect_mask)}개)', 
                       alpha=0.8, s=25, edgecolors='black', linewidth=0.5)
    
    axes[2].set_title(f'{method_name}: 정분류 vs 오분류\n({data_name})', 
                      fontsize=16, fontweight='bold', pad=15)
    axes[2].set_xlabel(f'{method_name} 차원 1', fontsize=12)
    axes[2].set_ylabel(f'{method_name} 차원 2', fontsize=12)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ {method_name} 시각화가 {save_path}에 저장되었습니다.")
    plt.show()
    
    # 통계 정보 출력
    print(f"\n📊 {method_name} 시각화 통계:")
    print(f"   • 입력 데이터: {data_name}")
    print(f"   • 사용된 샘플 수: {len(tsne_results)}")
    print(f"   • 정분류 샘플: {np.sum(correct_mask)} ({np.sum(correct_mask)/len(tsne_results)*100:.1f}%)")
    print(f"   • 오분류 샘플: {np.sum(incorrect_mask)} ({np.sum(incorrect_mask)/len(tsne_results)*100:.1f}%)")
    if method_name == 't-SNE':
        print(f"   • Perplexity: {perplexity}")
        print(f"   • 반복 횟수: {n_iter}")
    
    # 클래스별 클러스터 밀집도 분석
    print(f"\n🔍 클래스별 클러스터 분석:")
    for i in range(5):
        class_mask = true_labels == i
        class_count = np.sum(class_mask)
        if class_count > 1:
            class_points = tsne_results[class_mask]
            
            # 샘플이 너무 많으면 일부만 사용하여 거리 계산
            if len(class_points) > 100:
                sample_indices = np.random.choice(len(class_points), 100, replace=False)
                class_points_sample = class_points[sample_indices]
            else:
                class_points_sample = class_points
            
            # 클래스 내 점들 간의 평균 거리 계산
            if len(class_points_sample) > 1:
                from scipy.spatial.distance import pdist
                distances = pdist(class_points_sample)
                avg_distance = np.mean(distances)
                std_distance = np.std(distances)
                print(f"   • {class_names[i]} ({class_count}개): 평균 거리 {avg_distance:.2f} ± {std_distance:.2f}")
            else:
                print(f"   • {class_names[i]} ({class_count}개): 거리 계산 불가 (샘플 부족)")
        else:
            print(f"   • {class_names[i]} ({class_count}개): 분석 불가 (샘플 부족)")

def compare_tsne_inputs(results, save_path=None):
    """다양한 입력 데이터를 사용한 t-SNE 비교 시각화"""
    if save_path is None:
        save_path = os.path.join(EXPERIMENTS_DIR, 'tsne_comparison.png')
    
    print("\n🔍 다양한 입력 데이터를 사용한 t-SNE 비교 분석을 시작합니다...")
    
    # 사용할 입력 데이터들
    input_types = ['features', 'raw_logits', 'softmax']
    input_names = ['Feature Vectors', 'Raw Logits', 'Softmax Probabilities']
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # 명확하게 구분되는 5가지 색상
    colors = ['#D32F2F', '#1976D2', '#388E3C', '#7B1FA2', '#F57C00']  # 빨강, 파랑, 초록, 보라, 주황
    class_names = [LABEL_NAMES[i] for i in range(5)]
    
    for idx, (input_type, input_name) in enumerate(zip(input_types, input_names)):
        print(f"\n📊 {input_name} 분석 중...")
        
        # 입력 데이터 준비
        if input_type == 'features':
            input_data = results['features']
        elif input_type == 'raw_logits':
            input_data = results['raw_logits']
        else:  # softmax
            input_data = results['logits']
        
        true_labels = results['labels']
        
        # 샘플링 (메모리 절약)
        max_samples = 2000  # 비교를 위해 더 적은 샘플 사용
        if len(input_data) > max_samples:
            indices = np.random.choice(len(input_data), max_samples, replace=False)
            input_data = input_data[indices]
            true_labels = true_labels[indices]
        
        try:
            # t-SNE 수행
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, 
                       random_state=405, verbose=0)
            tsne_results = tsne.fit_transform(input_data)
            
            # 시각화
            for i in range(5):
                mask = true_labels == i
                if np.sum(mask) > 0:
                    axes[idx].scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                                     c=colors[i], label=class_names[i], alpha=0.8, s=20, 
                                     edgecolors='black', linewidth=0.3)
            
            axes[idx].set_title(f't-SNE: {input_name}\n'
                               f'({input_data.shape[0]} samples, {input_data.shape[1]} dims)', 
                               fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('t-SNE 차원 1', fontsize=12)
            axes[idx].set_ylabel('t-SNE 차원 2', fontsize=12)
            if idx == 0:  # 첫 번째 플롯에만 범례 표시
                axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            axes[idx].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"❌ {input_name} t-SNE 실행 중 오류: {e}")
            axes[idx].text(0.5, 0.5, f'오류 발생\n{input_name}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ t-SNE 비교 시각화가 {save_path}에 저장되었습니다.")
    plt.show()

def plot_confidence_distribution(results, save_path=None):
    """예측 확신도 분포 시각화"""
    if save_path is None:
        save_path = os.path.join(EXPERIMENTS_DIR, 'confidence_distribution.png')
    
    print("🎯 예측 확신도 분포를 시각화합니다...")
    
    # 각 예측에 대한 최대 확률 (확신도) 계산
    max_probs = np.max(results['logits'], axis=1)
    true_labels = results['labels']
    predicted_labels = results['predictions']
    correct_mask = true_labels == predicted_labels
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 전체 확신도 분포
    axes[0, 0].hist(max_probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(max_probs), color='red', linestyle='--', 
                       label=f'평균: {np.mean(max_probs):.3f}')
    axes[0, 0].set_title('전체 예측 확신도 분포', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('확신도 (최대 확률)', fontsize=12)
    axes[0, 0].set_ylabel('빈도', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 정분류 vs 오분류 확신도 비교
    correct_probs = max_probs[correct_mask]
    incorrect_probs = max_probs[~correct_mask]
    
    axes[0, 1].hist(correct_probs, bins=20, alpha=0.7, color='green', 
                    label=f'정분류 (평균: {np.mean(correct_probs):.3f})', density=True)
    axes[0, 1].hist(incorrect_probs, bins=20, alpha=0.7, color='red', 
                    label=f'오분류 (평균: {np.mean(incorrect_probs):.3f})', density=True)
    axes[0, 1].set_title('정분류 vs 오분류 확신도 분포', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('확신도 (최대 확률)', fontsize=12)
    axes[0, 1].set_ylabel('밀도', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 클래스별 확신도 분포
    class_names = [LABEL_NAMES[i] for i in range(5)]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i in range(5):
        class_mask = predicted_labels == i
        if np.sum(class_mask) > 0:
            class_probs = max_probs[class_mask]
            axes[1, 0].hist(class_probs, bins=15, alpha=0.6, color=colors[i], 
                           label=f'{class_names[i]} (평균: {np.mean(class_probs):.3f})', density=True)
    
    axes[1, 0].set_title('클래스별 예측 확신도 분포', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('확신도 (최대 확률)', fontsize=12)
    axes[1, 0].set_ylabel('밀도', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 확신도 임계값별 정확도
    thresholds = np.arange(0.3, 1.0, 0.05)
    accuracies = []
    sample_counts = []
    
    for thresh in thresholds:
        high_conf_mask = max_probs >= thresh
        if np.sum(high_conf_mask) > 0:
            high_conf_correct = correct_mask[high_conf_mask]
            accuracy = np.sum(high_conf_correct) / len(high_conf_correct)
            accuracies.append(accuracy)
            sample_counts.append(np.sum(high_conf_mask))
        else:
            accuracies.append(0)
            sample_counts.append(0)
    
    ax_twin = axes[1, 1].twinx()
    
    line1 = axes[1, 1].plot(thresholds, accuracies, 'b-o', label='정확도', linewidth=2)
    line2 = ax_twin.plot(thresholds, sample_counts, 'r-s', label='샘플 수', linewidth=2)
    
    axes[1, 1].set_title('확신도 임계값별 정확도 & 샘플 수', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('확신도 임계값', fontsize=12)
    axes[1, 1].set_ylabel('정확도', fontsize=12, color='blue')
    ax_twin.set_ylabel('샘플 수', fontsize=12, color='red')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 범례 통합
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 확신도 분포 차트가 {save_path}에 저장되었습니다.")
    plt.show()
    
    # 확신도 통계 출력
    print(f"\n📊 확신도 분석 결과:")
    print(f"   • 전체 평균 확신도: {np.mean(max_probs):.4f}")
    print(f"   • 정분류 평균 확신도: {np.mean(correct_probs):.4f}")
    print(f"   • 오분류 평균 확신도: {np.mean(incorrect_probs):.4f}")
    print(f"   • 확신도 차이: {np.mean(correct_probs) - np.mean(incorrect_probs):+.4f}")
    
    # 높은 확신도로 틀린 케이스 분석
    high_conf_wrong = (max_probs > 0.8) & (~correct_mask)
    if np.sum(high_conf_wrong) > 0:
        print(f"   • 높은 확신도(>0.8)로 틀린 케이스: {np.sum(high_conf_wrong)}개")
    
    # 낮은 확신도로 맞춘 케이스 분석
    low_conf_correct = (max_probs < 0.5) & correct_mask
    if np.sum(low_conf_correct) > 0:
        print(f"   • 낮은 확신도(<0.5)로 맞춘 케이스: {np.sum(low_conf_correct)}개")

def analyze_misclassifications(results, top_k=10, save_misclassified=True):
    """오분류 사례 분석 및 format 저장"""
    print(f"\n🔍 오분류 분석 (상위 {top_k}개 패턴)")
    print("-" * 60)
    
    # 오분류 패턴 수집
    misclass_patterns = defaultdict(int)
    misclassified_data = defaultdict(list)  # 오분류된 케이스의 상세 정보
    
    for idx, (true_label, pred_label, format_name) in enumerate(zip(results['labels'], results['predictions'], results['formats'])):
        if true_label != pred_label:
            pattern = f"{LABEL_NAMES[true_label]} → {LABEL_NAMES[pred_label]}"
            misclass_patterns[pattern] += 1
            
            # 오분류된 케이스 상세 정보 저장 (softmax 값 포함)
            softmax_probs = results['logits'][idx]  # 해당 케이스의 softmax 확률
            misclassified_data[pattern].append({
                'format': format_name,
                'true_label': true_label,
                'pred_label': pred_label,
                'true_label_name': LABEL_NAMES[true_label],
                'pred_label_name': LABEL_NAMES[pred_label],
                'softmax_probs': softmax_probs,
                'pred_confidence': softmax_probs[pred_label],  # 예측된 클래스의 확률
                'true_confidence': softmax_probs[true_label],  # 실제 정답 클래스의 확률
                'confidence_diff': softmax_probs[pred_label] - softmax_probs[true_label]  # 확률 차이
            })
    
    # 상위 패턴 출력
    sorted_patterns = sorted(misclass_patterns.items(), key=lambda x: x[1], reverse=True)
    
    for i, (pattern, count) in enumerate(sorted_patterns[:top_k]):
        percentage = (count / len(results['predictions'])) * 100
        print(f"{i+1:2d}. {pattern:<50} : {count:4d}회 ({percentage:.2f}%)")
    
    # 🔬 Softmax 확률 분포 분석
    print(f"\n🔬 상위 {top_k}개 패턴 Softmax 확률 분포 분석")
    print("=" * 100)
    
    softmax_analysis = {}
    
    for i, (pattern, count) in enumerate(sorted_patterns[:top_k]):
        print(f"\n[패턴 {i+1}] {pattern} ({count}개 케이스)")
        print("-" * 80)
        
        cases = misclassified_data[pattern]
        
        # 확률 통계 계산
        pred_confidences = [case['pred_confidence'] for case in cases]
        true_confidences = [case['true_confidence'] for case in cases]
        confidence_diffs = [case['confidence_diff'] for case in cases]
        
        # 통계 정보
        stats = {
            'count': count,
            'pred_conf_mean': np.mean(pred_confidences),
            'pred_conf_std': np.std(pred_confidences),
            'pred_conf_min': np.min(pred_confidences),
            'pred_conf_max': np.max(pred_confidences),
            'true_conf_mean': np.mean(true_confidences),
            'true_conf_std': np.std(true_confidences),
            'true_conf_min': np.min(true_confidences),
            'true_conf_max': np.max(true_confidences),
            'diff_mean': np.mean(confidence_diffs),
            'diff_std': np.std(confidence_diffs),
            'high_confidence_wrong': sum(1 for conf in pred_confidences if conf > 0.7),  # 높은 확신도로 틀린 케이스
            'low_confidence_wrong': sum(1 for conf in pred_confidences if conf < 0.4),   # 낮은 확신도로 틀린 케이스
        }
        
        softmax_analysis[pattern] = stats
        
        print(f"📊 예측 클래스 확률 통계:")
        print(f"   평균: {stats['pred_conf_mean']:.4f} ± {stats['pred_conf_std']:.4f}")
        print(f"   범위: {stats['pred_conf_min']:.4f} ~ {stats['pred_conf_max']:.4f}")
        
        print(f"📊 실제 클래스 확률 통계:")
        print(f"   평균: {stats['true_conf_mean']:.4f} ± {stats['true_conf_std']:.4f}")
        print(f"   범위: {stats['true_conf_min']:.4f} ~ {stats['true_conf_max']:.4f}")
        
        print(f"📊 확률 차이 (예측 - 실제):")
        print(f"   평균: {stats['diff_mean']:+.4f} ± {stats['diff_std']:.4f}")
        
        print(f"🎯 확신도 분석:")
        print(f"   높은 확신도(>0.7)로 틀린 케이스: {stats['high_confidence_wrong']}/{count} ({stats['high_confidence_wrong']/count*100:.1f}%)")
        print(f"   낮은 확신도(<0.4)로 틀린 케이스: {stats['low_confidence_wrong']}/{count} ({stats['low_confidence_wrong']/count*100:.1f}%)")
        
        # 확률 분포 구간별 분석
        prob_bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
        print(f"📈 예측 확률 구간별 분포:")
        for low, high in prob_bins:
            count_in_bin = sum(1 for conf in pred_confidences if low <= conf < high)
            percentage_in_bin = count_in_bin / count * 100
            print(f"   {low:.1f}~{high:.1f}: {count_in_bin:3d}개 ({percentage_in_bin:5.1f}%)")
    
    # 텍스트 파일로 저장
    if save_misclassified:
        print(f"\n💾 오분류 케이스를 텍스트 파일로 저장합니다...")
        
        # Softmax 분석 결과 저장
        save_file_path = os.path.join(EXPERIMENTS_DIR, 'misclassified_softmax_analysis.txt')
        with open(save_file_path, 'w', encoding='utf-8') as f:
            f.write("오분류 케이스 Softmax 확률 분포 분석\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"총 예측 수: {len(results['predictions'])}\n")
            f.write(f"총 오분류 수: {np.sum(results['labels'] != results['predictions'])}\n")
            f.write(f"전체 정확도: {results['overall_accuracy']:.4f}\n\n")
            
            for i, (pattern, count) in enumerate(sorted_patterns[:top_k]):
                stats = softmax_analysis[pattern]
                f.write(f"\n{'='*60}\n")
                f.write(f"[패턴 {i+1}] {pattern} ({count}개 케이스)\n")
                f.write(f"{'='*60}\n")
                
                f.write(f"\n📊 예측 클래스 확률 통계:\n")
                f.write(f"   평균: {stats['pred_conf_mean']:.4f} ± {stats['pred_conf_std']:.4f}\n")
                f.write(f"   범위: {stats['pred_conf_min']:.4f} ~ {stats['pred_conf_max']:.4f}\n")
                
                f.write(f"\n📊 실제 클래스 확률 통계:\n")
                f.write(f"   평균: {stats['true_conf_mean']:.4f} ± {stats['true_conf_std']:.4f}\n")
                f.write(f"   범위: {stats['true_conf_min']:.4f} ~ {stats['true_conf_max']:.4f}\n")
                
                f.write(f"\n📊 확률 차이 (예측 - 실제):\n")
                f.write(f"   평균: {stats['diff_mean']:+.4f} ± {stats['diff_std']:.4f}\n")
                
                f.write(f"\n🎯 확신도 분석:\n")
                f.write(f"   높은 확신도(>0.7)로 틀린 케이스: {stats['high_confidence_wrong']}/{count} ({stats['high_confidence_wrong']/count*100:.1f}%)\n")
                f.write(f"   낮은 확신도(<0.4)로 틀린 케이스: {stats['low_confidence_wrong']}/{count} ({stats['low_confidence_wrong']/count*100:.1f}%)\n")
                
                # 상세 케이스별 정보
                f.write(f"\n📋 상세 케이스별 확률 정보:\n")
                f.write(f"{'No.':<4} {'Format':<30} {'예측확률':<8} {'실제확률':<8} {'차이':<8}\n")
                f.write("-" * 60 + "\n")
                
                cases = misclassified_data[pattern]
                # 예측 확률이 높은 순으로 정렬
                sorted_cases = sorted(cases, key=lambda x: x['pred_confidence'], reverse=True)
                
                for j, case in enumerate(sorted_cases[:20]):  # 상위 20개만 표시
                    f.write(f"{j+1:<4} {case['format']:<30} {case['pred_confidence']:<8.4f} {case['true_confidence']:<8.4f} {case['confidence_diff']:+8.4f}\n")
                
                if len(sorted_cases) > 20:
                    f.write(f"... (총 {len(sorted_cases)}개 중 상위 20개만 표시)\n")
        
        print(f"   ✅ {save_file_path}: Softmax 확률 분포 상세 분석")
   

# 메인 실험 실행 함수
def run_comprehensive_evaluation(model, dataloader, device, save_plots=True):
    """종합적인 모델 평가 실행"""
    print("🚀 종합 모델 성능 평가를 시작합니다...")
    
    # 성능 지표 계산
    results = evaluate_model_performance(model, dataloader, device)
    
    # 보고서 출력
    print_performance_report(results)
    
    # 오분류 분석
    analyze_misclassifications(results)
    
    if save_plots:
        # 기존 시각화
        plot_confusion_matrix(results['confusion_matrix'])
        plot_class_performance(results)
        
        # 새로운 시각화 추가
        plot_tsne_visualization(results, input_type='features', max_samples=5000)  # Feature vectors 사용 (권장)
        plot_tsne_visualization(results, input_type='raw_logits', max_samples=5000)  # Raw logits 사용
        plot_confidence_distribution(results)
        
        # 비교 분석 (선택사항)
        compare_tsne_inputs(results)
    
    # 요약 통계
    print(f"\n📌 요약 통계:")
    print(f"   • 전체 정확도: {results['overall_accuracy']:.4f}")
    print(f"   • 평균 F1-점수: {np.mean(results['f1_score']):.4f}")
    print(f"   • 가중 F1-점수: {np.average(results['f1_score'], weights=results['support']):.4f}")
    print(f"   • 총 테스트 샘플: {len(results['predictions'])}")
    print(f"   • 총 오분류: {np.sum(results['labels'] != results['predictions'])}")
    
    return results

# 기존 실험 코드를 확장
def run_basic_test(model, dataloader, device):
    """기본 테스트 (기존 코드)"""
    print("기본 테스트를 실행합니다...")
    
    for batch in dataloader:
        landmarks = batch['landmarks'].to(device)
        label = batch['label'].long().to(device)
        logits = model(landmarks)
        _, predicted = torch.max(logits.data, 1)
        print("예측값:", predicted)
        print("실제값:", label)
        break

def test_experiment():
    model = PointNetClassifier(num_classes=5)
    
    # 모델 로드
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Validation 데이터셋 로드 (이미 검증용 데이터)
    test_dataset = FocusDataset_multi(base_path)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Validation 데이터셋 크기: {len(test_dataset)}")
    print(f"디바이스: {device}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 기본 테스트 (기존 코드)
    print("\n" + "="*50)
    print("1. 기본 테스트")
    print("="*50)
    run_basic_test(model, test_dataloader, device)
    
    # 종합적인 성능 평가
    print("\n" + "="*50)  
    print("2. 종합 성능 평가")
    print("="*50)
    results = run_comprehensive_evaluation(model, test_dataloader, device, save_plots=True)
    
    # 추가 분석 (선택사항)
    print("\n" + "="*50)
    print("3. 추가 분석")
    print("="*50)
    
    # 클래스 불균형 분석
    class_distribution = results['class_total']
    total_samples = sum(class_distribution.values())
    
    print("📊 클래스 분포 분석:")
    for i in range(5):
        count = class_distribution.get(i, 0)
        percentage = (count / total_samples) * 100
        print(f"   {LABEL_NAMES[i]}: {count}개 ({percentage:.1f}%)")
    
    # 성능이 가장 좋은/나쁜 클래스
    f1_scores = results['f1_score']
    best_class = np.argmax(f1_scores)
    worst_class = np.argmin(f1_scores)
    
    print(f"\n🏆 성능이 가장 좋은 클래스: {LABEL_NAMES[best_class]} (F1: {f1_scores[best_class]:.4f})")
    print(f"🚨 성능이 가장 나쁜 클래스: {LABEL_NAMES[worst_class]} (F1: {f1_scores[worst_class]:.4f})")
    
    return results

if __name__ == "__main__":
    print("🚀 FocusSu AI 모델 성능 평가를 시작합니다...")
    
    # matplotlib 백엔드 설정
    import matplotlib

    
    # 실험 실행
    print("\n" + "="*60)
    results = test_experiment()
    
    print("\n" + "="*80)
    print("🎉 실험이 완료되었습니다!")
    print("생성된 파일:")
    print(f"  📊 시각화 (저장 위치: {EXPERIMENTS_DIR}/)")
    print("    - confusion_matrix.png: 혼동 행렬 히트맵")
    print("    - class_performance.png: 클래스별 성능 지표 차트")
    print("    - tsne_raw_logits_visualization.png: Raw logits 사용한 t-SNE 시각화")
    print("    - tsne_softmax_visualization.png: Softmax 확률 사용한 t-SNE 시각화")
    print("    - confidence_distribution.png: 예측 확신도 분포 분석")
    print("  📄 오분류 분석:")
    print("    - misclassified_softmax_analysis.txt: Softmax 확률 분포 상세 분석")
    print("="*80)