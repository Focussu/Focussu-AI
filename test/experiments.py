import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import defaultdict
import pandas as pd
from torch.utils.data import DataLoader
from model.PointNet import PointNetClassifier
from train.data import FocusDataset_multi


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
    all_logits = []
    all_formats = []  # format 정보 저장
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    print("모델 평가를 시작합니다...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            landmarks = batch['landmarks'].to(device)
            labels = batch['label'].long().to(device)
            formats = batch['format']  # format 정보 가져오기
            
            # 예측 수행
            logits = model(landmarks)
            _, predicted = torch.max(logits.data, 1)
            
            # 결과 저장
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(torch.softmax(logits, dim=1).cpu().numpy())
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
    
    print(f"\n총 {len(all_predictions)}개 샘플 평가 완료")
    
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
        'logits': all_logits,
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

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Confusion Matrix 시각화"""
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

def plot_class_performance(results, save_path='class_performance.png'):
    """클래스별 성능 지표 시각화"""
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
        with open('misclassified_softmax_analysis.txt', 'w', encoding='utf-8') as f:
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
        
      
        
     
        
        print(f"   ✅ misclassified_softmax_analysis.txt: Softmax 확률 분포 상세 분석")
   

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
        # 시각화
        plot_confusion_matrix(results['confusion_matrix'])
        plot_class_performance(results)
    
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
    print("  📊 시각화:")
    print("    - confusion_matrix.png: 혼동 행렬 히트맵")
    print("    - class_performance.png: 클래스별 성능 지표 차트")
    print("  📄 오분류 분석:")
    print("    - misclassified_softmax_analysis.txt: Softmax 확률 분포 상세 분석")
    print("="*80)