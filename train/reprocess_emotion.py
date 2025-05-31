import os, json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import glob
from pathlib import Path
import time

# 상수 정의
BASE_PATH = '/shared_data/focussu/109.학습태도_및_성향_관찰_데이터/3.개방데이터/1.데이터/Training'
META_PATH = os.path.join(BASE_PATH, '02.라벨링데이터')
OUTPUT_DIR = os.path.join(META_PATH, 'train_meta_with_emotion')  # emotion 포함된 파일 저장 경로

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_json_with_emotion(file):
    """
    JSON 파일에서 모든 필요한 정보를 한 번에 추출 (emotion 포함)
    """
    try:
        if not os.path.exists(file):
            return None
        with open(file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            return {
                'filename': metadata['이미지']['filename'],
                'format': metadata['이미지']['format'],
                'idx': metadata['이미지']['timeline']['id'],
                'category_id': metadata['이미지']['category']['id'],
                'emotion': metadata['이미지']['emotion']  # emotion 추가
            }
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Error processing {file}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error processing {file}: {str(e)}")
        return None

def process_chunk(chunk):
    """청크 데이터를 DataFrame으로 변환"""
    return pd.DataFrame([r for r in chunk if r is not None])

def reprocess_with_emotion():
    """
    효율적인 방식: 처음부터 다시 처리 (emotion 포함)
    - 각 JSON 파일을 한 번만 읽음
    - 병렬 처리로 성능 최적화
    """
    print("🚀 emotion 열을 포함한 전체 재처리 시작...")
    start_time = time.time()
    
    # JSON 파일 목록 가져오기
    meta_files = glob.glob(os.path.join(META_PATH, '*/*.json'))
    print(f"처리할 JSON 파일 수: {len(meta_files):,}")

    if not meta_files:
        print("❌ 처리할 JSON 파일이 없습니다!")
        return

    CHUNK_SIZE = 1000

    with ProcessPoolExecutor() as executor:
        for i in range(0, len(meta_files), CHUNK_SIZE):
            chunk = meta_files[i:i + CHUNK_SIZE]

            results = list(tqdm(
                executor.map(parse_json_with_emotion, chunk),
                total=len(chunk),
                desc=f"처리 중인 청크 {i//CHUNK_SIZE + 1}/{(len(meta_files) + CHUNK_SIZE - 1)//CHUNK_SIZE}"
            ))

            df_chunk = process_chunk(results)
            
            if not df_chunk.empty:
                # 저장
                output_path = os.path.join(OUTPUT_DIR, f'part_{i//CHUNK_SIZE:04d}.parquet')
                df_chunk.to_parquet(output_path)
                print(f"✅ 청크 {i//CHUNK_SIZE + 1} 저장 완료 ({len(df_chunk)} 행)")

            # 메모리 해제
            del df_chunk
            del results

    elapsed_time = time.time() - start_time
    print(f"✅ 완료! emotion 포함된 parquet 파일들이 저장됨: {OUTPUT_DIR}")
    print(f"⏱️  총 처리 시간: {elapsed_time:.2f}초")

if __name__ == "__main__":
    print("=" * 60)
    print("📊 Emotion 열을 포함한 데이터 전처리")
    print("=" * 60)
    print()
    print("💡 처리 방식: JSON 파일을 한 번만 읽어 효율적으로 처리")
    print(f"📁 출력 경로: {OUTPUT_DIR}")
    print()
    
    reprocess_with_emotion() 