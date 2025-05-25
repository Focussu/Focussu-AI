import os, json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import glob
from pathlib import Path

# 상수 정의
BASE_PATH = '/shared_data/focussu/109.학습태도_및_성향_관찰_데이터/3.개방데이터/1.데이터/Validation'
META_PATH = os.path.join(BASE_PATH, '02.라벨링데이터')
OUTPUT_DIR = os.path.join(META_PATH, 'val_meta_parts')  # 실질 저장 경로 (폴더)

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_json(file):
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
            }
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Error processing {file}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error processing {file}: {str(e)}")
        return None

def process_chunk(chunk):
    return pd.DataFrame([r for r in chunk if r is not None])

if __name__ == "__main__":
    # JSON 파일 목록 가져오기
    meta_files = glob.glob(os.path.join(META_PATH, '*/*.json'))
    print(f"Found {len(meta_files)} JSON files to process")

    CHUNK_SIZE = 1000

    with ProcessPoolExecutor() as executor:
        for i in range(0, len(meta_files), CHUNK_SIZE):
            chunk = meta_files[i:i + CHUNK_SIZE]

            results = list(tqdm(
                executor.map(parse_json, chunk),
                total=len(chunk),
                desc=f"Processing chunk {i//CHUNK_SIZE + 1}"
            ))

            df_chunk = process_chunk(results)

            # 직접 실제 저장 디렉토리에 저장 (append 불필요)
            output_path = os.path.join(OUTPUT_DIR, f'part_{i//CHUNK_SIZE:04d}.parquet')
            df_chunk.to_parquet(output_path)

            # 메모리 해제
            del df_chunk
            del results

    print(f"✅ Done! Parquet chunks saved in: {OUTPUT_DIR}")
