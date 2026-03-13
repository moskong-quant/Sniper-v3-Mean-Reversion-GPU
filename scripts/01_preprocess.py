import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm

# --- SETTINGS ---
RAW_CSV_PATH = '/app/data/raw/EURUSD_M1_Combined_2015_2026.csv' 
PROCESSED_PATH = '/app/data/processed/eurusd_m1.parquet'

def preprocess_data():
    start_time = time.time()
    
    if not os.path.exists(RAW_CSV_PATH):
        print(f"❌ Error: ไม่พบไฟล์ที่ {RAW_CSV_PATH}")
        return

    print(f"🚀 เริ่มต้นอ่านไฟล์ EURUSD M1 (Timestamp Column detected)")

    dtype_dict = {
        'Open': 'float32', 
        'High': 'float32', 
        'Low': 'float32', 
        'Close': 'float32', 
        'Volume': 'float32'
    }
    
    chunks = []
    try:
        # เจาะจงใช้ 'Timestamp' เป็นตัวแปรเวลา
        reader = pd.read_csv(RAW_CSV_PATH, 
                            parse_dates=['Timestamp'], 
                            index_col='Timestamp', 
                            dtype=dtype_dict, 
                            chunksize=500000, 
                            engine='c')
        
        for chunk in tqdm(reader, desc="Loading Data"):
            chunks.append(chunk)
            
        df = pd.concat(chunks)
        
        # ปรับชื่อคอลัมน์ให้เป็นพิมพ์เล็กทั้งหมด (Standardization)
        df.columns = [c.lower() for c in df.columns]
        df.index.name = 'datetime'
        
        # เรียงลำดับเวลาให้ถูกต้อง
        df.sort_index(inplace=True)
        
        print(f"✅ โหลดและเรียงข้อมูลสำเร็จ: {len(df):,} แถว")

        # --- Basic Feature Engineering ---
        # Forward fill ข้อมูลที่หายไป (ถ้ามี)
        df.ffill(inplace=True)
        
        # คำนวณ Log Return สำหรับงานสถิติ
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # ลบค่า NaN
        df.dropna(inplace=True)

        # บันทึกเป็น Parquet (Snappy compression เพื่อความสมดุลระหว่างความเร็วและขนาดไฟล์)
        os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
        df.to_parquet(PROCESSED_PATH, engine='pyarrow', compression='snappy')
        
        print(f"📦 บันทึก Parquet สำเร็จ: {PROCESSED_PATH}")
        print(f"⚡ ใช้เวลาทั้งหมด: {time.time() - start_time:.2f} วินาที")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    preprocess_data()