import os
import time
import pandas as pd
import numpy as np
from numba import cuda
import xgboost as xgb
import math
from core.kernels import calc_features_gpu
from config.settings import XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE

# --- CONFIGURATION ---
TIMEFRAMES = ['1min', '5min', '15min', '30min']  # ตัด H1 ออกตามสั่งครับ
BASE_DATA = '/app/data/processed/eurusd_m1.parquet'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_multi_tf():
    print(f"📂 Loading Base M1 Data...")
    df_base = pd.read_parquet(BASE_DATA)

    for tf in TIMEFRAMES:
        print(f"\n--- 🔄 Processing Timeframe: {tf} ---")
        
        # 1. Resampling
        if tf == '1min':
            df = df_base.copy()
        else:
            df = df_base.resample(tf).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()

        # 2. GPU Feature Engineering
        n = len(df)
        close_gpu = cuda.to_device(df['close'].values.astype(np.float64))
        high_gpu = cuda.to_device(df['high'].values.astype(np.float64))
        low_gpu = cuda.to_device(df['low'].values.astype(np.float64))
        z_out = cuda.device_array(n, dtype=np.float64)
        atr_out = cuda.device_array(n, dtype=np.float64)
        
        calc_features_gpu[(n+255)//256, 256](close_gpu, high_gpu, low_gpu, 20, z_out, atr_out)
        df['z_score'] = z_out.copy_to_host()
        df['atr'] = atr_out.copy_to_host()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # 3. Target Labeling (ถือครอง 10 แท่งของ TF นั้นๆ)
        df['target'] = np.where(df['close'].shift(-10) < df['close'], 1, 0)
        
        # 4. เตรียมข้อมูลสำหรับ ML (Alignment Check)
        features = ['z_score', 'atr', 'hour', 'day_of_week']
        target_col = 'target'
        
        # สร้าง DataFrame ย่อยที่รวมทั้ง Features และ Target เพื่อทำการ dropna พร้อมกัน
        df_ml = df[features + [target_col]].dropna()
        
        X = df_ml[features]
        y = df_ml[target_col]
        
        # 5. การแบ่งข้อมูล (Train/Test Split)
        # สร้าง mask สำหรับกรองข้อมูลตามวันเวลาจาก index ของ X โดยตรง
        train_mask = X.index < '2023-01-01'
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        # เช็คความถูกต้องก่อนเข้า Train
        print(f"📊 {tf} Alignment: X_train {X_train.shape}, y_train {y_train.shape}")

        if len(X_train) == 0:
            print(f"⚠️ Warning: {tf} ไม่มีข้อมูลสำหรับเทรนในช่วงเวลาที่กำหนด")
            continue

        print(f"🧠 Training XGBoost on GPU for {tf}...")
        model = xgb.XGBClassifier(
        tree_method='hist', 
        device='cuda', 
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE
        )

        model.fit(X_train, y_train)

        # 5. Save Model with Professional Naming
        model_dir = os.path.join(PROJECT_ROOT, 'models', tf.upper())
        os.makedirs(model_dir, exist_ok=True)
        model_name = f"MREV_{tf.upper()}_v1.json"
        model.save_model(os.path.join(model_dir, model_name))
        
        print(f"✅ Model Saved: {model_dir}/{model_name}")

if __name__ == "__main__":
    train_multi_tf()