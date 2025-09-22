# -*- coding: utf-8 -*-

# 1. 匯入我們需要的主要計算函式和 argparse.Namespace
from llmcarbon_calculator import calculate_carbon_footprint
import argparse

# 2. 手動建立一個參數物件 (Namespace)，模擬命令列輸入
#    您可以直接修改這裡的數值來進行不同的計算
args = argparse.Namespace(
    # --- LLM 模型設定 ---
    model_type='dense',
    parameters_b=13,
    base_model_params_b=2.3, # 只有 model_type='MoE' 時會用到
    tokens_t=500,
    
    # --- 硬體與資料中心設定 ---
    device='A100',
    device_num=512,
    system_power_w=500,
    hardware_efficiency_perc=40,
    pue=1.1,
    co2_intensity_g_kwh=429
)

# 3. 呼叫計算函式，並傳入我們建立的參數物件
try:
    results = calculate_carbon_footprint(args)
    
    # 4. 印出結果
    print("計算成功！結果如下：")
    print("-" * 30)
    print(f"總碳足跡: {results['total_co2_t']:.2f} tCO₂eq")
    print(f"  - 營運碳排: {results['operational_co2_t']:.2f} tCO₂eq")
    print(f"  - 隱含碳排: {results['embodied_co2_t']:.2f} tCO₂eq")
    print(f"預估訓練天數: {results['training_days']:.2f} 天")
    print(f"總消耗電量: {results['total_energy_mwh']:.2f} MWh")

except FileNotFoundError as e:
    print(f"錯誤：找不到必要的數據文件。請確保 'hardware.csv' 與您的腳本在同一個資料夾中。")
    print(f"詳細錯誤訊息: {e}")
except Exception as e:
    print(f"計算過程中發生錯誤: {e}")
