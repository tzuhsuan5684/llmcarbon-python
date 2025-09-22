# LLM 碳足跡計算工具

這是一個基於 **LLMCarbon 論文** 的命令列工具，可以快速估算大型語言模型（LLM）在訓練過程中的 **營運碳排** 與 **隱含碳排**。

---

## 📦 環境需求
- Python 3.x  
- Pandas  
  ```bash
  pip install pandas
  ```

---

## 📂 檔案結構
```
.
├── llmcarbon_calculator.py   # 主要的計算腳本
└── hardware.csv              # 計算隱含碳排所需的硬體數據
```

⚠️ **注意**：  
`hardware.csv` 文件目前基於 **Meta XLM 訓練設置** 的範例，您需要根據實際的伺服器配置修改此文件，以獲得精確的隱含碳排估算。

---

## 🚀 如何使用
1. 確保 `llmcarbon_calculator.py` 和 `hardware.csv` 在同一個資料夾下。  
2. 打開您的 **終端機 (Terminal)** 或 **命令提示字元 (Command Prompt)**。  
3. 執行以下指令。  

---

### 🔹 快速開始（使用預設值）
直接執行腳本，將會使用類似 **GPT-3 (175B)** 的預設參數進行計算：  

```bash
python llmcarbon_calculator.py
```

---

### 🔹 自訂參數
您可以透過添加不同的參數進行 **客製化計算**。  

範例：計算一個 **13B 參數的密集模型**，使用 **512 張 A100 GPU**：  

```bash
python llmcarbon_calculator.py \
    --model-type dense \
    --parameters-b 13 \
    --tokens-t 500 \
    --device A100 \
    --device-num 512 \
    --system-power-w 500 \
    --hardware-efficiency-perc 40 \
    --pue 1.1 \
    --co2-intensity-g-kwh 429
```

---

### 🔹 查看所有可用的參數
```bash
python llmcarbon_calculator.py --help
```

這將列出所有可設定的項目，包括：
- 模型設定  
- 硬體配置  
- 資料中心設定  

---

### 🔹 使用範例
```python
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

```

