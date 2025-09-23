### LLM 碳足跡計算工具 (LLM Carbon Calculator)
這是一個基於命令列的 Python 工具，用於估算大型語言模型 (LLM) 在訓練 (training) 和推論 (inference) 階段的營運碳足跡。計算方法主要參考了 LLMCarbon: Modeling the End-to-End Carbon Footprint of Large Language Models 研究論文中的簡化模型，讓使用者可以快速評估不同模型大小、硬體配置和使用情境下的碳排放量。
### ✨ 主要功能
* 🧮 雙模式計算：可分別計算模型訓練與推論兩個不同階段的碳足跡。  
* 🤖 支援不同模型架構：支援標準的密集模型 (Dense) 與混合專家模型 (MoE)。  
* 🔌 硬體預設：內建多種主流 AI 硬體（如 NVIDIA V100, H100, A100, Google TPUv3, TPUv4）的峰值算力參數。  
* 🔧 高度可自訂：幾乎所有參數，從模型大小、硬體效率、裝置數量到資料中心的 PUE 值與電網碳強度，皆可透過命令列參數進行調整。
* 🖥️ 簡單易用：無需安裝額外套件，只需一個 Python 檔案即可執行。

### ⚙️ 如何使用
此工具僅使用 Python 標準函式庫，無需安裝任何額外套件。
1. 確保您的環境中已安裝 Python 3。
2. 將程式碼儲存為 `llm_carbon_calculator.py`。
3. 打開您的終端機 (Terminal) 或命令提示字元 (Command Prompt)，並切換到檔案所在的目錄。

#### 基本指令結構
``` python
python llm_carbon_calculator.py [mode] [options]
```
* `[mode]`：必須選擇 `train` (訓練) 或 `infer` (推論)。
* `[options]`：用來設定計算所需的各項參數。  
#### 範例  

1. 計算模型訓練的碳足跡  
假設我們要估算一個 1750 億參數的密集模型，使用 300 兆 (Trillion) Tokens 進行訓練，並部署在 1024 張 V100 GPU 上的碳排放：
``` bash
python llmcarbon_calculator.py train \
    --parameters-b 175 \
    --train-tokens-t 300 \
    --device V100 \
    --device-num 1024 \
    --hardware-efficiency-perc 25
```
2. 計算模型推論的碳足跡假設我們要估算一個 70 億參數的模型，處理 500 萬 (5000 K) 個 Tokens 的推論請求，並部署在 8 張 A100 GPU 上的碳排放：
``` bash
python llmcarbon_calculator.py infer \
    --parameters-b 70 \
    --infer-tokens-k 5000 \
    --device A100 \
    --device-num 8
```
#### 參數說明  
您可以使用 `-h` 或 `--help` 來查看所有可用的參數。
```bash
# 查看通用說明與模式選項
python llmcarbon_calculator.py -h

# 查看 'train' 模式的專屬說明
python llmcarbon_calculator.py train -h
```

#### 通用參數 (訓練與推論共用)
* `--model-type`: 模型類型，可選 dense 或 MoE (預設: dense)。　　
* `--parameters-b`: 模型總參數數量 (單位: 十億 B) (預設: 175)。
* `--base-model-params-b`: 若為 MoE 模型，其基礎模型的參數數量 (預設: 2.3)。
* `--device`: 運算裝置類型，可選 V100, H100, TPUv3, TPUv4, A100 (預設: V100)。
* `--device-num`: 運算裝置總數量 (預設: 10000)。
* `--system-power-w`: 單一裝置的平均系統功耗 (W) (預設: 330)。
* `--hardware-efficiency-perc`: 硬體效率 (%)，代表實際達到的算力與理論峰值的比例 (預設: 19.7)。
* `--pue`: 資料中心的 PUE (Power Usage Effectiveness) 值 (預設: 1.1)。
* `--co2-intensity-g-kwh`: 電網碳強度 (gCO₂eq/kWh)，代表每度電產生的碳排放量 (預設: 429, 美國平均)。
#### 專屬參數
* `--train-tokens-t`: [訓練專用] 處理的 Token 總數 (單位: 兆 T) (預設: 300)。
* `--infer-tokens-k`: [推論專用] 處理的 Token 總數 (單位: 千 K) (預設: 5)。

### 🚀 進階使用：在其他 Python 腳本中引用
除了直接透過命令列執行，您也可以將此工具的計算類別 (TrainingCarbonCalculator 和 InferenceCarbonCalculator) 匯入到您自己的 Python 專案中，以便進行更複雜的分析或整合。

首先，請確保 llm_carbon_calculator.py 與您的腳本在同一個目錄下，或者已將其安裝為一個模組。

以下是如何直接呼叫計算類別的範例：
```python
# 假設您的計算工具檔案名稱為 llmcarbon_calculator.py
# 以下程式碼需要與 llmcarbon_calculator.py 在同一目錄下

import argparse
from llm_carbon_calculator import TrainingCarbonCalculator, InferenceCarbonCalculator

# --- 範例 1: 計算訓練碳排 ---

# 1. 模擬命令列參數
# 您需要建立一個包含所有必要參數的 argparse.Namespace 物件
train_args = argparse.Namespace(
    # LLM 模型設定
    model_type='dense',
    parameters_b=175,
    base_model_params_b=2.3, # 即使是 dense 模型也需提供預設值
    
    # 硬體與資料中心設定
    device='V100',
    device_num=1024,
    system_power_w=400,
    hardware_efficiency_perc=25,
    pue=1.1,
    co2_intensity_g_kwh=429,
    
    # 訓練專用參數
    train_tokens_t=300
)

# 2. 建立計算器實例並執行
train_calculator = TrainingCarbonCalculator(train_args)
train_results = train_calculator.run()

print("--- 訓練碳排估算 ---")
print(f"預估訓練時間: {train_results['execution_days']:.2f} 天")
print(f"總消耗電量: {train_results['total_energy_mwh']:.2f} MWh")
print(f"營運碳排: {train_results['operational_co2_t']:.4f} tCO₂eq\n")


# --- 範例 2: 計算推論碳排 ---

# 1. 模擬推論所需的參數
infer_args = argparse.Namespace(
    model_type='dense',
    parameters_b=70,
    base_model_params_b=2.3,
    
    device='A100',
    device_num=8,
    system_power_w=330,
    hardware_efficiency_perc=30,
    pue=1.1,
    co2_intensity_g_kwh=429,
    
    # 推論專用參數
    infer_tokens_k=5000
)

# 2. 建立實例並執行
infer_calculator = InferenceCarbonCalculator(infer_args)
infer_results = infer_calculator.run()

print("--- 推論碳排估算 ---")
print(f"預估執行時間: {infer_results['execution_seconds']:.2f} 秒")
print(f"營運碳排: {infer_results['operational_co2_t']:.6f} tCO₂eq")
```

### 🔬 計算方法論
本工具的計算核心基於以下簡化公式：
1. 計算總運算量 (FLOPs)：
    * 訓練：FLOPs ≈ 6 * 模型參數 * 訓練 Tokens 總數
    * 推論：FLOPs ≈ 2 * 模型參數 * 推論 Tokens 總數
2. 估算執行時間：
    * 總執行時間 = 總 FLOPs / (裝置數量 * 單一裝置的有效算力)
    * 其中，有效算力 = 硬體理論峰值算力 * 硬體效率
3. 計算總耗電量：
    * 總耗電量 (kWh) = (裝置數量 * 單裝置功耗) * 總執行時間 * PUE
4. 計算營運碳排：
    * 總碳排 (tCO₂eq) = 總耗電量 (kWh) * 電網碳強度
