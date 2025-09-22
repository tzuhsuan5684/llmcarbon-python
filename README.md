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
