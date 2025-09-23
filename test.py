# my_project.py
# 匯入我們建立的計算工具類別
from llmcarbon_calculator import TrainingCarbonCalculator, InferenceCarbonCalculator

# --- 為了讓類別可以運作，我們需要模擬命令列參數 ---
# argparse 在命令列環境下會自動建立 Namespace 物件，
# 在函式庫模式下，我們手動建立一個簡單的物件或字典來傳遞參數。
class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def main():
    """主程式：展示如何使用函式庫"""
    print("===== 範例 2: 計算線上服務單次推論的碳足跡 =====")
    
    # 設定推論場景的參數
    inference_args = Args(
        # LLM 模型設定
        model_type='dense',
        parameters_b=175,  # 假設一個 8B 參數的模型用於線上服務
        base_model_params_b=0,
        
        # 硬體與資料中心設定
        device='A100',
        device_num=8,       # 通常單次推論只使用 1 張 GPU
        system_power_w=550, # H100 伺服器功耗
        hardware_efficiency_perc=19.7, # 推論時效率可能較高
        pue=1.1,
        co2_intensity_g_kwh=429,
        
        # 推論專用 Token 數
        infer_tokens_k=5 # 處理一個 4000 (4K) tokens 的請求
    )

    try:
        # 建立推論計算器實例並執行計算
        infer_calculator = InferenceCarbonCalculator(inference_args)
        infer_results = infer_calculator.run()

        # 顯示結果
        print(f"單次推論碳足跡估算：")
        print(f"  - 預估執行時間: {infer_results['execution_seconds']:.4f} 秒")
        print(f"  - 總消耗電量:   {infer_results['total_energy_mwh']:.8f} MWh")
        print(f"  - 營運碳排:     {infer_results['operational_co2_t']:.8f} tCO₂eq")
        # 為了可讀性，轉換為克 (g)
        print(f"                  (相當於 {infer_results['operational_co2_t'] * 1_000_000:.4f} gCO₂eq)\n")

    except Exception as e:
        print(f"計算發生錯誤: {e}\n")


if __name__ == '__main__':
    main()