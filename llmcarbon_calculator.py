# -*- coding: utf-8 -*-
import argparse

# --- 基礎設定與通用邏輯 ---

class LLMCarbonCalculatorBase:
    """
    計算模型的基礎類別，包含共享的參數與設定。
    """
    # 預設硬體參數 (參考 LLMCarbon 論文 Table 4 & 相關資料)
    HARDWARE_PRESETS = {
        'V100': {'peak_tflops': 125},
        'H100': {'peak_tflops': 1979},
        'TPUv3': {'peak_tflops': 123},
        'TPUv4': {'peak_tflops': 275},
        'A100': {'peak_tflops': 312}, 
    }

    def __init__(self, args):
        self.args = args
        self.preset = self.HARDWARE_PRESETS.get(args.device)
        if not self.preset:
            raise ValueError(
                f"不支援的硬體裝置: {args.device}。"
                f"請從 {list(self.HARDWARE_PRESETS.keys())} 中選擇。"
            )

    def _calculate_carbon_emission(self, execution_days):
        """
        根據執行天數計算總耗電量與碳排 (通用邏輯)。
        """
        total_power_kw = (self.args.system_power_w * self.args.device_num) / 1000
        total_energy_kwh = total_power_kw * (execution_days * 24) * self.args.pue
        total_energy_mwh = total_energy_kwh / 1000

        carbon_intensity_t_per_kwh = self.args.co2_intensity_g_kwh / 1_000_000
        operational_co2_t = total_energy_kwh * carbon_intensity_t_per_kwh
        
        return {
            "operational_co2_t": operational_co2_t,
            "total_energy_mwh": total_energy_mwh
        }

    def run(self):
        """主執行方法，由子類別實現"""
        raise NotImplementedError("子類別必須實現 run() 方法")


# --- 訓練碳排計算邏輯 ---

class TrainingCarbonCalculator(LLMCarbonCalculatorBase):
    """
    專門計算 LLM 訓練階段的營運碳排。
    """
    def run(self):
        """執行訓練碳排計算"""
        # 根據論文公式 4: TC ≈ 6PD
        flop_multiplier = 6
        active_params_b = self.args.base_model_params_b if self.args.model_type == 'MoE' else self.args.parameters_b
        
        # 單位: P (Billion), D (Trillion)
        total_zettaflops = flop_multiplier * active_params_b * self.args.train_tokens_t
        total_flops = total_zettaflops * 1e21

        achieved_tflops_per_second = self.preset['peak_tflops'] * (self.args.hardware_efficiency_perc / 100)
        
        if achieved_tflops_per_second == 0 or self.args.device_num == 0:
            execution_days = 0
        else:
            execution_seconds = total_flops / (self.args.device_num * achieved_tflops_per_second * 1e12)
            execution_days = execution_seconds / (3600 * 24)

        results = self._calculate_carbon_emission(execution_days)
        results['execution_days'] = execution_days
        return results


# --- 推論碳排計算邏輯 ---

class InferenceCarbonCalculator(LLMCarbonCalculatorBase):
    """
    專門計算 LLM 推論階段的營運碳排。
    """
    def run(self):
        """執行推論碳排計算"""
        # 根據論文公式 5: IC ≈ 2PD
        flop_multiplier = 2
        active_params_b = self.args.base_model_params_b if self.args.model_type == 'MoE' else self.args.parameters_b

        # 單位轉換: P (Billion), D (Thousand) -> (1e9 * 1e3)
        # 為了與 ZettaFLOPs 對應，先將 K 轉為 T (除以 1e9)
        tokens_t = self.args.infer_tokens_k / 1_000_000_000
        
        total_zettaflops = flop_multiplier * active_params_b * tokens_t
        total_flops = total_zettaflops * 1e21

        achieved_tflops_per_second = self.preset['peak_tflops'] * (self.args.hardware_efficiency_perc / 100)

        if achieved_tflops_per_second == 0 or self.args.device_num == 0:
            execution_days = 0
        else:
            execution_seconds = total_flops / (self.args.device_num * achieved_tflops_per_second * 1e12)
            execution_days = execution_seconds / (3600 * 24)
            
        results = self._calculate_carbon_emission(execution_days)
        # 推論時間通常較短，轉換為秒或分鐘可能更合適
        results['execution_seconds'] = execution_days * 24 * 3600
        return results


# --- 命令列介面與主程式 ---

def main():
    """主程式進入點"""
    parser = argparse.ArgumentParser(
        description="LLM 營運碳足跡計算工具 (獨立 Token 參數版)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='mode', required=True, help='選擇計算模式: train (訓練) 或 infer (推論)')

    # --- 通用參數 ---
    def add_common_args(p):
        model_group = p.add_argument_group('1. LLM 模型設定')
        model_group.add_argument('--model-type', type=str, default='dense', choices=['dense', 'MoE'], help='模型類型 (預設: dense)')
        model_group.add_argument('--parameters-b', type=float, default=175, help='模型總參數數量 (單位: 十億 B) (預設: 175)')
        model_group.add_argument('--base-model-params-b', type=float, default=2.3, help='MoE 模型的基礎模型參數數量 (B) (預設: 2.3)')
        
        hw_group = p.add_argument_group('2. 硬體與資料中心設定')
        hw_group.add_argument('--device', type=str, default='V100', choices=LLMCarbonCalculatorBase.HARDWARE_PRESETS.keys(), help=f'運算裝置類型 (預設: V100)')
        hw_group.add_argument('--device-num', type=int, default=10000, help='運算裝置總數量 (預設: 10000)')
        hw_group.add_argument('--system-power-w', type=float, default=330, help='單一裝置的平均系統功耗 (W) (預設: 330)')
        hw_group.add_argument('--hardware-efficiency-perc', type=float, default=19.7, help='硬體效率 (%%) (預設: 19.7)')
        hw_group.add_argument('--pue', type=float, default=1.1, help='資料中心的 PUE 值 (預設: 1.1)')
        hw_group.add_argument('--co2-intensity-g-kwh', type=float, default=429, help='電網碳強度 (gCO₂eq/kWh) (預設: 429, 美國平均)')

    # --- 建立 'train' 子命令與其專屬參數 ---
    parser_train = subparsers.add_parser('train', help='計算訓練階段的碳足跡')
    add_common_args(parser_train)
    parser_train.add_argument('--train-tokens-t', type=float, default=300, help='[訓練專用] 處理的 Token 總數 (單位: 兆 T) (預設: 300)')

    # --- 建立 'infer' 子命令與其專屬參數 ---
    parser_infer = subparsers.add_parser('infer', help='計算推論階段的碳足跡')
    add_common_args(parser_infer)
    parser_infer.add_argument('--infer-tokens-k', type=float, default=5, help='[推論專用] 處理的 Token 總數 (單位: 千 K) (預設: 5)')
    
    args = parser.parse_args()

    # --- 執行計算與輸出 ---
    try:
        if args.mode == 'train':
            calculator = TrainingCarbonCalculator(args)
            results = calculator.run()
            
            print("\n" + "="*50)
            print("LLM 訓練 (Training) 碳足跡計算結果".center(50))
            print("="*50)
            print(f"  - 預估訓練時間: {results['execution_days']:.2f} 天")
            print(f"  - 總消耗電量:   {results['total_energy_mwh']:.2f} MWh")
            print(f"  - 營運碳排:     {results['operational_co2_t']:.4f} tCO₂eq")

        elif args.mode == 'infer':
            calculator = InferenceCarbonCalculator(args)
            results = calculator.run()
            
            print("\n" + "="*50)
            print("LLM 推論 (Inference) 碳足跡計算結果".center(50))
            print("="*50)
            print(f"  - 預估執行時間: {results['execution_seconds']:.2f} 秒")
            print(f"  - 總消耗電量:   {results['total_energy_mwh']:.6f} MWh")
            print(f"  - 營運碳排:     {results['operational_co2_t']:.6f} tCO₂eq")

        print("="*50)
        print("\n免責聲明：此為基於 LLMCarbon 研究的估算值，實際碳排可能因多種因素而異。\n")

    except Exception as e:
        print(f"\n計算過程中發生錯誤: {e}")

if __name__ == '__main__':
    main()