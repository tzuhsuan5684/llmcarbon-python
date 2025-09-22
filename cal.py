# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import os

# --- 核心計算邏輯 ---

class OperationalCarbonModel:
    """
    負責計算營運碳排 (Operational Carbon Footprint)。
    此部分的計算邏輯基於 Faiz et al., 2024 的 LLMCarbon 論文。
    """

    # 預設硬體參數 (參考 LLMCarbon 論文 Table 4 & 相關資料)
    HARDWARE_PRESETS = {
        'V100': {'peak_tflops': 125, 'tdp_w': 300},
        'H100': {'peak_tflops': 1979, 'tdp_w': 700},
        'TPUv3': {'peak_tflops': 123, 'tdp_w': 450},
        'TPUv4': {'peak_tflops': 275, 'tdp_w': 175},
        'A100': {'peak_tflops': 624, 'tdp_w': 400}, # A100 SXM4 80GB
    }

    def __init__(self, args):
        self.args = args
        self.preset = self.HARDWARE_PRESETS.get(args.device)
        if not self.preset:
            raise ValueError(f"不支援的硬體裝置: {args.device}。請從 {list(self.HARDWARE_PRESETS.keys())} 中選擇。")

    def _calculate_total_flops(self):
        """根據論文公式 4: TC ≈ 6PD 計算總 FLOPs"""
        active_params_b = self.args.base_model_params_b if self.args.model_type == 'MoE' else self.args.parameters_b
        # 單位轉換: P (Billion), D (Trillion) -> 1e9 * 1e12 = 1e21 (Zetta)
        total_zettaflops = 6 * active_params_b * self.args.tokens_t / 1000 # 修正原始論文的單位表示
        return total_zettaflops * 1e21

    def _calculate_training_days(self, total_flops):
        """根據論文公式 7 計算訓練所需天數"""
        achieved_tflops_per_second = self.preset['peak_tflops'] * (self.args.hardware_efficiency_perc / 100)
        if achieved_tflops_per_second == 0 or self.args.device_num == 0:
            return 0
        training_seconds = total_flops / (self.args.device_num * achieved_tflops_per_second * 1e12)
        return training_seconds / (3600 * 24)

    def calculate(self):
        """計算總營運碳排 (tCO₂eq)"""
        total_flops = self._calculate_total_flops()
        training_days = self._calculate_training_days(total_flops)
        
        # 根據論文公式 8, 9, 10 計算
        total_power_kw = (self.args.system_power_w * self.args.device_num) / 1000
        total_energy_kwh = total_power_kw * (training_days * 24) * self.args.pue
        
        # 將 gCO₂eq/kWh 轉換為 tCO₂eq/MWh
        # (g/kWh) * (1 MWh / 1000 kWh) * (1 t / 1,000,000 g) = t / MWh
        carbon_intensity_t_per_mwh = self.args.co2_intensity_g_kwh / 1000 
        
        operational_co2_t = (total_energy_kwh / 1000) * carbon_intensity_t_per_mwh
        
        return {
            "operational_co2_t": operational_co2_t,
            "training_days": training_days,
            "total_energy_mwh": total_energy_kwh / 1000
        }


class EmbodiedCarbonModel:
    """
    負責計算隱含碳排 (Embodied Carbon Footprint)。
    此部分邏輯基於原始碼 embodied.py。
    """
    def __init__(self, hardware_data_path, training_days):
        if not os.path.exists(hardware_data_path):
            raise FileNotFoundError(f"找不到硬體數據文件: {hardware_data_path}")
        self.hardware_df = pd.read_csv(hardware_data_path)
        self.training_days = training_days
        self.hardware_lifespan_days = 5 * 365 # 根據論文，硬體生命週期為 5 年

    def calculate(self):
        """計算總隱含碳排 (tCO₂eq)"""
        # 根據論文公式 12
        time_ratio = self.training_days / self.hardware_lifespan_days
        
        # 計算每種硬體的隱含碳排 (kgCO2eq)
        self.hardware_df['embodied_co2_kg'] = self.hardware_df['unit (cm2 or GB)'] * \
                                              self.hardware_df['CPA (kgCO2/cm2 or GB)'] * \
                                              self.hardware_df['num']
                                              
        # 依據使用時間比例計算，並加總
        total_embodied_co2_kg = self.hardware_df['embodied_co2_kg'].sum()
        
        # 將其他組件 (主機板、機殼等) 的碳排也算進去 (根據論文，約佔 15%)
        # 這裡的計算方式參考了原始論文的 Table 5
        # 假設 'others' 佔總伺服器碳排(不含GPU)的15%
        # 這裡簡化處理，直接使用論文中的比例，因詳細計算需要伺服器BOM表
        total_embodied_co2_kg_with_others = total_embodied_co2_kg / (1 - 0.15)
        
        # 最終的隱含碳排(噸)
        embodied_co2_t = (total_embodied_co2_kg_with_others * time_ratio) / 1000
        
        return embodied_co2_t


def calculate_carbon_footprint(args):
    """主計算函式"""
    # 1. 計算營運碳排
    op_model = OperationalCarbonModel(args)
    op_results = op_model.calculate()

    # 2. 計算隱含碳排
    # 注意: 這裡的 hardware.csv 是基於 Meta XLM 的範例，使用者需依實際情況修改
    emb_model = EmbodiedCarbonModel('hardware.csv', op_results['training_days'])
    embodied_co2_t = emb_model.calculate()

    # 3. 整合結果
    total_co2_t = op_results['operational_co2_t'] + embodied_co2_t
    
    return {
        "operational_co2_t": op_results['operational_co2_t'],
        "embodied_co2_t": embodied_co2_t,
        "total_co2_t": total_co2_t,
        "training_days": op_results['training_days'],
        "total_energy_mwh": op_results['total_energy_mwh']
    }


def main():
    """命令列介面"""
    parser = argparse.ArgumentParser(
        description="LLM 碳足跡計算工具 (命令列版)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- 分隔線: LLM 模型設定 ---
    model_group = parser.add_argument_group('1. LLM 模型設定')
    model_group.add_argument('--model-type', type=str, default='dense', choices=['dense', 'MoE'], help='模型類型 (預設: dense)')
    model_group.add_argument('--parameters-b', type=float, default=175, help='模型總參數數量 (單位: 十億 B) (預設: 175)')
    model_group.add_argument('--base-model-params-b', type=float, default=2.3, help='MoE 模型的基礎模型參數數量 (B) (預設: 2.3)')
    model_group.add_argument('--tokens-t', type=float, default=300, help='處理的 Token 總數 (單位: 兆 T) (預設: 300)')
    
    # --- 分隔線: 硬體與資料中心設定 ---
    hw_group = parser.add_argument_group('2. 硬體與資料中心設定')
    hw_group.add_argument('--device', type=str, default='V100', choices=OperationalCarbonModel.HARDWARE_PRESETS.keys(), help=f'運算裝置類型 (預設: V100)')
    hw_group.add_argument('--device-num', type=int, default=10000, help='運算裝置總數量 (預設: 10000)')
    hw_group.add_argument('--system-power-w', type=float, default=330, help='單一裝置的平均系統功耗 (W) (包含主機等) (預設: 330)')
    hw_group.add_argument('--hardware-efficiency-perc', type=float, default=19.7, help='硬體效率 (%) (預設: 19.7)')
    hw_group.add_argument('--pue', type=float, default=1.1, help='資料中心的 PUE 值 (預設: 1.1)')
    hw_group.add_argument('--co2-intensity-g-kwh', type=float, default=429, help='電網碳強度 (gCO₂eq/kWh) (預設: 429, 美國平均)')
    
    args = parser.parse_args()

    # 執行計算
    try:
        results = calculate_carbon_footprint(args)
        
        # 格式化輸出
        print("\n" + "="*40)
        print("LLM 碳足跡計算結果".center(40))
        print("="*40)
        print(f"  營運碳排 (Operational): {results['operational_co2_t']:.2f} tCO₂eq")
        print(f"  隱含碳排 (Embodied):    {results['embodied_co2_t']:.2f} tCO₂eq")
        print("-" * 40)
        print(f"  總碳足跡 (Total):       {results['total_co2_t']:.2f} tCO₂eq")
        print("="*40)
        print("\n詳細資訊:")
        print(f"  - 預估訓練天數: {results['training_days']:.2f} 天")
        print(f"  - 總消耗電量:   {results['total_energy_mwh']:.2f} MWh")
        print("\n")

    except Exception as e:
        print(f"\n計算過程中發生錯誤: {e}")

if __name__ == '__main__':
    main()
