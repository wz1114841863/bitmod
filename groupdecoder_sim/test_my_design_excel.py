import argparse
import os
import pandas as pd
from accelerator import Accelerator

# ==========================================
# 提取自压缩算法真实结果 (GS=512)
# ==========================================
model_bpw_config = {
    "facebook/opt-125m": {"base_bpw": 4.0781, "prop_bpw": 3.5171},
    "facebook/opt-1.3b": {"base_bpw": 4.0781, "prop_bpw": 3.4416},
    "facebook/opt-6.7b": {"base_bpw": 4.0781, "prop_bpw": 3.5526},
    "facebook/opt-13b": {"base_bpw": 4.0781, "prop_bpw": 3.6053},
    "huggyllama/llama-7b": {"base_bpw": 4.0781, "prop_bpw": 3.6488},
    "huggyllama/llama-13b": {"base_bpw": 4.0781, "prop_bpw": 3.6501},
    "Qwen/Qwen3-8B": {"base_bpw": 4.0781, "prop_bpw": 3.5528},
    "deepseek-ai/deepseek-llm-7b-chat": {"base_bpw": 4.0781, "prop_bpw": 3.6412},
}

if __name__ == "__main__":
    # ==========================================
    # 核心硬件参数配置 (与之前一致)
    # ==========================================
    PE_ARRAY_DIM = [32, 32]
    PE_AREA = 150.0
    PE_ENERGY = 0.15
    PE_DP_SIZE = 4
    DECODER_POWER_MW = 20.5
    SHARED_CACHE_R_COST = 1.2
    SHARED_CACHE_W_COST = 1.5

    # 用于存储所有实验结果的列表
    all_results = []

    print("=" * 60)
    print("🚀 开始全自动化端到端硬件评估 (Prefill + Decode)")
    print("=" * 60)

    # 循环两个阶段
    for is_gen in [False, True]:
        stage_name = "Decode" if is_gen else "Prefill"
        print(f"\n>>> 正在进行 [{stage_name}] 阶段评估...")

        for model_name, bpw_data in model_bpw_config.items():
            base_bpw = bpw_data["base_bpw"]
            prop_bpw = bpw_data["prop_bpw"]

            # 理论压缩比 (作为对照参考)
            theoretical_ratio = base_bpw / prop_bpw

            # 1. 评估 Baseline
            acc_base = Accelerator(
                model_name=model_name,
                i_prec=16,
                w_prec=4,
                avg_bpw=base_bpw,
                group_size=512,
                decoder_power_mw=0.0,
                extra_sram_r_cost=0.0,
                extra_sram_w_cost=0.0,
                is_bit_serial=False,
                pe_dp_size=PE_DP_SIZE,
                pe_energy=PE_ENERGY,
                pe_area=PE_AREA,
                pe_array_dim=PE_ARRAY_DIM,
                context_length=256,
                is_generation=is_gen,
            )
            cycle_base = acc_base.calc_cycle()[1]
            energy_base = (
                acc_base.calc_compute_energy()
                + acc_base.calc_sram_rd_energy()
                + acc_base.calc_sram_wr_energy()
                + acc_base.calc_dram_energy()
            )

            # 2. 评估 Proposed
            acc_prop = Accelerator(
                model_name=model_name,
                i_prec=16,
                w_prec=4,
                avg_bpw=prop_bpw,
                group_size=512,
                decoder_power_mw=DECODER_POWER_MW,
                extra_sram_r_cost=SHARED_CACHE_R_COST,
                extra_sram_w_cost=SHARED_CACHE_W_COST,
                is_bit_serial=False,
                pe_dp_size=PE_DP_SIZE,
                pe_energy=PE_ENERGY,
                pe_area=PE_AREA,
                pe_array_dim=PE_ARRAY_DIM,
                context_length=256,
                is_generation=is_gen,
            )
            cycle_prop = acc_prop.calc_cycle()[1]
            extra_e = acc_prop.calc_extra_onchip_energy()
            energy_prop = (
                acc_prop.calc_compute_energy()
                + acc_prop.calc_sram_rd_energy()
                + acc_prop.calc_sram_wr_energy()
                + acc_prop.calc_dram_energy()
                + extra_e
            )

            # 3. 统计指标
            speedup = cycle_base / cycle_prop
            energy_saving = (energy_base - energy_prop) / energy_base * 100

            # 记录到列表
            all_results.append(
                {
                    "Model": model_name,
                    "Stage": stage_name,
                    "Base_BPW": base_bpw,
                    "Prop_BPW": prop_bpw,
                    "Theoretical_Ratio": round(theoretical_ratio, 4),
                    "Baseline_Cycles": cycle_base,
                    "Proposed_Cycles": cycle_prop,
                    "Speedup": round(speedup, 4),
                    "Baseline_Energy_uJ": round(energy_base / 1e6, 2),
                    "Proposed_Energy_uJ": round(energy_prop / 1e6, 2),
                    "Energy_Saving_%": round(energy_saving, 2),
                }
            )
            print(f"  - {model_name} 完成. Speedup: {speedup:.2f}x")

    # ==========================================
    # 保存结果到 Excel
    # ==========================================
    df = pd.DataFrame(all_results)
    output_path = "Hardware_Evaluation_Results.xlsx"
    df.to_excel(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"✅ 所有评估完成！结果已保存至: {output_path}")
    print("=" * 60)
