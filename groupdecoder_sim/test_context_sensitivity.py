import os
import pandas as pd
from accelerator_kv import Accelerator

TARGET_MODEL = "huggyllama/llama-7b"
BASE_BPW = 4.0781
PROP_BPW = 3.6488

# 指数级增长的 Context 长度
context_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

# 同时测试两种 KV Cache 精度场景：传统的 FP16 vs. 前沿的 INT4
kv_precs = [16, 4]

if __name__ == "__main__":
    # ==========================================
    # 核心硬件参数配置
    # ==========================================
    PE_ARRAY_DIM = [32, 32]
    PE_AREA = 150.0
    PE_ENERGY = 0.15
    PE_DP_SIZE = 4
    DECODER_POWER_MW = 20.5
    SHARED_CACHE_R_COST = 1.2
    SHARED_CACHE_W_COST = 1.5

    all_results = []

    print("=" * 65)
    print(f"🚀 开始运行 Context Length 与 KV Cache 精度敏感性分析")
    print(f"评估模型: {TARGET_MODEL}")
    print("=" * 65)

    for kv_prec in kv_precs:
        kv_mode = "FP16" if kv_prec == 16 else "INT4"
        print(f"\n==========================================")
        print(f"🌟 当前 KV Cache 精度模式: {kv_mode} (kv_prec={kv_prec})")
        print(f"==========================================")

        for ctx_len in context_lengths:
            # 1. 评估 Baseline
            acc_base = Accelerator(
                model_name=TARGET_MODEL,
                i_prec=16,
                kv_prec=kv_prec,  # 动态注入 KV Cache 精度
                w_prec=4,
                avg_bpw=BASE_BPW,
                group_size=512,
                decoder_power_mw=0.0,
                extra_sram_r_cost=0.0,
                extra_sram_w_cost=0.0,
                is_bit_serial=False,
                pe_dp_size=PE_DP_SIZE,
                pe_energy=PE_ENERGY,
                pe_area=PE_AREA,
                pe_array_dim=PE_ARRAY_DIM,
                cxt_len=ctx_len,
                is_generation=True,
            )
            cycle_base = acc_base.calc_cycle()[1]

            # 2. 评估 Proposed
            acc_prop = Accelerator(
                model_name=TARGET_MODEL,
                i_prec=16,
                kv_prec=kv_prec,  # 动态注入 KV Cache 精度
                w_prec=4,
                avg_bpw=PROP_BPW,
                group_size=512,
                decoder_power_mw=DECODER_POWER_MW,
                extra_sram_r_cost=SHARED_CACHE_R_COST,
                extra_sram_w_cost=SHARED_CACHE_W_COST,
                is_bit_serial=False,
                pe_dp_size=PE_DP_SIZE,
                pe_energy=PE_ENERGY,
                pe_area=PE_AREA,
                pe_array_dim=PE_ARRAY_DIM,
                cxt_len=ctx_len,
                is_generation=True,
            )
            cycle_prop = acc_prop.calc_cycle()[1]

            # 3. 统计指标
            speedup = cycle_base / cycle_prop

            all_results.append(
                {
                    "Model": TARGET_MODEL,
                    "KV_Cache_Precision": kv_mode,
                    "Context_Length": ctx_len,
                    "Baseline_Cycles": cycle_base,
                    "Proposed_Cycles": cycle_prop,
                    "Speedup": round(speedup, 4),
                }
            )
            print(f"  - Context: {ctx_len:<4} | Speedup: \033[92m{speedup:.3f}x\033[0m")

    # ==========================================
    # 保存结果到 Excel
    # ==========================================
    df = pd.DataFrame(all_results)
    output_path = "Context_Sensitivity_with_KV_Results.xlsx"
    df.to_excel(output_path, index=False)

    print("\n" + "=" * 65)
    print(f"✅ 敏感性分析完成！结果已保存至: {output_path}")
    print("=" * 65)
