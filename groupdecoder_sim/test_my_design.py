import argparse
import os
from accelerator import Accelerator

# ==========================================
# 提取自压缩算法真实结果 (GS=512)
# 字典结构: "模型名称": {"base_bpw": 压缩前平均bit, "prop_bpw": 压缩后平均bit}
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is_generation",
        action="store_true",
        default=False,
        help="默认为 Prefill 阶段，加上此参数为 Decode (Generation) 阶段",
    )
    args = parser.parse_args()
    is_generation = args.is_generation

    # ==========================================
    # 核心硬件参数配置 (基于评估推算)
    # ==========================================
    PE_ARRAY_DIM = [32, 32]  # 32x32 脉动阵列
    PE_AREA = 150.0  # 修正后的真实 INT4 MAC 面积 (um^2)
    PE_ENERGY = 0.15  # INT4 MAC 的预估能耗 (pJ/op)
    PE_DP_SIZE = 4  # 统一设置点积大小，释放阵列算力，使 DRAM 成为瓶颈

    # 额外硬件开销 (基于 DC 和 CACTI 数据)
    DECODER_POWER_MW = 20.5  # DecoderBank 动态功耗估算值 (mW = pJ/ns)
    SHARED_CACHE_R_COST = 1.2  # 2KB Shared Cache 读能耗 (pJ/access)
    SHARED_CACHE_W_COST = 1.5  # 2KB Shared Cache 写能耗 (pJ/access)
    # ==========================================

    print("=" * 60)
    print("🚀 开始端到端硬件加速比与能耗评估 (Group Size = 512)")
    print(
        f"设定硬件: {PE_ARRAY_DIM[0]}x{PE_ARRAY_DIM[1]} INT4 PE Array (DP Size={PE_DP_SIZE})"
    )
    print(
        f"评估阶段: {'Decode (Generation)' if is_generation else 'Prefill (Prompt Processing)'}"
    )
    print("=" * 60)

    # 遍历字典中的所有模型进行评估
    for model_name, bpw_data in model_bpw_config.items():
        base_bpw = bpw_data["base_bpw"]
        prop_bpw = bpw_data["prop_bpw"]

        print(f"\n评估模型: {model_name}")
        print(f"  -> Baseline 物理平均位宽: {base_bpw:.4f} bits/weight")
        print(f"  -> Proposed 物理平均位宽: {prop_bpw:.4f} bits/weight")

        # ----------------------------------------------------
        # 1. 评估 Baseline (传统 INT4, 无额外开销)
        # ----------------------------------------------------
        acc_base = Accelerator(
            model_name=model_name,
            i_prec=16,
            w_prec=4,  # 基础计算精度为 INT4
            avg_bpw=base_bpw,  # 注入 Baseline 的真实物理位宽
            group_size=512,
            decoder_power_mw=0.0,  # Baseline 无解码器
            extra_sram_r_cost=0.0,
            extra_sram_w_cost=0.0,
            is_bit_serial=False,
            pe_dp_size=PE_DP_SIZE,
            pe_energy=PE_ENERGY,
            pe_area=PE_AREA,
            pe_array_dim=PE_ARRAY_DIM,
            context_length=256,
            is_generation=is_generation,
        )

        cycle_base = acc_base.calc_cycle()[1]
        compute_e_base = acc_base.calc_compute_energy()
        sram_rd_e_base = acc_base.calc_sram_rd_energy()
        sram_wr_e_base = acc_base.calc_sram_wr_energy()
        dram_e_base = acc_base.calc_dram_energy()
        total_energy_base = (
            compute_e_base + sram_rd_e_base + sram_wr_e_base + dram_e_base
        )

        # ----------------------------------------------------
        # 2. 评估 Proposed (你的变长架构)
        # ----------------------------------------------------
        acc_prop = Accelerator(
            model_name=model_name,
            i_prec=16,
            w_prec=4,  # 基础计算精度仍为 INT4
            avg_bpw=prop_bpw,  # 注入你的真实压缩位宽
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
            is_generation=is_generation,
        )

        cycle_prop = acc_prop.calc_cycle()[1]
        compute_e_prop = acc_prop.calc_compute_energy()
        sram_rd_e_prop = acc_prop.calc_sram_rd_energy()
        sram_wr_e_prop = acc_prop.calc_sram_wr_energy()
        dram_e_prop = acc_prop.calc_dram_energy()
        # 解码引起的开销
        extra_onchip_e = acc_prop.calc_extra_onchip_energy()

        total_energy_prop = (
            compute_e_prop
            + sram_rd_e_prop
            + sram_wr_e_prop
            + dram_e_prop
            + extra_onchip_e
        )

        # ----------------------------------------------------
        # 3. 计算加速比与能效收益
        # ----------------------------------------------------
        speedup = cycle_base / cycle_prop
        energy_saving = (
            (total_energy_base - total_energy_prop) / total_energy_base * 100
        )

        # 打印结果对比
        print(f"  [性能 - Latency]")
        print(f"    - Baseline 总周期 : {cycle_base:,}")
        print(f"    - Proposed 总周期 : {cycle_prop:,}")
        print(f"    => 端到端吞吐量加速比 (Speedup) : \033[92m{speedup:.2f}x\033[0m")

        print(f"  [能耗 - Energy (uJ)]")
        print(f"    - Baseline DRAM 能耗  : {dram_e_base / 1e6:,.2f}")
        print(f"    - Proposed DRAM 能耗  : {dram_e_prop / 1e6:,.2f} (因压缩下降)")
        print(f"    - Proposed 额外片上开销: {extra_onchip_e / 1e6:,.2f} (含解码功耗)")
        print(f"    - Baseline 总能耗     : {total_energy_base / 1e6:,.2f}")
        print(f"    - Proposed 总能耗     : {total_energy_prop / 1e6:,.2f}")
        print(
            f"    => 系统总能耗降低 (Energy Saving): \033[92m{energy_saving:.2f}%\033[0m"
        )
        print("-" * 60)
