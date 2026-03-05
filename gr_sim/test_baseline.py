import argparse
import os
from accelerator import DecoderAccelerator

# model_list = ["facebook/opt-1.3b", "microsoft/phi-2", "01-ai/Yi-6B", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Meta-Llama-3-8B"]
model_list = ["facebook/opt-125m", "facebook/opt-1.3b"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is_generation", action="store_true", help="If enabled, then evaluate"
    )
    args = parser.parse_args()
    is_generation = args.is_generation

    if is_generation:
        pe_array_dim = [64, 12]
        mode_str = "generation"
    else:
        pe_array_dim = [32, 24]
        mode_str = "non-generation"

    # 创建结果文件夹
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)

    total_energy_list = [[0, 0] for _ in model_list]
    total_latency_list = [0 for _ in model_list]

    transmission_prec = 3.5
    decoder_cfg = {
        "transmission_prec": transmission_prec,
        "energy_per_bit": 0.78,  # pJ/bit (基于你的 N 值调整)
        "area_logic": 0.022,  # mm2
        "throughput_gbps": 8.0,  # Gbps (基于 P=8, F=1GHz, N=1)
        "small_sram_area": 0.005,  # 如果有额外的 Buffer 面积
        "small_sram_energy_per_access": 0.01,  # pJ/access
    }

    for idx, model_name in enumerate(model_list):
        acc = DecoderAccelerator(
            model_name=model_name,
            i_prec=16,
            w_prec=16,
            is_bit_serial=False,
            pe_dp_size=1,
            pe_energy=0.77,
            pe_area=1968.7,
            pe_array_dim=pe_array_dim,
            context_length=256,
            is_generation=is_generation,
            decoder_config=decoder_cfg,
        )

        total_cycle = acc.calc_cycle()
        compute_energy = acc.calc_compute_energy() / 1e6
        sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
        sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
        dram_energy = acc.calc_dram_energy() / 1e6
        onchip_energy = compute_energy + sram_rd_energy + sram_wr_energy
        total_energy = compute_energy + sram_rd_energy + sram_wr_energy + dram_energy

        # 打印到控制台
        print(f"model: {model_name}")
        print(f"total cycle:        {total_cycle}")
        total_latency_list[idx] = total_cycle[1]

        print(f"pe array area:      {acc.pe_array_area / 1e6} mm2")
        print(f"weight buffer area: {acc.w_sram.area} mm2")
        print(f"input buffer area:  {acc.i_sram.area} mm2")
        print(f"dram energy:        {dram_energy} uJ")
        print(f"on-chip energy:     {onchip_energy} uJ")
        print(f"total energy:       {total_energy} uJ")

        total_energy_list[idx][0] = round(onchip_energy)
        total_energy_list[idx][1] = round(total_energy)
        print("\n")

        # 保存到文件
        filename = f"{result_dir}/{model_name.replace('/', '_')}_{mode_str}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"model: {model_name}\n")
            f.write(f"mode: {mode_str}\n")
            f.write(f"total cycle: {total_cycle}\n")
            f.write(f"pe array area: {acc.pe_array_area / 1e6} mm2\n")
            f.write(f"weight buffer area: {acc.w_sram.area} mm2\n")
            f.write(f"input buffer area: {acc.i_sram.area} mm2\n")
            f.write(f"dram energy: {dram_energy} uJ\n")
            f.write(f"on-chip energy: {onchip_energy} uJ\n")
            f.write(f"total energy: {total_energy} uJ\n")

        print(f"结果已保存到: {filename}\n")

    print(f"Latency: {total_latency_list}")
    print(f"Energy: {total_energy_list}")
