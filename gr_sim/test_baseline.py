import argparse
import os
from accelerator import DecoderAccelerator
from decoder_config import get_decoder_config


# model_list = ["facebook/opt-125m", "facebook/opt-1.3b", "microsoft/phi-2", "01-ai/Yi-6B", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Meta-Llama-3-8B"]
model_list = ["facebook/opt-125m", "facebook/opt-1.3b"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is_generation",
        action="store_true",
        help="If enabled, then evaluate",
    )
    parser.add_argument(
        "--use_decoder",
        action="store_true",
        help="If enabled, then include decoder energy",
    )
    args = parser.parse_args()
    is_generation = args.is_generation
    use_decoder = args.use_decoder

    if is_generation:
        mode_str = "generation"
    else:
        mode_str = "non-generation"

    pe_array_dim = [32, 16]

    # 创建结果文件夹
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)

    total_energy_list = []
    total_latency_list = []

    transmission_prec = 3.5
    decoder_cfg = (
        get_decoder_config(transmission_prec=transmission_prec) if use_decoder else None
    )

    for idx, model_name in enumerate(model_list):
        acc = DecoderAccelerator(
            model_name=model_name,
            i_prec=16,
            w_prec=4 if use_decoder else 4.5,  # 如果使用解码器,则权重精度为压缩后的位宽
            is_bit_serial=False,
            pe_dp_size=1,
            pe_energy=0.77,
            pe_area=1968.7,
            pe_array_dim=pe_array_dim,
            context_length=256,
            is_generation=is_generation,
            decoder_config=decoder_cfg if use_decoder else None,
        )

        total_cycle = acc.calc_cycle()
        compute_energy = acc.calc_compute_energy() / 1e6
        sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
        sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
        dram_energy = acc.calc_dram_energy() / 1e6

        # 引入的额外解码器和缓冲能耗
        extra_onchip_energy = acc.calc_extra_onchip_energy() / 1e6
        if hasattr(acc, "calc_extra_onchip_energy"):
            extra_decoder_energy = acc.calc_extra_onchip_energy() / 1e6
        else:
            extra_decoder_energy = 0.0

        base_onchip_energy = compute_energy + sram_rd_energy + sram_wr_energy
        total_onchip_energy = base_onchip_energy + extra_decoder_energy
        total_energy = total_onchip_energy + dram_energy

        # 打印到控制台
        print(f"model: {model_name}")
        print(f"total cycle:        {total_cycle}")
        print(f"pe array area:      {acc.pe_array_area} um2")
        print(f"weight buffer area: {acc.w_sram.area} um2")
        print(f"input buffer area:  {acc.i_sram.area} um2")
        if acc.decoder:
            extra_area = acc.decoder.get_total_area_overhead()
            print(f"extra decoder area: {extra_area} um2")

        print(f"dram energy:        {dram_energy:.4f} uJ")
        print(f"on-chip energy:     {total_onchip_energy:.4f} uJ")
        # 以树状图形式明确拆解片上能耗,让开销一目了然
        print(f"  ├─ compute energy:      {compute_energy:.4f} uJ")
        print(f"  ├─ sram rd/wr energy:   {(sram_rd_energy + sram_wr_energy):.4f} uJ")
        print(f"  └─ EXTRA DECODER ENERGY:{extra_decoder_energy:.4f} uJ")

        print(f"total energy:       {total_energy:.4f} uJ\n")

        total_latency_list.append(total_cycle[1])
        total_energy_list.append(
            [
                int(total_onchip_energy),
                int(total_energy),
                int(extra_decoder_energy),
            ]
        )

        # 保存到文件
        safe_model_name = model_name.replace("/", "_")
        phase_str = "generation" if is_generation else "non-generation"
        file_path = f"results/{safe_model_name}_{phase_str}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"model: {model_name}\n")
            f.write(f"phase: {phase_str}\n")
            f.write(f"total cycle:        {total_cycle}\n")
            f.write("-" * 40 + "\n")
            f.write(f"pe array area:      {acc.pe_array_area:.6f} mm2\n")
            f.write(f"weight buffer area: {acc.w_sram.area:.6f} mm2\n")
            f.write(f"input buffer area:  {acc.i_sram.area:.6f} mm2\n")

            # 如果存在解码器,写入解码器额外面积
            if acc.decoder:
                extra_area = acc.decoder.get_total_area_overhead()
                f.write(f"extra decoder area: {extra_area:.6f} mm2\n")

            f.write("-" * 40 + "\n")
            f.write(f"dram energy:        {dram_energy:.4f} uJ\n")
            f.write(f"on-chip energy:     {total_onchip_energy:.4f} uJ\n")

            # 详细的片上能耗拆解树状图
            f.write(f"  ├─ compute energy:      {compute_energy:.4f} uJ\n")
            f.write(
                f"  ├─ sram rd/wr energy:   {(sram_rd_energy + sram_wr_energy):.4f} uJ\n"
            )
            f.write(f"  └─ extra decoder energy:{extra_decoder_energy:.4f} uJ\n")

            f.write("-" * 40 + "\n")
            f.write(f"total energy:       {total_energy:.4f} uJ\n")

        print(f"\n[Success] 详细结果已保存到: {file_path}\n")
        print("=" * 60)

    print(f"Latency: {total_latency_list}")
    print(f"Energy: {total_energy_list}")
