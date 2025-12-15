from mem.mem_instance import MemoryInstance


def get_cacti_params(size_bytes, bus_width_bits, bank_count=1):
    """
    利用 CACTI 计算小 Buffer 的参数
    """
    config = {
        "technology": 0.028,  # 保持和主 SRAM 一致
        "mem_type": "ram",  # 使用 RAM 模式
        "size": size_bytes * 8,  # CACTI 需要位
        "bank_count": bank_count,
        "rw_bw": bus_width_bits,  # 接口位宽
        "r_port": 1,
        "w_port": 1,
        "rw_port": 0,  # 1读1写端口 (Ping-Pong常用)
    }

    print(
        f"Running CACTI for Size: {size_bytes/1024:.2f} KB, Bus: {bus_width_bits} bits..."
    )

    # 调用 CACTI 接口
    mem = MemoryInstance(
        config, get_cost_from_cacti=True, min_w_granularity=64  # 最小写粒度
    )

    print(f"  => Area:   {mem.area:.6f} mm^2")
    print(f"  => Energy: {mem.r_cost:.4f} pJ/access (Read)")
    print(f"  => Energy: {mem.w_cost:.4f} pJ/access (Write)")
    print("-" * 30)

    # 返回平均能耗
    return mem.area, (mem.r_cost + mem.w_cost) / 2


if __name__ == "__main__":
    # --- 场景设定 ---
    # 假设你的 Small Buffer 是用来存解压后权重的 Ping-Pong Buffer
    # 大小 = 2 * P * Group_Size * 16bit
    # 假设 P=32 (PE行数), Group=128
    # Size = 2 * 32 * 128 * 2 Bytes = 16 KB

    small_buffer_size = 16 * 1024  # 16 KB
    bus_width = 128  # 假设内部总线 128 bit

    area, energy = get_cacti_params(small_buffer_size, bus_width)

    print("\n[请将以下数据填入 decoder_config]:")
    print(f"'small_sram_area': {area},")
    print(f"'small_sram_energy_per_access': {energy},")
