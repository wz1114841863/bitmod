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
    # WeightSRAM参数推导:
    # 采用了双缓冲结构, 总容量需要乘以2.
    # 存储深度: 每个Bank内部有P个独立的存储单元.
    # Total Size = 2 * P * Depth_Per_Bank * weightWidth
    # 具体结果: 2 * 8 * 256Bytes = 4 KB
    # 总线位宽为32bits.

    small_buffer_size = 2 * 8 * 256
    bus_width = 32
    bank_count = 8  # P

    area, energy = get_cacti_params(small_buffer_size, bus_width, bank_count)

    print("\n WeightSRAM 相关参数:")
    # 0.013013 mm^2
    print(f"'small_sram_area': {area},")
    # 单次 32-bit 访问的能耗, 2.090436 pJ/access
    print(f"'small_sram_energy_per_access': {energy},")
