from mem.mem_instance import MemoryInstance


def get_cacti_params(size_bytes, bus_width_bits, bank_count=1):
    """
    利用 CACTI 计算 Buffer 的参数
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
        config,
        get_cost_from_cacti=True,
    )

    # print(f"  => Area:   {mem.area:.6f} mm^2")
    # print(f"  => Energy: {mem.r_cost:.4f} pJ/access (Read)")
    # print(f"  => Energy: {mem.w_cost:.4f} pJ/access (Write)")
    # print("-" * 30)

    # 返回平均能耗
    return mem.area, mem.r_cost, mem.w_cost


if __name__ == "__main__":
    # WeightSRAM"
    area, r_cost, w_cost = get_cacti_params(256, 32, 1)
    total_area = area * 16  # 整体面积
    total_r_cost = r_cost * 8  # 读功耗
    total_w_cost = w_cost * 8  # 写功耗
    """
    WeightSRAM 相关参数:
        small_sram_area: 0.015319104 mm^2
        total_r_cost: 2.3003839999999998 pJ/access
        total_w_cost: 4.02528 pJ/access
    """
    print("\n WeightSRAM 相关参数:")
    print(f"small_sram_area: {total_area} mm^2")
    print(f"total_r_cost: {total_r_cost} pJ/access")
    print(f"total_w_cost: {total_w_cost} pJ/access")
    print("-" * 30)

    # Shared Cache:
    area, r_cost, w_cost = get_cacti_params(2048, 64, 1)
    """
    Shared Cache 相关参数:
    small_sram_area: 0.00695606 mm^2
    total_r_cost: 1.35969 pJ/access
    total_w_cost: 2.03232 pJ/access
    """
    print("\n Shared Cache 相关参数:")
    print(f"small_sram_area: {area} mm^2")
    print(f"total_r_cost: {r_cost} pJ/access")
    print(f"total_w_cost: {w_cost} pJ/access")
    print("-" * 30)

    # ==========================================
    # 2. 评估 MetaSRAM (差分对比)
    # ==========================================
    print("\n=== MetaSRAM: Baseline (传统 INT4) ===")
    # 包含 Scale(16) + ZP(8) = 24-bit
    base_area, base_read_energy, base_write_energy = get_cacti_params(
        size_bytes=750000, bus_width_bits=32, bank_count=8
    )
    base_energy = (base_read_energy + base_write_energy) / 2
    print("\n=== MetaSRAM: Proposed (Golomb-Rice 变长压缩) ===")
    # 包含 Offset(32) + Scale(16) + ZP(8) = 56-bit
    prop_area, prop_read_energy, prop_write_energy = get_cacti_params(
        size_bytes=1750000, bus_width_bits=56, bank_count=8
    )
    prop_energy = (prop_read_energy + prop_write_energy) / 2
    # ==========================================
    # 3. 最终开销计算 (Hardware Overhead)
    # ==========================================
    print("\n" + "=" * 40)
    print(f"变长解码引入的 MetaSRAM 额外面积代价: {prop_area - base_area:.6f} mm^2")
    print(f"变长解码引入的额外读写功耗代价: {prop_energy - base_energy:.4f} pJ/access")
    print("=" * 40)

    # 1. 评估共有的 WeightSRAM (以 N=32 为例，总计 16KB)
    # 物理上依然切分成多个 Bank (例如 8 个 2KB 的 Bank) 以匹配读写带宽
    print("=== Inherent: WeightSRAM (16 KB) ===")
    # 假定每个 Bank 2KB，32-bit 位宽
    area_w, r_w, w_w = get_cacti_params(
        size_bytes=2048, bus_width_bits=32, bank_count=1
    )
    total_area_w = area_w * 8  # 8个Bank的总面积

    # 2. 评估额外引入的 Shared Cache (2 KB)
    print("\n=== Introduced: Shared Cache (2 KB) ===")
    # 宽总线 64-bit 抓取
    area_s, r_s, w_s = get_cacti_params(
        size_bytes=2048, bus_width_bits=64, bank_count=1
    )
    """
    [Result] WeightSRAM Area: 0.0515 mm^2 (Baseline & Proposed)
    [Result] Extra Shared Cache Area: 0.0070 mm^2 (Proposed Only)
    """
    print(f"\n[Result] WeightSRAM Area: {total_area_w:.4f} mm^2 (Baseline & Proposed)")
    print(f"[Result] Extra Shared Cache Area: {area_s:.4f} mm^2 (Proposed Only)")
