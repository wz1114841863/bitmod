def get_decoder_config(transmission_prec=3.5):
    """
    获取 Decoder 的配置参数, 根据DC仿真结果和 CACTI 计算得到的能耗/面积数据进行设置
    这些参数将被传递给 Decoder 类, 用于后续的能耗和面积计算.

    transmission_prec:
        压缩后的平均传输位宽 (用于 DRAM 带宽和 SRAM 容量计算)
    energy_per_bit:
        解码器逻辑每处理 1 bit 压缩数据的能耗 (pJ).
        energy_per_bit = sum(power dynamic) / system throughput (bit/s)
    area_logic:
        解码器纯逻辑电路面积 (mm^2).
        Area(DecoderBank) + Area(MetadataLoade) + Area(WeightLoader)
    throughput_gbps:
        解码器吞吐率 (Gbps), 用于计算流水线延迟
        throughput_gbps = P * 每周期解码的bit数 * 频率 (Gbps = bits/ns),
        eg: 8 * 4bit * 1GHz = 32 Gbps
    small_sram_area:
        额外缓冲区的面积 (mm^2).
        Area(WeightSRAM)+ Area(MetaSRAMBuffer)
    small_sram_energy_per_weight:
        额外缓冲区每次访问(读/写)的能耗 (pJ/weight)
    """
    scale_factor = 8
    decoder_cfg = {
        "transmission_prec": transmission_prec,
        "energy_per_bit": 0.1876,  # pJ/bit.
        "area_logic": 0.021941 * scale_factor,  # mm^2.
        "throughput_gbps": 32.0 * scale_factor,  # Gbps.
        "small_sram_area": 0.0977784 * scale_factor,  # mm^2.
        "small_sram_energy_per_weight": 0.274,  # pJ/weight
    }
    return decoder_cfg
