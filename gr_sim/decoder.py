import math


class Decoder:
    def __init__(self, config: dict):
        """
        初始化解码器硬件模块
        config: 包含硬件综合参数的字典
            - transmission_prec: 压缩后的平均传输位宽 (用于 DRAM 带宽和 SRAM 容量计算)
            - energy_per_bit: 解码器逻辑每处理 1 bit 压缩数据的能耗 (pJ)
            - area_logic: 解码器纯逻辑电路面积 (mm^2)
            - throughput_gbps: 解码器吞吐率 (Gbps), 用于计算流水线延迟

            # 额外的小缓冲 (Small Buffer / Ping-Pong Buffer) 参数
            - small_sram_area: 额外缓冲区的面积 (mm^2)
            - small_sram_energy_per_access: 额外缓冲区每次访问(读/写)的能耗 (pJ/access)
        """
        # 核心压缩参数
        self.trans_prec = config.get("transmission_prec", 4.0)

        # 逻辑开销
        self.energy_per_bit = config.get("energy_per_bit", 0.0)
        self.area_logic = config.get("area_logic", 0.0)
        # 吞吐率: P * 每周期解码的bit数 * 频率 (Gbps = bits/ns), eg: 8 * 4bit * 1GHz = 32 Gbps
        self.throughput_gbps = config.get("throughput_gbps", float("inf"))

        # 引入的额外存储开销
        self.small_sram_area = config.get("small_sram_area", 0.0)
        self.small_sram_energy_per_access = config.get(
            "small_sram_energy_per_access", 0.0
        )
        self.decoded_weight_bits = config.get("weight_bits", 4)
        self.bus_width = config.get("bus_width", 32)

        # DC仿真得到的频率 (GHz)
        self.frequency_ghz = config.get("frequency_ghz", 1.0)

    def calc_logic_energy(self, total_compressed_bits):
        """计算纯解码逻辑的动态能耗"""
        return total_compressed_bits * self.energy_per_bit

    def calc_small_buffer_energy(self, num_weights):
        """
        计算 Small Buffer 的读写能耗
        过程: Main SRAM -> (写) -> Small Buffer -> (读) -> PE
        所以是 1次写 + 1次读 = 2次访问
        """

        output = (
            (num_weights * self.decoded_weight_bits / self.bus_width)
            * 2
            * self.small_sram_energy_per_access
        )
        return output

    def get_total_area_overhead(self):
        """
        返回额外的总面积 (逻辑 + 小缓冲)
        """
        return self.area_logic + self.small_sram_area

    def get_transmission_precision(self):
        return self.trans_prec

    def get_throughput_bits_per_ns(self):
        # Gbps = bits / ns
        return self.throughput_gbps

    def get_frequency_ghz(self):
        return self.frequency_ghz
