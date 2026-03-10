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
            - small_sram_area: 额外缓冲区的面积 (mm^2)
            - small_sram_energy_per_weight: 额外缓冲区每次访问(读/写)的能耗 (pJ/weight)
            #
        """
        # 核心压缩参数
        self.trans_prec = config.get("transmission_prec", 4.0)

        # 逻辑开销
        self.energy_per_bit = config.get("energy_per_bit", 0.0)
        self.area_logic = config.get("area_logic", 0.0)
        self.throughput_gbps = config.get("throughput_gbps", float("inf"))

        # 引入的额外存储开销
        self.small_sram_area = config.get("small_sram_area", 0.0)
        self.small_sram_energy_per_weight = config.get(
            "small_sram_energy_per_weight", 0.0
        )

        # DC仿真得到的频率 (GHz)
        self.frequency_ghz = config.get("frequency_ghz", 1.0)

    def calc_logic_energy(self, total_compressed_bits):
        """计算纯解码逻辑的动态能耗"""
        return total_compressed_bits * self.energy_per_bit

    def calc_small_buffer_energy(self, num_weights):
        """计算 Small Buffer 的读写能耗 (1次写入 + 1次读取 = 2次访问)"""
        output = num_weights * 2 * self.small_sram_energy_per_weight
        return output

    def get_total_area_overhead(self):
        """返回额外的总面积"""
        return self.area_logic + self.small_sram_area

    def get_transmission_precision(self):
        """获取压缩后的权重位宽"""
        return self.trans_prec

    def get_throughput_bits_per_ns(self):
        """获取吞吐率 (bits/ns)"""
        return self.throughput_gbps

    def get_frequency_ghz(self):
        """获取解码器的工作频率 (GHz)"""
        return self.frequency_ghz
