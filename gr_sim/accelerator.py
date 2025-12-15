import math
import torch.nn as nn
import numpy as np

from typing import List
from mem.mem_instance import MemoryInstance
from pe_array import PE_Array


# Stripes accelerator
class Accelerator(PE_Array):
    """结构级模拟器,用于估算Transformer模型在特定PE阵列 + SRAM + DRAM 架构上运行时的总周期与总能耗."""

    PR_SCALING = 1.5  # scaling factor to account for post placement and routing

    def __init__(
        self,
        model_name: str,
        i_prec: int = 16,
        w_prec: int = 8,
        is_bit_serial: bool = False,
        pe_dp_size: int = 1,
        pe_energy: float = 0,
        pe_area: float = 0,
        pe_array_dim: List[int] = [],
        init_mem: bool = True,
        context_length: int = 256,
        is_generation: bool = False,
    ):
        super().__init__(
            model_name,
            i_prec,
            w_prec,
            is_bit_serial,
            pe_dp_size,
            pe_energy,
            pe_area,
            pe_array_dim,
            context_length,
            is_generation,
        )

        self.cycle_compute = None
        if init_mem:
            # 实例化三块内存
            self._init_mem()
            # 检查每层所需的内存大小
            self._check_layer_mem_size()
            # 并计算是否需要refetch(数据重取)
            self._calc_num_mem_refetch()

    def calc_cycle(self):
        """顶层接口,计算总运行周期.
        总周期 = max(计算周期, DRAM 访问周期), 按层累加.
        模拟了 Double Buffering (双缓冲) 或流水线机制.计算和加载是并行进行的,所以总时间取决于那个"慢"的步骤(Compute-bound vs Memory-bound).
        """
        self._calc_compute_cycle()  # 计算每层的计算周期
        self._calc_dram_cycle()  # 计算每层的DRAM访问周期
        total_cycle = 0
        total_cycle_compute = 0
        for name in self.layer_name_list:
            cycle_layer_compute = self._layer_cycle_compute[name]
            cycle_layer_dram = self._layer_cycle_dram[name]
            total_cycle_compute += cycle_layer_compute
            print(
                f"Layer: {name}, Compute Cycle: {cycle_layer_compute}, DRAM Cycle: {cycle_layer_dram}"
            )
            total_cycle += max(cycle_layer_compute, cycle_layer_dram)
        self.cycle_compute = total_cycle_compute
        return total_cycle_compute, total_cycle

    def _calc_compute_cycle(self):
        """计算所有层在 PE 阵列上运算需要的周期数."""
        self._layer_cycle_compute = {}
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            if w_dim is not None:
                tile_layer = self._calc_tile_fc(w_dim, o_dim)  # 分块数
                cycle_layer_compute = tile_layer * self.pe_latency  # 总周期
                self._layer_cycle_compute[name] = cycle_layer_compute

    def calc_pe_array_tile(self):
        total_tile = 0
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            o_dim = self.output_dim[name]
            total_tile += self._calc_tile_fc(w_dim, o_dim)
        return total_tile

    def _calc_tile_fc(self, w_dim, o_dim):
        """计算一个矩阵乘法需要切分成多少个 Tile(小块)才能在 PE 阵列上算完成."""
        pe_dp_size = self.pe_dp_size  # PE的点积大小
        num_pe_row = self.pe_array_dim["h"]  # PE阵列行数
        num_pe_col = self.pe_array_dim["w"]  # PE阵列列数

        # output channel, input channel
        cout, cin = w_dim  # [output_dim, input_dim]
        # num token, output channel
        num_token, _ = o_dim  # [num_token, output_dim]

        # tile_in_channel:   number of tiles along input channel
        # tile_cout:  number of tiles along output channel
        tile_in_channel = math.ceil(cin / pe_dp_size)  # 输入维度分块
        tile_cout = math.ceil(cout / num_pe_row)  # 输出维度分块
        tile_token = math.ceil(num_token / num_pe_col)  # token维度分块)

        total_tile = tile_in_channel * tile_cout * tile_token
        return total_tile

    def _calc_dram_cycle(self):
        """根据每层权重/输入/输出的数据量和DRAM带宽,算DRAM访问周期"""
        self._layer_cycle_dram = {}
        dram_bandwidth = self.dram.rw_bw * 2  # DDR双倍带宽

        for name in self.layer_name_list:
            i_prec = self.i_prec
            if ("attn_qk" in name) or ("attn_v" in name):
                w_prec = self.i_prec
            else:
                w_prec = self.w_prec
            w_dim = self.weight_dim[name]
            # 权重或激活重取次数
            num_dram_fetch_w, num_dram_fetch_i = self._layer_mem_refetch[name]
            # 权重加载周期
            cycle_dram_load_w = self._w_mem_required[name] * 8 / dram_bandwidth
            cycle_dram_load_w *= num_dram_fetch_w
            # 输入加载周期
            cycle_dram_load_i = self._i_mem_required[name] * 8 / dram_bandwidth
            cycle_dram_load_i *= num_dram_fetch_i
            # 输出写回周期
            cycle_dram_write_o = self._o_mem_required[name] * 8 / dram_bandwidth

            cycle_layer_dram = (
                cycle_dram_load_w + cycle_dram_write_o + cycle_dram_load_i
            )
            self._layer_cycle_dram[name] = int(cycle_layer_dram)

    def calc_compute_energy(self):
        """计算 PE 阵列运算消耗的动态能耗"""
        if self.cycle_compute is None:
            self.cycle_compute, _ = self.calc_cycle()
        # PE阵列的计算能耗 = PE能耗 × PE数量 × 计算周期
        compute_energy = self.pe_energy * self.total_pe_count * self.cycle_compute
        return compute_energy

    def calc_sram_rd_energy(self):
        """计算从片上 SRAM 读取数据供给 PE 的能耗"""
        w_sram_rd_cost = self.w_sram.r_cost  # 权重SRAM读能耗
        i_sram_rd_cost = self.i_sram.r_cost  # 输入SRAM读能耗
        num_pe_row = self.pe_array_dim["h"]
        num_pe_col = self.pe_array_dim["w"]
        if self.cycle_compute is None:
            self.cycle_compute, _ = self.calc_cycle()
        num_cycle_compute = self.cycle_compute
        num_tile = self.calc_pe_array_tile()  # 总计算块数
        # SRAM读取能耗 = 总计算块数 × (权重SRAM读能耗 + 输入SRAM读能耗)
        sram_rd_energy = num_tile * (w_sram_rd_cost + i_sram_rd_cost)
        return sram_rd_energy

    def calc_sram_wr_energy(self):
        """计算把数据从 DRAM 写入到 SRAM 的能耗"""
        total_energy = 0
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]
            total_energy += self._calc_sram_wr_energy_fc(
                name, w_dim, i_dim, o_dim, self.w_prec, self.i_prec
            )
        return total_energy

    def _calc_sram_wr_energy_fc(self, layer_name, w_dim, i_dim, o_dim, w_prec, i_prec):
        w_sram_wr_cost = self.w_sram.w_cost_min  # 权重SRAM写能耗
        i_sram_wr_cost = self.i_sram.w_cost_min  # 输入SRAM写能耗
        w_sram_min_wr_bw = self.w_sram.w_bw_min  # 最小写带宽
        i_sram_min_wr_bw = self.i_sram.w_bw_min  # 最小写带宽
        num_fetch_w, num_fetch_i = self._layer_mem_refetch[layer_name]

        # output channel, weight hidden size
        cout, cin_w = w_dim
        # num token, input hidden size
        _, cin_i = i_dim
        # num token, output channel
        num_token, _ = o_dim

        # write energy, read from DRAM and write to SRAM
        # 权重写入次数(考虑SRAM写带宽限制)
        num_w_sram_wr = math.ceil(cin_w * w_prec / w_sram_min_wr_bw) * cout
        energy_w_sram_wr = num_w_sram_wr * w_sram_wr_cost * num_fetch_w
        # 输入写入次数
        num_i_sram_wr = math.ceil(cin_i * i_prec / i_sram_min_wr_bw) * num_token
        energy_i_sram_wr = num_i_sram_wr * i_sram_wr_cost * num_fetch_i
        num_o_sram_wr = math.ceil(cout * i_prec / i_sram_min_wr_bw) * num_token
        # 输出写入次数
        energy_o_sram_wr = num_o_sram_wr * i_sram_wr_cost
        # SRAM写入能耗 = 权重写入能耗 + 输入写入能耗 + 输出写入能耗
        total_energy = energy_w_sram_wr + energy_i_sram_wr + energy_o_sram_wr
        return total_energy

    def calc_dram_energy(self):
        """计算 DRAM 访问的能耗"""
        energy = 0
        for name in self.layer_name_list:
            energy += self._calc_dram_energy_fc(name)
        return energy

    def _calc_dram_energy_fc(self, layer_name):
        """计算单层 DRAM 访问能耗
        DRAM能耗 = (数据量(bit) ÷ 总线宽度) x 单位访问能耗
        """
        size_sram_i = self.i_sram.size / 8  # 输入SRAM容量(字节)
        bus_width = self.dram.rw_bw  # DRAM总线宽度
        rd_cost = self.dram.r_cost  # DRAM读能耗
        wr_cost = self.dram.w_cost  # DRAM写能耗

        num_fetch_w, num_fetch_i = self._layer_mem_refetch[layer_name]

        # energy_weight: energy to read weight from DRAM
        w_mem_required = self._w_mem_required[layer_name]
        energy_weight = w_mem_required * 8 / bus_width * rd_cost
        # energy_input:  energy to read input feature from DRAM
        i_mem_required = self._i_mem_required[layer_name]
        energy_input = i_mem_required * 8 / bus_width * rd_cost
        # energy_output: energy to write output feature to DRAM
        o_mem_required = self._o_mem_required[layer_name]
        energy_output = o_mem_required * 8 / bus_width * wr_cost
        # 考虑数据重取次数
        energy_weight *= num_fetch_w
        energy_input *= num_fetch_i
        total_energy = energy_weight + energy_input + energy_output
        return total_energy

    def _check_layer_mem_size(self):
        """计算每一次(layer)计算所需的权重/输入/输出内存大小"""
        self._w_mem_required = {}  # 每层权重所需内存大小
        self._i_mem_required = {}  # 每层输入所需内存大小
        self._o_mem_required = {}  # 每层输出所需内存大小

        for layer_idx, name in enumerate(self.layer_name_list):
            i_prec = self.i_prec
            if ("attn_qk" in name) or ("attn_v" in name):
                w_prec = self.i_prec
            else:
                w_prec = self.w_prec

            w_dim = self.weight_dim[name]
            i_dim = self.input_dim[name]
            o_dim = self.output_dim[name]

            # output channel, weight hidden size
            cout, cin_w = w_dim  # [output_dim, input_dim]
            # num token, input hidden size
            _, cin_i = i_dim  # [num_token, input_dim]
            # num token, output channel
            num_token, _ = o_dim  # [num_token, output_dim]
            self._w_mem_required[name] = math.ceil(cin_w * w_prec / 8) * cout
            self._i_mem_required[name] = math.ceil(cin_i * i_prec / 8) * num_token
            self._o_mem_required[name] = math.ceil(cout * i_prec / 8) * num_token

    def _calc_num_mem_refetch(self):
        """核心数据流调度逻辑
        如果权重所需空间 > 权重SRAM大小 且 输入所需空间 > 输入SRAM大小,
        则需要在DRAM和SRAM之间反复取回数据.
        方案A: 每次取回所有权重数据, 对每个输入tile重复使用这些权重数据.
        方案B: 每次取回所有输入数据, 对每个权重tile重复使用这些输入数据.
        选择总数据传输量更小的方案进行模拟.
        """
        # If the on-chip buffer size is not big enough,
        # we need to refetch input tiles or weight tiles from DRAM
        # (num_fetch_w, num_fetch_i): 权重和输入需要重取的次数
        self._layer_mem_refetch = {}
        size_sram_w = self.w_sram.size / 8  # 权重SRAM容量(字节)
        size_sram_i = self.i_sram.size / 8  # 输入SRAM容量(字节)
        for name in self.layer_name_list:
            w_dim = self.weight_dim[name]
            if w_dim is not None:
                w_mem_required = self._w_mem_required[name]
                i_mem_required = self._i_mem_required[name]
                if (w_mem_required > size_sram_w) and (i_mem_required > size_sram_i):
                    # 两个缓存都不够用, 需要数据重取
                    # need DRAM refetch
                    num_refetch_input = math.ceil(w_mem_required / size_sram_w)
                    num_refetch_weight = math.ceil(i_mem_required / size_sram_i)
                    total_fetch_weight = num_refetch_weight * w_mem_required
                    total_fetch_input = num_refetch_input * i_mem_required
                    # print(f'{name}, Need DRAM refetch ...')
                    # print(f'w_dim: {w_dim}, i_dim: {i_dim}')
                    # 选择总数据传输量最小的方案
                    if (total_fetch_weight + i_mem_required) < (
                        total_fetch_input + w_mem_required
                    ):
                        # print(f'Refetch weight for {num_refetch_weight} times ...')
                        # refetch all weight for every input tile
                        # 反复取权重
                        self._layer_mem_refetch[name] = (num_refetch_weight, 1)
                    else:
                        # print(f'Refetch input for {num_refetch_input} times ...\n\n')
                        # refetch all input for every weight tile
                        # 反复取输入
                        self._layer_mem_refetch[name] = (1, num_refetch_input)
                else:
                    # no need refetch
                    self._layer_mem_refetch[name] = (1, 1)

    def _init_mem(self):
        """定义具体的存储硬件规格
        SRAM(片上缓存): 定义了 w_sram(权重缓存)和 i_sram(输入/激活缓存).
        DRAM(片外存储): 定义了 dram(主存储器).
        """
        # 权重SRAM
        if self.is_bit_serial:
            w_bandwidth = (
                self.pe_dp_size
                * math.ceil(self.w_prec / 4)
                * 4
                * self.pe_array_dim["h"]
                / 2
            )
        else:
            w_bandwidth = (
                self.pe_dp_size
                * math.ceil(self.w_prec / 4)
                * 4
                * self.pe_array_dim["h"]
            )
        w_sram_bank = 8
        w_sram_config = {
            "technology": 0.028,  # 工艺节点 28nm
            "mem_type": "ram",  # 内存类型
            "size": 512 * 1024 * 8,  # 内存总大小 512KB(bit -> byte转换)
            "bank_count": w_sram_bank,  # bank数量
            "rw_bw": w_bandwidth,  # 读写带宽 (bit/cycle)
            "r_port": 1,
            "w_port": 1,
            "rw_port": 0,
        }
        self.w_sram = MemoryInstance(
            w_sram_config,
            r_cost=0,
            w_cost=0,
            latency=1,
            min_r_granularity=None,
            min_w_granularity=64,
            get_cost_from_cacti=True,
        )

        # 输入SRAM
        if self.is_bit_serial:
            i_bandwidth = self.pe_dp_size * self.i_prec * self.pe_array_dim["w"] / 2
        else:
            i_bandwidth = self.pe_dp_size * self.i_prec * self.pe_array_dim["w"]
        i_sram_bank = 8
        i_sram_config = {
            "technology": 0.028,
            "mem_type": "ram",
            "size": 512 * 1024 * 8,
            "bank_count": i_sram_bank,
            "rw_bw": i_bandwidth,
            "r_port": 1,
            "w_port": 1,
            "rw_port": 0,
        }
        self.i_sram = MemoryInstance(
            i_sram_config,
            r_cost=0,
            w_cost=0,
            latency=1,
            min_r_granularity=64,
            min_w_granularity=64,
            get_cost_from_cacti=True,
        )

        # DRAM
        dram_rw_bw = 128
        dram_config = {
            "technology": 0.028,
            "mem_type": "dram",
            "size": 1e9 * 8,  # 1GB
            "bank_count": 1,
            "rw_bw": dram_rw_bw,  # 128 bit/cycle
            "r_port": 0,
            "w_port": 0,
            "rw_port": 1,
        }
        wr_cost = dram_rw_bw / 64 * 1200
        self.dram = MemoryInstance(
            dram_config,
            r_cost=wr_cost,
            w_cost=wr_cost,
            latency=1,
            min_r_granularity=dram_rw_bw,
            min_w_granularity=dram_rw_bw,
            get_cost_from_cacti=False,
        )
