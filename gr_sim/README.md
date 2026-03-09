# Group Decoder Hardware Simulator

```
# 1. Profile the LLM configuration and layer shape.
# The profiled information will be saved in a new folder **model_shape_config** under this directory.
python llm_shape_profile.py --model [model_name]

# 2. Get the latency and energy of different models for discriminative and generative tasks.
# --is_generation: optional, evaluate the hardware performance of generative / discriminative tasks.
# --is_lossless: optional, evaluate the hardware performance of lossless / lossy BitMoD quantization.
# 对于ANT, Olive， bitmod使用提前计算好的w_prec(The weight precision)来进行计算
# 整体来说算是从理论上来进行计算分析

python test_baseline.py --is_generation                  # Baseline FP16 accelerator
python test_ant.py      --is_generation                  # ANT accelerator
python test_olive.py    --is_generation                  # OliVe accelerator
python test_bitmod.py   --is_generation --is_lossless    # BitMoD accelerator

```
思路：
bitmod是对整个推理过程（两个阶段）的仿真估算，现在我要将我的解码器模块添加进去，解码器主要有五个模块（1）核心的解码模块DecoderBank，（2）为脉动阵列提供的元数据加载模块 MetadataLoader （3）为脉动阵列提供的解压缩后的权重加载模块WeightLoader （4）元数据存储模块MetaSRAMBuffer （5）解压后权重存储模块 WeightSRAM。其中（1）（2）（3）的相关面积、功耗和频率由DC仿真得到，（4）（5）由Cacti计算得到。然后我们要把解码器的整个流程添加到Accelerator中去得到DecoderAccelerator。


DC 综合报告中的功耗（Power, 单位 mW）是“全部的”（Total Power），绝不是 per-bit 的。

物理含义：功耗是单位时间内消耗的能量（$1\text{ mW} = 1\text{ mJ / second}$）。DC 报告中的 Power Dynamic (mW) 是指在特定的时钟频率（例如 $1\text{ GHz}$）和特定的翻转率（Toggle Rate，通常 DC 默认设为 0.1 或 0.2）下，整个模块运行时的总动态功耗。转换关系：我们要把它变成仿真器能用的 energy_per_bit (pJ/bit)，必须除以吞吐率（Gbps，即 $10^9\text{ bits / second}$）。

核心逻辑：将元数据的开销，均摊到每一个被压缩的权重比特上。

既然元数据是依附于权重存在的（每 512 个权重固定搭配一组元数据），我们可以将 DecoderBank、WeightLoader 和 MetadataLoader 视为一个不可分割的“黑盒解码子系统”。

具体操作步骤如下：

1）求总动态功耗 (Total Dynamic Power)：直接将 DC 报告中这三个模块在相同 $P$ 值下的 Power Dynamic (mW) 相加。总动态功耗 = $5.653 + 0.210 + 0.143 = \mathbf{6.006\text{ mW}}$。

2）定义系统的统一吞吐率 (System Throughput)：使用主数据流（权重压缩流）的吞吐率作为系统的吞吐率。系统吞吐率 = $P \times \text{每个单元每周期解压比特数} \times \text{频率}$（例如 $32\text{ Gbps}$）。计算统一的 energy_per_bit：$\text{energy\_per\_bit} = \frac{6.006\text{ mW}}{32\text{ Gbps}} = \mathbf{0.187\text{ pJ/bit}}$。

注：
必须使用 Power Dynamic (动态功耗)
在计算 energy_per_bit (pJ/bit) 时，只能且必须使用 Power Dynamic (mW)。

Power Dynamic (动态功耗)：是由于电路翻转（Processing Data）产生的功耗。它与你处理的数据量成正比，完美契合 energy_per_bit × total_bits 这个动态能耗计算公式。

Power Leakage (静态/漏电功耗)：只要芯片通电就会产生，无论它是不是在解码。它的耗能公式应该是：Leakage Power × Total Execution Time (总运行时间)，它与吞吐率和比特数无关。

Power Total = Dynamic + Leakage。如果用 Total Power 去除以吞吐率算 energy_per_bit，等于把随时间流逝的静态功耗强行平摊到了每 bit 数据上，这在仿真物理模型上是完全错误的。


使用均摊法（Amortization）：根据每次访问（Access）所覆盖的权重数量，将 pJ/access 折算为 pJ/weight，然后再相加。

具体的推导和计算步骤：

1. 计算 WeightSRAM 的单权重能耗CACTI 数据：2.090436 pJ/access。物理含义：这是对 32-bit 总线进行一次读或写的能耗。涵盖权重数：由于你的解压权重是 4-bit，一次 32-bit 的访问实际上包含了 $32 \div 4 = \mathbf{8\text{ 个权重}}$。折算 (Per-weight)：WeightSRAM_Energy_per_weight = $2.090436 \div 8 \approx \mathbf{0.2613\text{ pJ/weight}}$。

2. 计算 MetaSRAMBuffer 的单权重能耗CACTI 数据：6.499855 pJ/access。物理含义：跑 CACTI 时的 bus_width 只是单组元数据，那么这一次访问涵盖了 1 个 Group（512 个权重）。折算 = $6.499855 \div 512 \approx \mathbf{0.0127\text{ pJ/weight}}$。

3. 合并参数填入 decoder_config将两者的 pJ/weight 相加，就是你应该填入 small_sram_energy_per_access 的最终值。总计约为：$0.2613 + 0.0127 = \mathbf{0.274\text{ pJ/weight}}$
