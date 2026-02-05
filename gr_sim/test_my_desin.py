# 基于: P=32, GroupSize=512, Decoded_Prec=4bit
my_decoder_config = {
    # --- 1. 算法参数 ---
    # 您的压缩脚本算出的平均传输位宽 (例如 4.25 bit)
    "transmission_prec": 4.25,
    # --- 2. 逻辑综合参数 (修正后) ---
    # 0.18 pJ/bit (因为分母变小了,能耗/bit 上升)
    "energy_per_bit": 0.18,
    # 0.0827 mm^2 (包含 Banks 和 Loaders)
    "area_logic": 0.0827,
    # 142.1 Gbps (用于计算是否卡流水线)
    "throughput_gbps": 142.1,
    # --- 3. 存储参数 (16KB SRAM) ---
    # 建议运行 get_small_buffer_params.py 确认 16KB 的具体数值
    # 估算值参考:
    "small_sram_area": 0.005,
    "small_sram_energy_per_access": 2.0,
}
