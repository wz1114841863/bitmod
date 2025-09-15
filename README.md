## BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration

[源项目地址](https://github.com/abdelfattah-lab/BitMoD-HPCA-25)

# Code Structure
```
Repo Root
|---- SmoothQuant-BitMoD   # Running SmoothQuant with basic INT and our proposed BitMoD data types
|---- AWQ-BitMoD           # Running AWQ with basic INT and our proposed BitMoD data types
|---- OmniQuant-BitMoD     # Running OmniQuant with basic INT and our proposed BitMoD data types
|---- bitmod-quant         # Weight-only quantization with different precision and data types (e.g. INT, FP, BitMoD)
|---- bitmod-sim           # BitMoD accelerator simulator
```
# 代码调用流程
```
1. 搭建环境, 测试bitmod_quant只需要最常见的torch, transformers, datasets等库, 直接按照AWQ搭建环境
2. bitmod_quant/: 测试标准量化（伪量化）下采用不同的数据格式带来的困惑度差异并记录
3.
```
