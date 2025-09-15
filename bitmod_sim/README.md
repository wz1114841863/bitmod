# BitMoD hardware simulator

```
# 1. Profile the LLM configuration and layer shape.
# The profiled information will be saved in a new folder **model_shape_config** under this directory.

bash run_shape_profile.sh

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
