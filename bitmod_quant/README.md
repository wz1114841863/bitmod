
# Weight-only quantization using different datatypes

使用wikitext数据集计算困惑度
```
python llm_eval_wikitext.py \
  --model ${model} \
  --wq_bits ${quant_precision} \
  --wq_datatype ${quant_datatype} \
  --wq_groupsize ${quant_groupsize}
```

使用C4数据集计算困惑度
```
python llm_eval_c4.py \
  --model ${model} \
  --wq_bits ${quant_precision} \
  --wq_datatype ${quant_datatype} \
  --wq_groupsize ${quant_groupsize}
```

## Supported Data Types:

| **Data Type**              | Definition                                                                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **fp16**                   | The baseline FP16 model without quantization                                                                                                                  |
| **int{n}**                 | Symmetric integer quantization, {n} should be replaced by the precision (e.g., **int4**)                                                                      |
| **int{n}_asym**            | Asymmetric integer quantization, {n} should be replaced by the precision (e.g., **int4_asym**)                                                                |
| **fp5_e2m2**, **fp5_e3m1** | FP5 datatype with 2-bit exponent 2-bit mantissa, or 3-bit exponent 1-bit mantissa                                                                             |
| **fp6_e2m3**, **fp6_e3m2** | FP6 datatype with 2-bit exponent 3-bit mantissa, or 3-bit exponent 2-bit mantissa                                                                             |
| **mx_fp3**, **mx_fp4**     | [Microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) with element datatype **fp3** and **fp4**, respectively |
| **mixed_ant**              | [ANT](https://arxiv.org/abs/2208.14286) datatype, currently support 3-bit and 4-bit ANT                                                                       |
| **mixed_er**               | FP4 and FP3 datatype with extra resolution, i.e., FP4_ER and FP3_ER in our BitMoD paper                                                                       |
| **mixed_ea**               | FP4 and FP3 datatype with extra asymmetry, i.e., FP4_EA and FP3_EA in our BitMoD paper                                                                        |
| **mixed_bitmod**           | Our 4-bit and 3-bit BitMoD datatype with both extra resolution and extra asymmetry                                                                            |

## Supported Precision:
The following table shows the supported data types at different precision
| **Precision** | Supported Data Type                                                  |
| ------------- | -------------------------------------------------------------------- |
| 16            | fp16                                                                 |
| 8             | int8, int8_asym                                                      |
| 7             | int7, int7_asym                                                      |
| 6             | int6, int6_asym, fp6_e2m3, fp6_e3m2                                  |
| 5             | int5, int5_asym, fp5_e2m2, fp5_e3m1                                  |
| 4             | int4, int4_asym, mx_fp4, mixed_ant, mixed_er, mixed_ea, mixed_bitmod |
| 3             | int3, int3_asym, mx_fp3, mixed_ant, mixed_er, mixed_ea, mixed_bitmod |

## example

```
python llm_eval_wikitext.py \
    --model "facebook/opt-125m" \
    --wq_bits "6" \
    --wq_datatype "int6" \
    --wq_groupsize "128"

python llm_eval_wikitext.py --model "facebook/opt-125m" --wq_bits 8 --wq_datatype "int8_asym" --wq_groupsize 128
```
