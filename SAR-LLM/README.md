# SAR-LLM

SAR-LLM is a two-stage remote sensing language model pipeline built on top of pre-extracted SAR features and Qwen3-VL.

- `stage1`: caption pretraining with a frozen Qwen3-VL and a trainable SAR feature projector.
- `stage2`: VQA-style instruction tuning on top of the stage1 projector, with optional LoRA on the LLM.

The training layout has been refactored to follow the `RS-MLLM(multi-sensor)` style:

- `src/configs`: stage config entry points
- `src/dataset`: dataset entry points
- `src/model`: shared projector and Qwen3-VL wrapper
- `src/trainer`: stage-specific trainers
- `train/main.py`: unified training entrypoint

Legacy scripts under `train/stage1` and `train/stage2` are kept as compatibility wrappers.

## Project Layout

```text
SAR-LLM/
├── README.md
├── data_cache/
├── evaluate/
│   ├── stage1/
│   └── stage2/
├── src/
│   ├── configs/
│   ├── dataset/
│   ├── model/
│   └── trainer/
└── train/
    ├── main.py
    ├── stage1/
    └── stage2/
```

## Training

Use the unified entrypoint:

```bash
python train/main.py --stage stage1
python train/main.py --stage stage2
```

Compatibility entrypoints are still available:

```bash
python train/stage1/train_pt.py
python train/stage2/train_stage2.py
```

## Stage1

Stage1 reads pre-extracted SAR `.pt` tokens and trains only the projector:

- input: SARCLIP token features shaped like `(195, 768)`
- model: `TokenLinearProjector -> SarQwenVLForCausalLM`
- objective: caption modeling on mixed SAR caption datasets

Primary config file:

- [train/stage1/config_local.py](/home/zhuxiang/RS/SAR-LLM/train/stage1/config_local.py)

Core trainer:

- [src/trainer/stage1_trainer.py](/home/zhuxiang/RS/SAR-LLM/src/trainer/stage1_trainer.py)

## Stage2

Stage2 initializes from the stage1 projector and performs VQA/instruction tuning:

- loads the stage1 projector checkpoint
- optionally applies LoRA to Qwen3-VL
- trains on mixed VQA-style SAR datasets

Primary config file:

- [train/stage2/config_local.py](/home/zhuxiang/RS/SAR-LLM/train/stage2/config_local.py)

Core trainer:

- [src/trainer/stage2_trainer.py](/home/zhuxiang/RS/SAR-LLM/src/trainer/stage2_trainer.py)

## Evaluation

Evaluation scripts remain under `evaluate/`:

- [evaluate/stage1/test.py](/home/zhuxiang/RS/SAR-LLM/evaluate/stage1/test.py)
- [evaluate/stage1/test_pt.py](/home/zhuxiang/RS/SAR-LLM/evaluate/stage1/test_pt.py) (legacy compatibility)
- [evaluate/stage2/test_stage2.py](/home/zhuxiang/RS/SAR-LLM/evaluate/stage2/test_stage2.py)

## Notes

- `src/` is now the main source-of-truth training layout.
- `train/stage1/train_pt.py` and `train/stage2/train_stage2.py` are thin wrappers for backward compatibility.
- Configs are still edited in the original `train/stage1/config_local.py` and `train/stage2/config_local.py` files.
