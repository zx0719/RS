from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LocalPaths:
    # ---------- Qwen ----------
    qwen_path: str = "/mnt/data/zhuxiang/Qwen/Qwen3-VL-4B-Instruct"

    # ---------- Stage1 projector checkpoint ----------
    stage1_projector_ckpt: str = "/mnt/data/qianwentao/checkpoints_sarqwen/sar_projector_stage1_sarcap.pt"

    # ---------- SARTEXT VQA（多轮对话，MSAR-1.0 图片，pt 已提取）----------
    sarvqa_root: str          = "/home/zhuxiang/RS/SAR-LLM/data_cache"
    sarvqa_pt_train_json: str = "SAR-VQA_conv_train_pt.json"
    sarvqa_pt_test_json: str  = "SAR-VQA_conv_test_pt.json"

    # ---------- SARTEXT（单轮 caption，pt 已提取）----------
    sartext_root: str          = "/mnt/data/mm_data/SAR/SARTEXT/SAR-TEXT-data"
    sartext_pt_train_json: str = "SARTEXTtrain_pt.json"
    sartext_pt_test_json: str  = "SARTEXTtest_pt.json"

    # ---------- SARLANG-1M VQA（图片已解压，运行 get_pt_sarlang_vqa.py 提取特征后启用）----------
    # pt_cache 输出到 /mnt/data/mm_data/SAR/SARDet_100K/data/Images/pt_cache/
    sarlang_vqa_root: str          = "/mnt/data/mm_data/SAR/SARLANG-1M"
    sarlang_vqa_pt_train_json: str = "Text/VQA/train/SARVQA1_train_pt.json"
    sarlang_vqa_pt_test_json: str  = "Text/VQA/test/SARVQA1_test_pt.json"

    # ---------- 数据集截断（None 表示不截断）----------
    sarvqa_max_samples:      int | None = None
    sartext_max_samples:     int | None = None
    sarlang_vqa_max_samples: int | None = None


@dataclass
class TrainConfig:
    stage: str = "stage2_vqa"

    # 视觉 token 个数（与预提取 .pt 文件一致）
    num_image_tokens: int = 195

    # 训练超参（双卡 effective batch = 4*2=8，112275步 = 1 epoch全量数据）
    batch_size: int = 4
    lr: float = 2e-5
    lora_lr: float = 1e-4          # LoRA 学习率（从零开始，需要更大 lr）
    max_steps: int = 112275
    max_length: int = 1024         # VQA 多轮对话比 caption 长，适当增大

    grad_clip_norm: float = 1.0
    warmup_steps: int = 2000       # ~1.8% of max_steps

    # 设备 / 精度
    device: str = "cuda:0"
    fp16: bool  = False
    bf16: bool  = True

    # 双卡 DeepSpeed 模式
    use_multi_gpu: bool    = True
    qwen_device_map: str   = "cuda:0"
    main_device: str       = "cuda:0"

    gradient_checkpointing: bool = True   # 多轮长序列，节省显存

    # DeepSpeed（双卡时开启，单卡时关闭）
    use_deepspeed: bool = False  # 由启动脚本通过环境变量控制，不要手动改

    # BridgeGuidedProjector（与 Stage 1 保持一致，加载 Stage 1 训练好的权重）
    bridge_ckpt: str | None = "/home/zhuxiang/RS/Connector/experiment/linearity_exp/results_v2/bridge_W_image_sarclip_qwen.pt"
    bridge_dim_hidden: int  = 1024

    # LoRA 配置（stage2 微调 LLM）
    use_lora: bool         = True
    lora_r: int            = 16
    lora_alpha: int        = 32
    lora_dropout: float    = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # 数值稳定 / 调试（Stage 2 稳定后关闭 debug 减少开销）
    debug_nan: bool = False
    debug_print_every: int = 10
    debug_feat_absmax_threshold: float = 1e4
    debug_embed_absmax_threshold: float = 1e4

    # DataLoader
    num_workers: int         = 2
    persistent_workers: bool = True
    prefetch_factor: int     = 2
    shuffle: bool            = True
    drop_last: bool          = False

    # 混训权重（0 = 不使用）
    mixed_weight_sarvqa:      float = 1.0   # SARTEXT VQA 多轮对话（22,390条）
    mixed_weight_sartext:     float = 0.0   # SARTEXT caption 关掉，Stage2 专注 QA
    mixed_weight_sarlang_vqa: float = 1.0   # SARLANG-1M VQA（875,806条，pt 已提取）
    mixed_seed: int = 42

    # swanlab 日志
    use_swanlab: bool            = True
    swanlab_project: str         = "SARCLIP-Qwen"
    swanlab_experiment_name: str = "stage2_full_2gpu"

    # 保存
    save_dir: str  = "/mnt/data/zhuxiang/checkpoints_sarqwen_stage2"
    save_name: str = "sar_stage2_final.pt"

    resume_ckpt: str | None    = None
    save_every_steps: int      = 1000
    save_full_checkpoint: bool = True
    keep_last_n_checkpoints: int = 5


PATHS = LocalPaths()
TRAIN = TrainConfig()
