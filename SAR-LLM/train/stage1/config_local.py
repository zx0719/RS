from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LocalPaths:
    # ---------- Qwen ----------
    qwen_path: str = "/mnt/data/zhuxiang/Qwen/Qwen3-VL-4B-Instruct"

    # ---------- SARLANG ----------
    sarlang_root: str          = "/mnt/data/mm_data/SAR/SARLANG-1M"
    sarlang_pt_train_json: str = "Text/Caption/train/Caption_train_SARLANG-1M_pt.json"
    sarlang_pt_test_json: str  = "Text/Caption/test/Caption_test_SARLANG-1M_pt.json"
    sarlang_pt_root: str       = "/mnt/data/mm_data/SAR/SARLANG-1M/pt_cache"

    # ---------- SARCAP ----------
    sarcap_root: str          = "/mnt/data/mm_data/SAR/SARCAP"
    sarcap_pt_train_json: str = "SARCAPtrain_pt.json"
    sarcap_pt_test_json: str  = "SARCAPtest_pt.json"
    sarcap_pt_root: str       = "/mnt/data/mm_data/SAR/SARCAP/pt_cache"

    # ---------- SARTEXT ----------
    sartext_root: str          = "/mnt/data/mm_data/SAR/SARTEXT/SAR-TEXT-data"
    sartext_pt_train_json: str = "SARTEXTtrain_pt.json"
    sartext_pt_test_json: str  = "SARTEXTtest_pt.json"
    sartext_pt_root: str       = "/mnt/data/mm_data/SAR/SARTEXT/SAR-TEXT-data/pt_cache"

    # ---------- FSAR-Cap ----------
    fsarcap_root: str          = "/mnt/data/mm_data/SAR/FSAR-Cap"
    fsarcap_pt_train_json: str = "FSAR-Captrain_pt.json"
    fsarcap_pt_val_json: str   = "FSAR-Capval_pt.json"
    fsarcap_pt_test_json: str  = "FSAR-Captest_pt.json"
    fsarcap_pt_root: str       = "/mnt/data/mm_data/SAR/FSAR-Cap/pt_cache"

    # ---------- 数据集截断（None 表示不截断）----------
    sarlang_max_samples:  int | None = None
    sarcap_max_samples:   int | None = None
    sartext_max_samples:  int | None = None
    fsarcap_max_samples:  int | None = None


@dataclass
class TrainConfig:
    # 固定使用混训模式，通过 mixed_weight_* 控制各数据集比例（设为 0 即排除）
    stage: str = "stage1_mixed_caption"

    # 视觉 token 个数（与预提取 .pt 文件一致，不可随意修改）
    num_image_tokens: int = 195

    # 训练超参
    batch_size: int = 12      # 单卡 A800 80GB；vocab=151936 logits 很大，12 是安全上限
    lr: float = 1e-5
    max_steps: int = 22000    # Stage B: 从 step=14000 续训，再跑 8000 步
    max_length: int = 512

    grad_clip_norm: float = 1.0


    # 设备 / 精度
    device: str = "cuda:0"
    fp16: bool  = True       # 开启 AMP 混合精度（GradScaler + autocast）
    # fp16: bool  = False

    # 数值稳定 / 调试（NaN 修复已验证，关闭 debug 减少开销）
    debug_nan: bool = False
    debug_print_every: int = 10
    debug_param_check_after_step: bool = False
    debug_feat_absmax_threshold: float = 1e4
    debug_embed_absmax_threshold: float = 1e4

    # 单卡模式：use_multi_gpu=False，Qwen 和 projector 都在 cuda:0
    use_multi_gpu: bool    = False
    qwen_device_map: str   = "cuda:0"
    main_device: str       = "cuda:0"

    # 节省激活显存（以重算时间换空间，约减少 40~60% 激活显存）
    # gradient_checkpointing: bool = True
    gradient_checkpointing: bool = False

    # DataLoader
    num_workers: int        = 8
    persistent_workers: bool = True
    prefetch_factor: int    = 4
    shuffle: bool           = True
    drop_last: bool         = False

    # swanlab 日志 
    use_swanlab: bool              = True
    swanlab_project: str           = "SARCLIP-Qwen"
    swanlab_experiment_name: str   = "stage1_mixed_run"

    # 保存
    save_dir: str  = "/mnt/data/qianwentao/checkpoints_sarqwen"
    save_name: str = "sar_projector_stage1_sarcap.pt"

    # 断点续训（完整 checkpoint，含 optimizer/scaler 状态）
    resume_ckpt: str | None    = "/mnt/data/qianwentao/checkpoints_sarqwen/checkpoint_step_014000.pt"
    # resume_ckpt: str | None    = None
    save_every_steps: int      = 1000  # 原来 100 步一存，20000 步会产生 200 个文件
    save_full_checkpoint: bool = True
    keep_last_n_checkpoints: int = 10   # 配合 save_every_steps=1000 即可

    # 从已有 projector 接着训（只加载 projector 权重，optimizer 重置）
    projector_ckpt: str | None = None
    # projector_ckpt: str | None = "/mnt/data/qianwentao/checkpoints_sarqwen_2_nograd/sar_projector_stage1_sarcap.pt"

    # BridgeGuidedProjector：填入线性度实验导出的 W 矩阵路径即可启用
    # 设为 None 则退回 TokenLinearProjector（兼容旧行为）
    bridge_ckpt: str | None = "/home/zhuxiang/RS/Connector/experiment/linearity_exp/results_v2/bridge_W_image_sarclip_qwen.pt"
    bridge_dim_hidden: int  = 1024   # residual MLP 隐层维度
    # Stage B：解冻 bridge，bridge 用 0.1x lr；需同时设置 resume_ckpt 从 Stage A checkpoint 续训
    stage_b_unfreeze_bridge: bool = True
    resume_projector_only: bool   = True   # Stage B 切换时 optimizer 重置

    # ===== 混训参数 =====
    # 将某个数据集权重设为 0 即可排除，无需修改 stage
    mixed_epoch_length: int    = 50000  # 原来 30000，增大以覆盖更多不同样本
    mixed_seed: int            = 998
    mixed_weight_sarlang: float = 1.0
    mixed_weight_sarcap: float  = 0.0
    mixed_weight_sartext: float = 1.0   # 原来 0.0，sartext 有 130000 条训练数据不应浪费
    mixed_weight_fsarcap: float = 0.0


PATHS = LocalPaths()
TRAIN = TrainConfig()
