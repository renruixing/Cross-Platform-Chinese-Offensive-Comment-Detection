"""全局配置：路径与超参数（与论文表III/表IX一致）。改这里即可。"""
import os

# ---- 路径 ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "final_dataset")
OUT_DIR    = os.path.join(ROOT, "outputs")
FIG_DIR    = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TEST_FILE  = os.path.join(DATA_DIR, "test_crossplatform_3class.csv")
HARD_FILE  = os.path.join(DATA_DIR, "train_hardsamples_400.csv")
POOL_DIR   = os.path.join(DATA_DIR, "pool_pseudolabel")

# COLD 原始数据（用于步骤1基准微调）
COLD_TRAIN = os.path.join(DATA_DIR, "COLD", "train.csv")
COLD_DEV   = os.path.join(DATA_DIR, "COLD", "dev.csv")
COLD_TEST  = os.path.join(DATA_DIR, "COLD", "test.csv")

# 预训练权重（干净底座，未在COLD上训练过）
PRETRAINED = os.path.join(ROOT, "chinese-roberta-wwm-ext")   # 或本地路径
# 说明：原方案使用 roberta-base-cold（已在COLD上训练），会使基准源域分数虚高、
# 与传统ML对照不公平。改用未见COLD的干净底座从零在COLD上微调，0.93才是诚实的微调结果。

# ---- 基准微调超参（论文表III）----
BASE_HP = dict(
    batch_size=32, max_steps=10000, eval_steps=500,
    hidden=768, lr=2e-5, optimizer="AdamW",
    max_len=128, weight_decay=0.005,
)

# ---- 硬样本/消融微调超参（论文表IX）----
FT_HP = dict(
    epochs=10, batch_size=16, lr=2e-5, weight_decay=0.01,
    max_len=128, eval_strategy="epoch",
)

# ---- 双阈值（论文 IV.C）----
ALPHA = 0.85   # 高置信阈值
BETA  = 0.60   # 低置信阈值

# 消融重复种子
SEEDS = [42, 1, 2, 3, 4]

PLATFORMS = ["wb", "xhs", "tieba", "zhihu"]
PLAT_CN   = {"wb": "新浪微博", "xhs": "小红书", "tieba": "百度贴吧", "zhihu": "知乎"}

DEVICE = "cuda"
