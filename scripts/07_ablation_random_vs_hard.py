"""步骤7（核心新增）：消融实验 —— 随机320 vs 硬挖320。
目的：证明跨平台提升来自“硬样本挖掘策略”，而非单纯“多了320条标注数据”。
设计：
  - Baseline：不二次微调（步骤4结果）
  - Random-320：从候选池随机抽320条人工标注样本微调
  - Hard-320 ：双阈值硬样本筛选的320条微调
三组相同超参/验证集/测试集，各跑 len(SEEDS) 个种子，报 F1 均值±标准差。
产出: outputs/ablation.json
对应论文: R7

【重要】Random-320 需要“人工标注”的随机样本才能与 Hard-320 公平对比（都是人工标签）。
"""
import os, json, numpy as np, pandas as pd
import config as C, utils as U
from importlib import import_module
load_test = import_module("04_eval_crossplatform").load_test

RANDOM_FILE = os.path.join(C.DATA_DIR, "random320_labeled.csv")     # 你标的随机组
HARD_RELABEL = os.path.join(C.DATA_DIR, "hard320_relabel.csv")       # 你重标的硬样本组


def load_labeled(path, name):
    """读取你标注的 csv，校验 label_binary 已填。"""
    d = pd.read_csv(path)
    d["label_binary"] = d["label_binary"].astype(str).str.strip()
    bad = (~d["label_binary"].isin(["0", "1"])).sum()
    if bad > 0:
        print(f"!! {name} 还有 {bad} 条未标注（label_binary 须为 0/1），请先标注。")
        return None
    d["label_binary"] = d["label_binary"].astype(int)
    return d


def eval_all_platforms(model, tok, df):
    f1s = []
    for p in C.PLATFORMS:
        sub = df[df.platform == p]
        ds = U.TextDS(sub.text.tolist(), sub.label_binary.tolist(), tok, C.FT_HP["max_len"])
        f1s.append(U.evaluate(model, ds)["f1"])
    return float(np.mean(f1s)), f1s


def run_condition(train_df, seed):
    U.set_seed(seed)
    tok, model = U.build_model(f"{C.OUT_DIR}/baseline_model")
    ds = U.TextDS(train_df.text.tolist(), train_df.label_binary.astype(int).tolist(),
                  tok, C.FT_HP["max_len"])
    U.train_loop(model, ds, hp=C.FT_HP)
    return eval_all_platforms(model, tok, load_test())


def main():
    # 关键：Hard 与 Random 都用"你"重标的标签，两组标注同源、同质，
    # 唯一差别是"难例挖掘 vs 随机采样"。测试集对两组是同一把尺子，对称、不构成组间混淆。
    conditions = {}
    if os.path.exists(HARD_RELABEL):
        hard = load_labeled(HARD_RELABEL, "hard320_relabel.csv")
        if hard is not None:
            conditions["hard320"] = hard
    else:
        H = pd.read_csv(C.HARD_FILE); H["label_binary"] = H["label_binary"].astype(int)
        conditions["hard320"] = H[H.split == "train"]
        print("提示：未找到 hard320_relabel.csv，暂用原始硬样本标签；建议重标以与 Random 同源。")

    if os.path.exists(RANDOM_FILE):
        rnd = load_labeled(RANDOM_FILE, "random320_labeled.csv")
        if rnd is not None:
            conditions["random320"] = rnd.sample(320, random_state=0) if len(rnd) > 320 else rnd
    else:
        print("!! 未找到 random320_labeled.csv —— 请先完成标注。")

    out = {}
    for name, tdf in conditions.items():
        macro = []
        for s in C.SEEDS:
            mean_f1, per = run_condition(tdf, s)
            macro.append(mean_f1)
            print(f"{name} seed={s} macroF1={mean_f1:.4f}")
        out[name] = {"macro_f1_mean": float(np.mean(macro)),
                     "macro_f1_std": float(np.std(macro)),
                     "runs": macro}
    json.dump(out, open(f"{C.OUT_DIR}/ablation.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("R7 =", out)
    if "random320" in out and "hard320" in out:
        d = out["hard320"]["macro_f1_mean"] - out["random320"]["macro_f1_mean"]
        print(f"Hard - Random = {d:+.4f}  -> {'核心论点成立' if d > 0.01 else '增益不显著, 需诚实弱化结论'}")


if __name__ == "__main__":
    main()
