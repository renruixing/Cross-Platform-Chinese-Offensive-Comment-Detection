"""步骤7b（稳健性检验）：控制类别比例与规模后的 Hard vs Random。
动机：实测 Hard-320 冒犯占比 44.4%、Random-320 占比 29.1%，存在类别比例混淆。
本脚本把两组**下采样到相同的 (正,负) 计数 = (93, 178)**（两组都能供给的上限），
规模与比例完全一致，唯一差别仍是"难例 vs 随机"。各跑 SEEDS 个种子。
若 Hard 仍显著高于 Random，则"Hard 赢只是因为正样本多"的质疑被排除。
产出: outputs/ablation_balanced.json
对应论文: IV.E 稳健性脚注
"""
import os, json, numpy as np, pandas as pd
import config as C
from importlib import import_module
abl = import_module("07_ablation_random_vs_hard")
load_test = import_module("04_eval_crossplatform").load_test

N_POS, N_NEG = 93, 178   # 两组共同可供给的计数

def subsample(d, seed):
    pos = d[d.label_binary == 1]; neg = d[d.label_binary == 0]
    rng = np.random.default_rng(seed)
    pos = pos.iloc[rng.choice(len(pos), N_POS, replace=False)]
    neg = neg.iloc[rng.choice(len(neg), N_NEG, replace=False)]
    return pd.concat([pos, neg])

def load(path):
    d = pd.read_csv(path); d["label_binary"] = d["label_binary"].astype(str).str.strip()
    assert d["label_binary"].isin(["0", "1"]).all(), f"{path} 有未标注样本"
    d["label_binary"] = d["label_binary"].astype(int); return d

def main():
    hard = load(os.path.join(C.DATA_DIR, "hard320_relabel.csv"))
    rnd  = load(os.path.join(C.DATA_DIR, "random320_labeled.csv"))
    test = load_test()
    out = {}
    for name, d in [("hard_bal", hard), ("random_bal", rnd)]:
        macro = []
        for s in C.SEEDS:
            sub = subsample(d, s)
            mean_f1, per = abl.run_condition(sub, s)
            macro.append(mean_f1)
            print(f"{name} seed={s} macroF1={mean_f1:.4f} per={[round(x,3) for x in per]}")
        out[name] = {"macro_f1_mean": float(np.mean(macro)),
                     "macro_f1_std": float(np.std(macro)), "runs": macro}
    json.dump(out, open(f"{C.OUT_DIR}/ablation_balanced.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    d = out["hard_bal"]["macro_f1_mean"] - out["random_bal"]["macro_f1_mean"]
    print(f"\n[平衡后] Hard - Random = {d:+.4f} "
          f"-> {'核心结论稳健(非比例所致)' if d > 0.01 else '平衡后差异消失, 需重新解读'}")


if __name__ == "__main__":
    main()
