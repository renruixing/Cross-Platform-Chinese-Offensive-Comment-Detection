"""步骤6：优化模型在四平台测试集上推理，与基准对比 (表X)，画优化后混淆矩阵(图7)。
产出: outputs/cross_optimized.json, figures/fig_cm_v2.png
对应论文: R6, 图7
注：图中文字一律用英文，避免 matplotlib 缺中文字体导致显示空白。
"""
import json, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C, utils as U
from importlib import import_module
load_test = import_module("04_eval_crossplatform").load_test

plt.rcParams["axes.unicode_minus"] = False
PLAT_EN = {"wb": "Weibo", "xhs": "Xiaohongshu", "tieba": "Tieba", "zhihu": "Zhihu"}


def main():
    tok, model = U.build_model(f"{C.OUT_DIR}/optimized_model")
    df = load_test()
    res = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, p in zip(axes.ravel(), C.PLATFORMS):
        sub = df[df.platform == p]
        ds = U.TextDS(sub.text.tolist(), sub.label_binary.tolist(), tok, C.FT_HP["max_len"])
        m = U.evaluate(model, ds)
        res[p] = {k: m[k] for k in ["accuracy", "precision", "recall", "f1", "confusion"]}
        cm = np.array(m["confusion"])
        ax.imshow(cm, cmap="YlOrBr")
        ax.set_title(f"{PLAT_EN[p]} (Optimized)")        # 英文标题
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Offensive"])      # 英文刻度
        ax.set_yticklabels(["Normal", "Offensive"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(f"{C.FIG_DIR}/fig_cm_v2.pdf", dpi=200)   # 要矢量PDF就改成 fig_cm_v2.pdf
    plt.close()

    # 合并基准 vs 优化对比表
    base = json.load(open(f"{C.OUT_DIR}/cross_baseline.json"))
    table = {}
    for p in C.PLATFORMS:
        table[p] = {"baseline": {k: base[p][k] for k in ["accuracy", "precision", "recall", "f1"]},
                    "optimized": {k: res[p][k] for k in ["accuracy", "precision", "recall", "f1"]}}
    U.save_json({"per_platform": res, "comparison": table}, f"{C.OUT_DIR}/cross_optimized.json")
    for p in C.PLATFORMS:
        print(p, "F1", round(base[p]["f1"], 4), "->", round(res[p]["f1"], 4))


if __name__ == "__main__":
    main()