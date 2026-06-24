"""步骤4：基准模型直接在四个平台测试集上推理 (表VIII)，画各平台混淆矩阵(图5)。
产出: outputs/cross_baseline.json, figures/fig_cm_v1.pdf
对应论文: R5, 图5
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C, utils as U

plt.rcParams["axes.unicode_minus"] = False
# 平台英文名（图中显示用）
PLAT_EN = {"wb": "Weibo", "xhs": "Xiaohongshu", "tieba": "Tieba", "zhihu": "Zhihu"}


def load_test():
    df = pd.read_csv(C.TEST_FILE)
    df["label_binary"] = df["label_binary"].astype(int)
    return df


def main():
    tok, model = U.build_model(f"{C.OUT_DIR}/baseline_model")
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
        ax.set_title(f"{PLAT_EN[p]} (Baseline)")          # 英文标题
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Offensive"])       # 英文刻度
        ax.set_yticklabels(["Normal", "Offensive"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(f"{C.FIG_DIR}/fig_cm_v1.pdf", dpi=200)
    plt.close()
    U.save_json(res, f"{C.OUT_DIR}/cross_baseline.json")
    for p in C.PLATFORMS:
        print(p, {k: round(res[p][k], 4) for k in ["accuracy", "precision", "recall", "f1"]})


if __name__ == "__main__":
    main()