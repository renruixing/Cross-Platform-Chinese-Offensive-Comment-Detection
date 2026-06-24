"""步骤2：基准模型在 COLD 源域测试集上的混淆矩阵 + ROC (图4)。
产出: outputs/baseline_source_eval.json, figures/fig_cm_roc.png
对应论文: R2(混淆矩阵), R3(AUC), 图4
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import config as C, utils as U
from importlib import import_module
load_cold = import_module("01_train_baseline").load_cold


def main():
    tok, model = U.build_model(f"{C.OUT_DIR}/baseline_model")
    te_x, te_y = load_cold(C.COLD_TEST)
    te = U.TextDS(te_x, te_y, tok, C.BASE_HP["max_len"])
    m = U.evaluate(model, te)
    cm = np.array(m["confusion"])  # [[TN,FP],[FN,TP]]
    U.save_json({"confusion": m["confusion"], "auc": m["auc"],
                 "accuracy": m["accuracy"], "precision": m["precision"],
                 "recall": m["recall"], "f1": m["f1"]},
                f"{C.OUT_DIR}/baseline_source_eval.json")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    im = ax[0].imshow(cm, cmap="Blues")
    ax[0].set_xticks([0, 1]); ax[0].set_yticks([0, 1])
    ax[0].set_xticklabels(["Normal", "Offensive"]); ax[0].set_yticklabels(["Normal", "Offensive"])
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, cm[i, j], ha="center", va="center")
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True")
    fpr, tpr, _ = roc_curve(m["_y"], m["_prob1"])
    ax[1].plot(fpr, tpr, color="orange", label=f"ROC (AUC = {m['auc']:.4f})")
    ax[1].plot([0, 1], [0, 1], "--", color="navy")
    ax[1].set_xlabel("False Positive Rate"); ax[1].set_ylabel("True Positive Rate"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{C.FIG_DIR}/fig_cm_roc.pdf", dpi=200); plt.close()
    print("R2 confusion =", m["confusion"], " R3 AUC =", m["auc"])


if __name__ == "__main__":
    main()
