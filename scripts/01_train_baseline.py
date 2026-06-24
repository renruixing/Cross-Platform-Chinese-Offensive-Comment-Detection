"""步骤1：在 COLD 上微调基准模型，画训练损失曲线(图2)与四指标曲线(图3)。
产出: outputs/baseline_source.json, figures/fig_loss.pdf, figures/fig_metrics.pdf
对应论文: R1, 图2, 图3
"""
import pandas as pd, matplotlib.pyplot as plt, numpy as np
import config as C, utils as U


def load_cold(path):
    df = pd.read_csv(path)
    # COLD 列名通常为 TEXT / label，按需调整
    text_col = "TEXT" if "TEXT" in df.columns else ("text" if "text" in df.columns else df.columns[0])
    label_col = "label" if "label" in df.columns else df.columns[-1]
    return df[text_col].astype(str).tolist(), df[label_col].astype(int).tolist()


def main():
    U.set_seed(42)
    tok, model = U.build_model()
    tr_x, tr_y = load_cold(C.COLD_TRAIN)
    dv_x, dv_y = load_cold(C.COLD_DEV)
    te_x, te_y = load_cold(C.COLD_TEST)
    ml = C.BASE_HP["max_len"]
    tr = U.TextDS(tr_x, tr_y, tok, ml)
    dv = U.TextDS(dv_x, dv_y, tok, ml)
    te = U.TextDS(te_x, te_y, tok, ml)

    print("\n" + "="*60)
    print("🚀 开始训练基准模型")
    print(f"📊 总步数: {C.BASE_HP['max_steps']}")
    print(f"📊 评估间隔: {C.BASE_HP['eval_steps']}")
    print("="*60 + "\n")

    loss_hist, metric_hist = U.train_loop(
        model, tr, dv, hp=C.BASE_HP,
        max_steps=C.BASE_HP["max_steps"], eval_steps=C.BASE_HP["eval_steps"])

    # 在 COLD 测试集上的最终性能 (R1)
    final = U.evaluate(model, te)
    U.save_json({"source_test": {k: final[k] for k in
                ["accuracy", "precision", "recall", "f1", "auc", "confusion"]}},
                f"{C.OUT_DIR}/baseline_source.json")
    model.save_pretrained(f"{C.OUT_DIR}/baseline_model"); tok.save_pretrained(f"{C.OUT_DIR}/baseline_model")

    # 图2：损失曲线（原始+平滑）
    steps = [s for s, _ in loss_hist]; losses = [l for _, l in loss_hist]
    sm = pd.Series(losses).rolling(50, min_periods=1).mean()
    plt.figure(figsize=(6, 4))
    plt.plot(steps, losses, alpha=.3, label="Original Loss")
    plt.plot(steps, sm, label="Smoothed Loss")
    plt.xlabel("Steps"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{C.FIG_DIR}/fig_loss.pdf", dpi=200); plt.close()

    # 图3：四指标曲线
    ms = [s for s, _ in metric_hist]
    plt.figure(figsize=(6, 4))
    for key, lab in [("accuracy", "Accuracy"), ("f1", "F1 Score"),
                     ("precision", "Precision"), ("recall", "Recall")]:
        plt.plot(ms, [m[key] for _, m in metric_hist], marker="o", ms=3, label=lab)
    plt.xlabel("Steps"); plt.ylabel("Score"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{C.FIG_DIR}/fig_metrics.pdf", dpi=200); plt.close()
    print("R1 =", final["accuracy"], final["precision"], final["recall"], final["f1"])


if __name__ == "__main__":
    main()
