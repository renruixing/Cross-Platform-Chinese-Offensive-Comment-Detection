"""步骤5：双阈值硬样本筛选 + 二次微调，生成优化模型。
说明：硬样本集已在 final_dataset/train_hardsamples_400.csv 固化(400条, 含train/val切分)。
本脚本(1)演示双阈值筛选逻辑如何从伪标注池产生候选(可复现筛选流程)，
(2)用固化的400条硬样本做二次微调，保存优化模型。
产出: outputs/optimized_model/
对应论文: IV.C (筛选机制) + 生成优化模型供步骤6使用
"""
import pandas as pd, glob, os
import config as C, utils as U


def demo_dual_threshold_selection():
    """复现双阈值筛选：对伪标注池打分，按 c>=ALPHA(高置信错误) / c<=BETA(低置信) 分池。
    这里用基准模型重新推理池子，输出两个候选池的规模，证明筛选流程可复现。"""
    tok, model = U.build_model(f"{C.OUT_DIR}/baseline_model")
    high, low = {}, {}
    for p in C.PLATFORMS:
        f = os.path.join(C.POOL_DIR, f"{p}.csv")
        d = pd.read_csv(f)
        ds = U.TextDS(d.text.tolist(), [0] * len(d), tok, C.FT_HP["max_len"])
        preds, confs = U.predict(model, ds, return_conf=True)
        d = d.assign(pred=preds, conf=confs)
        # 注：真正的“错误”需对照人工标签；此处用 pseudo_label 与 pred 不一致近似演示
        d["pl"] = d["pseudo_label"].astype(int)
        wrong = d[d.pred != d.pl]
        high[p] = int((wrong.conf >= C.ALPHA).sum())
        low[p] = int((wrong.conf <= C.BETA).sum())
    print("高置信错误池规模:", high)
    print("低置信错误池规模:", low)
    return high, low


def finetune_on_hard():
    U.set_seed(42)
    tok, model = U.build_model(f"{C.OUT_DIR}/baseline_model")  # 在基准上增量微调
    H = pd.read_csv(C.HARD_FILE)
    H["label_binary"] = H["label_binary"].astype(int)
    tr = H[H.split == "train"]; va = H[H.split == "val"]
    tr_ds = U.TextDS(tr.text.tolist(), tr.label_binary.tolist(), tok, C.FT_HP["max_len"])
    va_ds = U.TextDS(va.text.tolist(), va.label_binary.tolist(), tok, C.FT_HP["max_len"])
    U.train_loop(model, tr_ds, va_ds, hp=C.FT_HP)
    os.makedirs(f"{C.OUT_DIR}/optimized_model", exist_ok=True)
    model.save_pretrained(f"{C.OUT_DIR}/optimized_model")
    tok.save_pretrained(f"{C.OUT_DIR}/optimized_model")
    print("optimized model saved")


if __name__ == "__main__":
    demo_dual_threshold_selection()
    finetune_on_hard()
