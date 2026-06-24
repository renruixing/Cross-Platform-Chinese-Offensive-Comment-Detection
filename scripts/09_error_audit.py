"""步骤9：错例审计 —— 抽样“模型预测≠人工标签”的样本，人工裁决是模型错还是标签错。
目的：量化测试集标注错误率，给数据质量背书 / 或暴露需清洗。
产出: outputs/error_audit.json + audit_sheet.csv(待人工填裁决)
对应论文: R9
流程：
  1. 本脚本生成 audit_sheet.csv（含模型预测、人工标签、置信度），抽样 N 条分歧样本。
  2. 你人工逐条裁决，在 verdict 列填：model_wrong / label_wrong / ambiguous
  3. 重新运行本脚本，它会读回 verdict 统计错误率。
"""
import os, json, pandas as pd, numpy as np
import config as C, utils as U
from importlib import import_module
load_test = import_module("04_eval_crossplatform").load_test

SHEET = f"{C.OUT_DIR}/audit_sheet.csv"
N_AUDIT = 200


def generate():
    tok, model = U.build_model(f"{C.OUT_DIR}/baseline_model")
    df = load_test().reset_index(drop=True)
    ds = U.TextDS(df.text.tolist(), df.label_binary.astype(int).tolist(), tok, C.FT_HP["max_len"])
    preds, confs = U.predict(model, ds, return_conf=True)
    df["pred"] = preds; df["conf"] = confs
    dis = df[df.pred != df.label_binary.astype(int)].copy()
    # 分层抽样：按平台 + 人工标签
    rng = np.random.default_rng(42)
    picks = []
    for (p, l), g in dis.groupby(["platform", "label"]):
        k = min(len(g), max(3, N_AUDIT // 12))
        picks.append(g.sample(k, random_state=int(l) * 10 + len(p)))
    sheet = pd.concat(picks).head(N_AUDIT)
    sheet["verdict"] = ""  # 待人工填
    sheet[["id", "platform", "text", "label", "pred", "conf", "verdict"]].to_csv(
        SHEET, index=False, encoding="utf-8-sig")
    print(f"已生成 {SHEET}（{len(sheet)}条）。请在 verdict 列填 model_wrong/label_wrong/ambiguous 后重跑。")


def summarize():
    s = pd.read_csv(SHEET, encoding="utf-8-sig")
    if s["verdict"].fillna("").eq("").all():
        print("verdict 列尚未填写，先人工裁决。"); return
    vc = s["verdict"].value_counts().to_dict()
    n = int(s["verdict"].notna().sum())
    label_wrong = vc.get("label_wrong", 0)
    out = {"audited": n, "verdict_counts": vc,
           "label_error_rate_in_disagreements": label_wrong / max(n, 1)}
    json.dump(out, open(f"{C.OUT_DIR}/error_audit.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("R9 =", out)


if __name__ == "__main__":
    if os.path.exists(SHEET):
        summarize()
    else:
        generate()
