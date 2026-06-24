"""步骤11（案例验证）：用当前的 baseline_model 与 optimized_model 对典型案例预测，
核对「基准判正常、优化判冒犯」等对比在新模型（chinese-roberta-wwm-ext）下是否成立。
产出: outputs/case_study.json + 终端打印
对应论文: F 节 典型案例表（反讽/隐喻、语境敏感词）

用法：把你论文里要用的案例文本填进下面 CASES（text 必填，platform/note 可选），
然后 python 11_case_study.py。脚本会输出每条的 基准预测 / 优化预测 / 各自置信度。
"""
import json, torch
import config as C, utils as U

# ====== 在这里填你的典型案例（均取自本文数据集真实样本）======
CASES = [
    # —— 第一类：反讽 / 隐喻 / 反问（期望 基准=0 漏判，优化=1 正确）——
    {"text": "都怪我我应该送花圈。这样以后也能用得上",
     "platform": "tieba", "note": "案例1 反讽自责"},
    {"text": "那你知道自己最后还是要嘎掉的, 怎么不天天随身携带棺材",
     "platform": "wb", "note": "案例2 极端类比隐喻"},
    {"text": "你的资本就是身后挺你的那帮人穷志不穷的兄弟啊, 不然你这屁大点地方能有啥资本？？？",
     "platform": "zhihu", "note": "案例3 反问式贬低"},

    # —— 第二类：语境敏感词成对样本（同词、极性相反；中性应判0，隐性有害应判1）——
    {"text": "是暂时管不了，但是这是普通民众朴素的愿望",
     "platform": "zhihu", "note": "普通-1a 中性"},
    {"text": "作者外国男朋友的特征，如果出现在国男身上，就不是优点了，毕竟这些特征那么普通又那么自信",
     "platform": "zhihu", "note": "普通-1b 隐性有害"},
    {"text": "ai 漫剧挺好看的，但 ai 真人是真难看",
     "platform": "xhs", "note": "好看-2a 中性"},
    {"text": "豆包说你好看，可能长在它的审美上",
     "platform": "xhs", "note": "好看-2b 隐性有害"},
    {"text": "保护我方输出",
     "platform": "xhs", "note": "保护-3a 中性"},
    {"text": "不要出生是对她们最好的保护",
     "platform": "zhihu", "note": "保护-3b 隐性有害"},
]
# ==================================================================


@torch.no_grad()
def predict_one(model, tok, text):
    enc = tok(str(text), truncation=True, padding="max_length",
              max_length=C.FT_HP["max_len"], return_tensors="pt")
    enc = {k: v.to(C.DEVICE) for k, v in enc.items()}
    prob = torch.softmax(model(**enc).logits, dim=-1)[0]
    pred = int(prob.argmax().item())
    return pred, float(prob[pred].item())


def main():
    if not CASES:
        print("请先在脚本顶部的 CASES 列表里填入案例文本再运行。")
        return
    tok, base = U.build_model(f"{C.OUT_DIR}/baseline_model")
    _, opt = U.build_model(f"{C.OUT_DIR}/optimized_model")
    base.eval(); opt.eval()

    rows = []
    print(f"{'note':<18}{'基准(置信)':<16}{'优化(置信)':<16}text")
    for c in CASES:
        bp, bc = predict_one(base, tok, c["text"])
        op, oc = predict_one(opt, tok, c["text"])
        rows.append({**c, "baseline_pred": bp, "baseline_conf": round(bc, 3),
                     "optimized_pred": op, "optimized_conf": round(oc, 3)})
        note = c.get("note", "")
        print(f"{note:<18}{bp}({bc:.2f}){'':<8}{op}({oc:.2f}){'':<8}{c['text'][:30]}")

    json.dump(rows, open(f"{C.OUT_DIR}/case_study.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已保存 outputs/case_study.json（0=正常, 1=冒犯）")


if __name__ == "__main__":
    main()
