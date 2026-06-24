"""步骤8（新增）：标注一致性 Cohen's / Fleiss' Kappa。
目的：回应审稿人对标注质量的质疑，尤其单独报告隐性类(label 2)的一致性。
需要：原始的多人标注记录(每条样本至少两位标注员的独立标签)。
放在 final_dataset/raw_annotations.csv，列：id, platform, ann1, ann2[, ann3]
(标签取值 0/1/2)。若只有两人用 Cohen，三人及以上用 Fleiss。
产出: outputs/kappa.json
对应论文: R8
"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
import config as C

RAW = os.path.join(C.DATA_DIR, "raw_annotations.csv")


def fleiss_kappa(M):
    """M: (N样本, K类别) 计数矩阵。"""
    N, k = M.shape
    n = M.sum(1)[0]
    p = M.sum(0) / (N * n)
    P = (np.square(M).sum(1) - n) / (n * (n - 1))
    Pbar = P.mean(); Pe = np.square(p).sum()
    return (Pbar - Pe) / (1 - Pe)


def main():
    if not os.path.exists(RAW):
        print("!! 未找到 raw_annotations.csv —— 请导出原始多人标注后再跑")
        print("   需要列: id, platform, ann1, ann2[, ann3]，标签 0/1/2")
        return
    df = pd.read_csv(RAW)
    ann_cols = [c for c in df.columns if c.startswith("ann")]
    out = {}

    def kappa_block(sub, tag):
        if len(ann_cols) == 2:
            a, b = sub[ann_cols[0]], sub[ann_cols[1]]
            out[tag] = {"cohen_kappa": float(cohen_kappa_score(a, b)),
                        "n": int(len(sub))}
        else:
            M = np.zeros((len(sub), 3))
            for i, (_, r) in enumerate(sub.iterrows()):
                for c in ann_cols:
                    M[i, int(r[c])] += 1
            out[tag] = {"fleiss_kappa": float(fleiss_kappa(M)), "n": int(len(sub))}

    kappa_block(df, "overall")
    # 三分类细分：单独看隐性类(label 2)所在样本的一致性
    # 用多数标签近似类别，再分组
    df["maj"] = df[ann_cols].mode(axis=1)[0].astype(int)
    for lab, tag in [(0, "label0_normal"), (1, "label1_explicit"), (2, "label2_implicit")]:
        sub = df[df["maj"] == lab]
        if len(sub) > 1:
            kappa_block(sub, tag)
    # 二分类口径(0 vs {1,2})下的一致性
    db = df.copy()
    for c in ann_cols:
        db[c] = (db[c] >= 1).astype(int)
    kappa_block(db, "binary")

    json.dump(out, open(f"{C.OUT_DIR}/kappa.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("R8 =", out)


if __name__ == "__main__":
    main()
