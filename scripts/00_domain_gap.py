"""步骤0：量化平台间域差距，回答"如何定义不同平台数据集差异"。
三个指标：
  (1) 词表 Jaccard 重叠（越低=用语差异越大）
  (2) Proxy-A-distance（Ben-David et al.）：训域分类器区分平台A/B，PAD=2(1-2err)，越大域差越大
  (3) 冒犯占比（标签分布漂移）
若提供 COLD 文本（config.COLD_TRAIN），会额外计算"各平台 vs COLD"的距离——
这才是判断"对 COLD 模型是否真跨域"的直接证据。
产出: outputs/domain_gap.json
对应论文: 表(域差距) + IV.A 跨域有效性分析
"""
import os, json, re, itertools, numpy as np, pandas as pd, jieba
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import config as C
jieba.setLogLevel(20)

CAP = 3000
def tok(texts):
    """
    文本分词函数：使用jieba进行中文分词
    参数：
        texts: 文本列表
    返回：
        分词后的文本列表，词语用空格分隔
    """
    return [" ".join(jieba.lcut(re.sub(r"\s+", "", str(t)))) for t in texts]

def load_pool(p):
    d = pd.read_csv(os.path.join(C.POOL_DIR, f"{p}.csv"), encoding="utf-8-sig",
                    dtype=str, keep_default_na=False)
    return d["text"].tolist()

def load_cold_text():
    if not os.path.exists(C.COLD_TRAIN): return None
    df = pd.read_csv(C.COLD_TRAIN)
    col = "TEXT" if "TEXT" in df.columns else ("text" if "text" in df.columns else df.columns[0])
    return df[col].astype(str).tolist()

def proxy_a_distance(Xa, Xb):
    """
    计算两个域之间的Proxy-A-distance (PAD)
    原理：训练一个二分类器区分域A和域B的数据，分类误差为err
          PAD = 2 * (1 - 2 * err)
          值域[0,2]，越大表示两个域差异越大
    参数：
        Xa: 域A的特征矩阵（稀疏矩阵）
        Xb: 域B的特征矩阵（稀疏矩阵）
    返回：
        PAD值
    """
    n = min(Xa.shape[0], Xb.shape[0])
    XX = sp.vstack([Xa[:n], Xb[:n]]); yy = np.r_[np.zeros(n), np.ones(n)]
    err = 1 - cross_val_score(LinearSVC(C=1.0), XX, yy, cv=3).mean()
    return float(2 * (1 - 2 * err))

def main():
    plats = C.PLATFORMS
    tokd = {p: tok(load_pool(p)[:CAP]) for p in plats}
    cold = load_cold_text()
    if cold: tokd["COLD"] = tok(cold[:CAP])
    keys = list(tokd.keys())

    # (1) ---- 指标1：词表Jaccard重叠率 ----
    vocab = {}
    for p in keys:
        cv = CountVectorizer(min_df=2); cv.fit(tokd[p]); vocab[p] = set(cv.vocabulary_)  # min_df=2：词至少在2个文档中出现
    # 计算所有数据对之间的Jaccard相似度
    jac = {f"{a}-{b}": len(vocab[a] & vocab[b]) / len(vocab[a] | vocab[b])
           for a, b in itertools.combinations(keys, 2)}

    # (2) ---- 指标2：Proxy-A-distance ----
    allv = TfidfVectorizer(min_df=3, max_features=20000)
    allv.fit([t for p in keys for t in tokd[p]])
    X = {p: allv.transform(tokd[p]) for p in keys}
    pad = {f"{a}-{b}": proxy_a_distance(X[a], X[b]) for a, b in itertools.combinations(keys, 2)}

    # (3) ---- 指标3：标签分布漂移（冒犯样本占比）- ---
    ratio = {}
    for p in plats:
        d = pd.read_csv(os.path.join(C.POOL_DIR, f"{p}.csv"), encoding="utf-8-sig",
                        dtype=str, keep_default_na=False)
        ratio[p] = float((d["pseudo_label"].astype(int) == 1).mean())

    out = {"jaccard_vocab_overlap": jac, "proxy_a_distance": pad,
           "offensive_ratio": ratio, "has_cold": cold is not None}
    json.dump(out, open(f"{C.OUT_DIR}/domain_gap.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    # 如果包含COLD数据，特别打印各平台与COLD的域差距
    if cold:
        print("\n各平台 vs COLD 的 Proxy-A-distance（判断是否真跨域）：")
        for p in plats:
            print(f"  {p}: PAD={pad.get(f'{p}-COLD', pad.get(f'COLD-{p}')):.2f}, "
                  f"Jaccard={jac.get(f'{p}-COLD', jac.get(f'COLD-{p}')):.3f}")


if __name__ == "__main__":
    main()
