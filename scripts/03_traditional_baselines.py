"""步骤3：传统机器学习对照 (逻辑回归/SVM/随机森林) 在 COLD 上。
产出: outputs/traditional.json（含 Accuracy/Precision/Recall/F1/AUC）
对应论文: R4 (表II 对照)
"""
import json, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
import config as C
from importlib import import_module
load_cold = import_module("01_train_baseline").load_cold


def scores_for_auc(clf, X):
    """LR/RF 用概率，SVM 用 decision_function（AUC 为排序指标，二者皆可）。"""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def main():
    tr_x, tr_y = load_cold(C.COLD_TRAIN)
    te_x, te_y = load_cold(C.COLD_TEST)
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    Xtr = vec.fit_transform(tr_x); Xte = vec.transform(te_x)

    res = {}
    for name, clf in [("logreg", LogisticRegression(max_iter=1000)),
                      ("svm", LinearSVC()),
                      ("rf", RandomForestClassifier(n_estimators=200, n_jobs=-1))]:
        clf.fit(Xtr, tr_y)
        p = clf.predict(Xte)
        s = scores_for_auc(clf, Xte)
        res[name] = dict(
            accuracy=accuracy_score(te_y, p),
            precision=precision_score(te_y, p, zero_division=0),
            recall=recall_score(te_y, p, zero_division=0),
            f1=f1_score(te_y, p, zero_division=0),
            auc=roc_auc_score(te_y, s))

    # 附 RoBERTa 行（来自步骤2）
    try:
        rb = json.load(open(f"{C.OUT_DIR}/baseline_source_eval.json"))
        res["roberta"] = {k: rb[k] for k in ["accuracy", "precision", "recall", "f1", "auc"]}
    except FileNotFoundError:
        pass

    json.dump(res, open(f"{C.OUT_DIR}/traditional.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("R4（含AUC）=")
    for k, v in res.items():
        print(f"  {k:8s} Acc={v['accuracy']:.4f} P={v['precision']:.4f} "
              f"R={v['recall']:.4f} F1={v['f1']:.4f} AUC={v['auc']:.4f}")


if __name__ == "__main__":
    main()
