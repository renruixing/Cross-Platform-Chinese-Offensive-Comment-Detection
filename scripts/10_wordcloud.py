"""步骤10：特征词云对比 (图8)。
基准词云：基准模型识别正确的冒犯样本提取关键词。
优化词云：基准漏报、优化后正确检出的隐性冒犯样本提取关键词。
产出: figures/fig_wordcloud_base.png, figures/fig_wordcloud_opt.png
对应论文: 图8
依赖中文字体：请把字体路径填到 FONT。
"""
import os, pandas as pd, jieba
from collections import Counter
from wordcloud import WordCloud
import config as C, utils as U
from importlib import import_module
load_test = import_module("04_eval_crossplatform").load_test

FONT = r"C:\Windows\Fonts\msyh.ttc"


def cut(texts):
    stop = set("的了是我你他她它们这那都也就还在有和与而吗呢啊吧把被给让都很太".split())
    words = []
    for t in texts:
        words += [w for w in jieba.lcut(str(t)) if len(w) > 1 and w not in stop]
    return words


def make(words, path):
    freq = Counter(words)
    wc = WordCloud(font_path=FONT, width=800, height=600, background_color="white",
                   max_words=200).generate_from_frequencies(freq)
    wc.to_file(path); print("saved", path)


def main():
    df = load_test()
    tok, base = U.build_model(f"{C.OUT_DIR}/baseline_model")
    _, opt = U.build_model(f"{C.OUT_DIR}/optimized_model")
    ds = U.TextDS(df.text.tolist(), df.label_binary.astype(int).tolist(), tok, C.FT_HP["max_len"])
    bp = U.predict(base, ds); op = U.predict(opt, ds)
    df["bp"] = bp; df["op"] = op; df["y"] = df.label_binary.astype(int)

    base_correct_off = df[(df.y == 1) & (df.bp == 1)]
    opt_recovered = df[(df.y == 1) & (df.bp == 0) & (df.op == 1)]  # 基准漏报、优化检出
    make(cut(base_correct_off.text), f"{C.FIG_DIR}/fig_wordcloud_base.pdf")
    make(cut(opt_recovered.text), f"{C.FIG_DIR}/fig_wordcloud_opt.pdf")


if __name__ == "__main__":
    main()
