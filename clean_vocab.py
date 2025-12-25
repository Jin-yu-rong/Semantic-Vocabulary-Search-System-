import pandas as pd
import re

INPUT_CSV = "raw_vocab.csv"
OUTPUT_CSV = "vocab_clean.csv"
MAX_WORDS = 8000

# 词性映射表（核心）
POS_MAP = {
    "n": "n.",
    "vt": "vt.",
    "vi": "vi.",
    "v": "v.",
    "adj": "a.",
    "adv": "ad.",
    "prep": "prep.",
    "pron": "pron.",
    "conj": "conj.",
    "abbr": "abbr."
}

df = pd.read_csv(INPUT_CSV)
df.columns = ["word", "meaning"]

df["word"] = df["word"].str.lower()
df = df[df["word"].str.fullmatch(r"[a-z]+", na=False)]
df = df.drop_duplicates("word")

def parse_meaning(word, text):
    if not isinstance(text, str):
        return ""

    results = []

    # 按行拆分（原始词典常用）
    lines = re.split(r"[\n\r]+", text)

    for line in lines:
        # 提取词性
        pos_match = re.match(r"([a-z]+)\.\s*(.*)", line.strip())
        if pos_match:
            pos_abbr = pos_match.group(1)
            definition = pos_match.group(2)
            pos_full = POS_MAP.get(pos_abbr)

            if pos_full:
                definition = re.sub(r"[；;、]", ", ", definition)
                definition = re.sub(r"\s+", " ", definition)
                results.append(
                    f"{pos_full} {definition}"
                )

    # 如果没有解析到词性，兜底处理
    if not results:
        clean = re.sub(r"[\"“”（）()]", "", text)
        clean = re.sub(r"\s+", " ", clean)
        return f"{word} means {clean}"

    return "; ".join(results)

df["meaning"] = df.apply(lambda x: parse_meaning(x["word"], x["meaning"]), axis=1)

# 删除太短或失败项
df = df[df["meaning"].str.len() > 1]

df = df.head(MAX_WORDS)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"✅ 完成：{len(df)} 个词，保留词性信息")
