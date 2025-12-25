import pandas as pd

# 1. 读取原始文件（改成你的实际文件名，我假设是 EnWords.csv 或 test.py 里用的那个）
filename = 'EnWords.csv'  # ← 如果你的文件叫别的（如 103976.csv），改这里

df = pd.read_csv(filename)  # 自动处理编码

print(f"原始数据共 {len(df)} 行")
print("原始列名:", list(df.columns))

# 2. 重命名并添加 pos 列（因为没有独立 pos，我们用 'unknown' 或从 translation 简单占位）
df = df.rename(columns={
    'word': 'word',
    'translation': 'meaning'  # 直接用 translation 作为 meaning
})

# 添加一个 pos 列（临时用 'unknown'，因为词性在 meaning 里）
df['pos'] = 'unknown'  # 你以后可以改进提取

# 3. 清理数据
df = df[['word', 'pos', 'meaning']]  # 固定三列顺序
df = df.dropna(subset=['word', 'meaning'])
df['word'] = df['word'].str.strip().str.lower()
df['meaning'] = df['meaning'].str.strip()

# 去重（同一个 word 只保留第一条）
df = df.drop_duplicates(subset=['word'], keep='first')

df = df.head(103976)

# 4. 保存
df.to_csv('vocabulary.csv', index=False, encoding='utf-8-sig')

print(f"\n处理完成！生成 vocabulary.csv，共 {len(df)} 个单词")
print("前5行预览：")
print(df.head())