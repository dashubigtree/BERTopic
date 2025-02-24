import pandas as pd
import re

def split_into_sentences(text):
    # 定義句子結束的標點符號
    sentence_ends = ['。', '！', '？', '…']
    
    # 初始化結果列表
    sentences = []
    current_sentence = ''
    
    # 遍歷文本
    for char in text:
        current_sentence += char
        if char in sentence_ends:
            # 確保句子不是空的且不只包含空白字符
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            current_sentence = ''
    
    # 處理最後一個可能沒有結束標點的句子
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return sentences

# 讀取原始CSV文件
df = pd.read_csv('/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/kinmen＿2025_0223.csv', index_col=0)

# 創建新的資料框來存儲切割後的結果
new_rows = []

# 遍歷原始數據
for index, row in df.iterrows():
    content = str(row['content'])
    sentences = split_into_sentences(content)
    
    # 為每個句子創建新的行
    for i, sentence in enumerate(sentences, 1):
        new_row = row.copy()
        new_row['content'] = sentence
        new_row['sentence_number'] = i
        new_row['original_article_id'] = row['ID']
        new_rows.append(new_row)

# 創建新的數據框
new_df = pd.DataFrame(new_rows)

# 保存結果
output_path = '/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/Kinmen_splitData_20250223_sentence.csv'
new_df.to_csv(output_path, index=False)

# 輸出基本統計信息
print(f"\n處理完成！統計信息：")
print(f"原始文章數：{len(df)}")
print(f"切割後句子總數：{len(new_df)}")
print(f"平均每篇文章句子數：{len(new_df)/len(df):.2f}")
print(f"最短句子長度：{new_df['content'].str.len().min()}")
print(f"最長句子長度：{new_df['content'].str.len().max()}")
print(f"平均句子長度：{new_df['content'].str.len().mean():.2f}")