import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 先讀取以雙空格為切割判斷的切割結果，接著在使用平均長度為切割判斷的切割結果
# 讀取切割後的資料
split_df = pd.read_csv('/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/Kinmen_splitData_20250223_paragraph.csv', index_col=0)

# 計算每個段落的字數
split_df['paragraph_length'] = split_df['content'].str.len()

# 計算統計資訊
stats = {
    '平均段落長度': split_df['paragraph_length'].mean(),
    '中位數段落長度': split_df['paragraph_length'].median(),
    '最短段落長度': split_df['paragraph_length'].min(),
    '最長段落長度': split_df['paragraph_length'].max(),
    '標準差': split_df['paragraph_length'].std(),
    '每篇文章平均段落數': split_df.groupby('original_article_id').size().mean()
}

# 輸出統計資訊
print("\n段落統計資訊：")
for key, value in stats.items():
    print(f"{key}: {value:.2f}")

# 繪製段落長度分布圖
plt.figure(figsize=(10, 6))
split_df['paragraph_length'].hist(bins=50)
plt.title('段落長度分布')
plt.xlabel('段落字數')
plt.ylabel('頻率')
plt.savefig('/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/paragraph_length_distribution.png')
plt.close()

# 根據統計結果重新切割原始資料
def split_by_length(text, target_length):
    # 使用標點符號作為可能的切割點
    delimiters = ['。', '！', '？', '；']
    current_length = 0
    current_paragraph = ''
    paragraphs = []
    
    # 先按句子分割
    sentences = []
    temp = ''
    for char in text:
        temp += char
        if char in delimiters:
            sentences.append(temp)
            temp = ''
    if temp:
        sentences.append(temp)
    
    # 根據目標長度組合句子
    for sentence in sentences:
        if len(current_paragraph) + len(sentence) <= target_length:
            current_paragraph += sentence
        else:
            if current_paragraph:
                paragraphs.append(current_paragraph)
            current_paragraph = sentence
    
    if current_paragraph:
        paragraphs.append(current_paragraph)
    
    return paragraphs

# 讀取原始資料
original_df = pd.read_csv('/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/kinmen＿2025_0223.csv', index_col=0)

# 使用平均長度作為目標長度
target_length = int(stats['平均段落長度'])

# 創建新的資料框來存儲重新切割的結果
new_rows = []

for index, row in original_df.iterrows():
    content = str(row['content'])
    paragraphs = split_by_length(content, target_length)
    
    for i, paragraph in enumerate(paragraphs, 1):
        new_row = row.copy()
        new_row['content'] = paragraph.strip()
        new_row['paragraph_number'] = i
        new_row['original_article_id'] = row['ID']
        new_rows.append(new_row)

# 創建新的數據框
new_df = pd.DataFrame(new_rows)

# 保存結果
output_path = '/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/Kinmen_splitData_20250223_paragraph_new.csv'
new_df.to_csv(output_path, index=False)

print(f"\n處理完成！新的切割結果已保存至：{output_path}")