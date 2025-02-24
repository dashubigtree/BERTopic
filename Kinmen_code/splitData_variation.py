import pandas as pd

# 讀取原始CSV文件，忽略索引列
df = pd.read_csv('/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/kinmen＿2025_0223.csv', index_col=0)
new_df = pd.read_csv('/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/Kinmen_splitData_20250223_paragraph.csv', index_col=0)
# 隨機抽取幾篇原始文章和切割後的結果進行比對
sample_size = 5
sample_ids = df['ID'].sample(n=sample_size)

for id in sample_ids:
    print("="*50)
    print(f"原始文章 ID: {id}")
    print("原始內容：")
    print(df[df['ID'] == id]['content'].values[0])
    print("\n切割後的段落：")
    print("---")
    for idx, row in new_df[new_df['original_article_id'] == id].iterrows():
        print(f"段落 {row['paragraph_number']}:")
        print(row['content'])
        print("---")