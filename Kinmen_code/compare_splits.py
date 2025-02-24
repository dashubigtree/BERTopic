import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 比較使用統計平均的切割結果
# 讀取兩個版本的切割結果
original_split = pd.read_csv('/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/Kinmen_splitData_20250223_paragraph.csv', index_col=0)
new_split = pd.read_csv('/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/Kinmen_splitData_20250223_paragraph_new.csv')

# 計算段落長度
original_split['paragraph_length'] = original_split['content'].str.len()
new_split['paragraph_length'] = new_split['content'].str.len()

# 計算統計資訊
def get_stats(df):
    return {
        '平均段落長度': df['paragraph_length'].mean(),
        '中位數段落長度': df['paragraph_length'].median(),
        '最短段落長度': df['paragraph_length'].min(),
        '最長段落長度': df['paragraph_length'].max(),
        '標準差': df['paragraph_length'].std(),
        '每篇文章平均段落數': df.groupby('original_article_id').size().mean(),
        '總段落數': len(df)
    }

# 輸出比較結果
print("\n原始切割方法統計：")
original_stats = get_stats(original_split)
for key, value in original_stats.items():
    print(f"{key}: {value:.2f}")

print("\n新切割方法統計：")
new_stats = get_stats(new_split)
for key, value in new_stats.items():
    print(f"{key}: {value:.2f}")

# 繪製對比圖
plt.figure(figsize=(15, 6))

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 創建子圖
plt.subplot(1, 2, 1)
sns.histplot(data=original_split, x='paragraph_length', bins=50)
plt.title('原始切割方法段落長度分布')
plt.xlabel('段落字數')
plt.ylabel('頻率')

plt.subplot(1, 2, 2)
sns.histplot(data=new_split, x='paragraph_length', bins=50)
plt.title('新切割方法段落長度分布')
plt.xlabel('段落字數')
plt.ylabel('頻率')

plt.tight_layout()
plt.savefig('/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/paragraph_length_comparison.png')
plt.close()

# 計算兩種方法的段落長度差異
print("\n兩種切割方法的差異：")
diff_stats = {
    '平均長度差異': new_stats['平均段落長度'] - original_stats['平均段落長度'],
    '中位數差異': new_stats['中位數段落長度'] - original_stats['中位數段落長度'],
    '標準差差異': new_stats['標準差'] - original_stats['標準差'],
    '段落數量差異': new_stats['總段落數'] - original_stats['總段落數']
}

for key, value in diff_stats.items():
    print(f"{key}: {value:.2f}")