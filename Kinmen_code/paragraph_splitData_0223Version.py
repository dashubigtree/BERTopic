import pandas as pd
import re

# 讀取原始CSV文件，忽略索引列
df = pd.read_csv('/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/kinmen＿2025_0223.csv', index_col=0)

# 創建一個列表來存儲拆分後的數據
new_rows = []

# 遍歷原始數據框的每一行
for index, row in df.iterrows():
    # 獲取文章內容
    content = str(row['content'])
    
    # 使用正則表達式分割段落
    # 這裡我們將通過換行符或者多個空格來分割段落
    paragraphs = re.split(r'\n+|\s{2,}', content)
    
    # 過濾掉空段落
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # 為每個段落創建新的行
    for i, paragraph in enumerate(paragraphs):
        # 複製原始行的所有數據
        new_row = row.copy()
        # 更新content為當前段落
        new_row['content'] = paragraph
        # 添加段落編號（可選）
        new_row['paragraph_number'] = i + 1
        # 添加原始文章ID（可選）
        new_row['original_article_id'] = row['ID']
        
        # 將新行添加到列表中
        new_rows.append(new_row)

# 創建新的數據框
new_df = pd.DataFrame(new_rows)

# 保存為新的CSV文件
output_path = '/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/Kinmen_splitData_20250223_paragraph.csv'
new_df.to_csv(output_path, index=False)

print(f"處理完成！共處理了{len(df)}篇文章，產生了{len(new_rows)}個段落。")