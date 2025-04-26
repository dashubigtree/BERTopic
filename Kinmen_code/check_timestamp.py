import pandas as pd

# 讀取數據
df = pd.read_csv("/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/Kinmen_splitData_20250223_paragraph_new.csv")

# 假設日期欄位名稱為 'date'，請根據實際情況修改
df['date'] = pd.to_datetime(df['date'])

# 計算時間跨度
time_span = df['date'].max() - df['date'].min()

print(f"數據時間跨度: {time_span}")
print(f"最早日期: {df['date'].min()}")
print(f"最晚日期: {df['date'].max()}")