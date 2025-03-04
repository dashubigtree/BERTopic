import pandas as pd

# 讀取 representative_docs.txt 並提取行號
representative_docs_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/results/four_categories_v1/representative_docs.txt"
with open(representative_docs_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 提取行號
line_numbers = []
for line in lines:
    if "(行號:" in line:
        line_number = int(line.split("(行號:")[1].split(")")[0])
        line_numbers.append(line_number)

# 從原始 CSV 文件中提取對應的內容
csv_file_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/Kinmen_splitData_20250223_paragraph_new.csv"
df = pd.read_csv(csv_file_path)

# 打印對應的內容
for line_number in line_numbers:
    content = df.iloc[line_number]['content']
    print(f"行號: {line_number}\n內容: {content}\n{'-' * 50}")