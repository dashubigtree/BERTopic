from bertopic import BERTopic
import pandas as pd
import jieba

# 讀取 CSV 文件
csv_file_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/chi_sp_20240119_parag.csv"
df = pd.read_csv(csv_file_path)

# 確保 fulltext 欄位存在
if 'fulltext' not in df.columns:
    raise ValueError("The CSV file does not contain a 'fulltext' column.")

# 處理空值與空字串
df = df[df["fulltext"].notnull() & (df["fulltext"].str.strip() != "")]

# 分詞（簡單版，無停用詞、無bigram）
df["tokens"] = df["fulltext"].apply(lambda x: " ".join(jieba.lcut(str(x))))
texts = df["tokens"].tolist()

# 建立 BERTopic 模型（全部參數自動）
topic_model = BERTopic()

# 訓練模型
topics, probs = topic_model.fit_transform(texts)

# 輸出主題分布
topic_info = topic_model.get_topic_info()
print(topic_info)

# 儲存模型
model_save_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/model/bertopic_chi_sp_20240119_auto"
topic_model.save(model_save_path)
print(f"模型已儲存至: {model_save_path}")