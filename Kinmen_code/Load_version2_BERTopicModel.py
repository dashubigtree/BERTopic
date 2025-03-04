from bertopic import BERTopic
import pandas as pd
import os

# 定義模型路徑和輸出路徑
model_load_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/model/bertopic_20250223Version_v2"
output_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/results/v2"

# 確保輸出路徑存在
os.makedirs(output_path, exist_ok=True)

print("載入 BERTopic 模型...")
# 載入模型
topic_model = BERTopic.load(model_load_path)
print("模型載入完成！")

print("獲取主題資訊...")
# 獲取主題資訊
topic_info = topic_model.get_topic_info()

print("儲存主題資訊到 CSV 檔案...")
# 儲存主題資訊到指定路徑
topic_info_file = os.path.join(output_path, "topic_info.csv")
topic_info.to_csv(topic_info_file, index=False, encoding="utf-8-sig")
print(f"主題資訊已儲存至: {topic_info_file}")

print("程式執行完成！")