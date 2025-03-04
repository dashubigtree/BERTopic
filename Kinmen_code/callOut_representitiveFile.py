from bertopic import BERTopic
import pandas as pd

# 載入之前訓練的模型
model_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/model/bertopic_four_categories_v1"
topic_model = BERTopic.load(model_path)

# 讀取原始數據
csv_file_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/Kinmen_splitData_20250223_paragraph_new.csv"
df = pd.read_csv(csv_file_path)

# 獲取主題資訊
topic_info = topic_model.get_topic_info()

# 用戶輸入想要列出的代表性文章數量
num_docs = int(input("請輸入每個主題要列出的代表性文章數量："))

# 準備文檔數據
docs = df['content'].tolist()

# 確保 docs 和 topic_model.topics_ 長度一致
if len(docs) != len(topic_model.topics_):
    print(f"警告：docs 長度 ({len(docs)}) 與 topic_model.topics_ 長度 ({len(topic_model.topics_)}) 不一致。")
    print("將根據 topic_model.topics_ 的長度截取 docs。")
    docs = docs[:len(topic_model.topics_)]

# 創建 DataFrame
documents = pd.DataFrame({"Document": docs, "Topic": topic_model.topics_, "ID": range(len(docs))})

# 提取更多代表性文檔
repr_docs, _, _, repr_doc_ids = topic_model._extract_representative_docs(
    topic_model.c_tf_idf_,
    documents,
    topic_model.topic_representations_,
    nr_samples=min(1000, len(docs)),  # 增加樣本數以獲得更多候選文檔
    nr_repr_docs=num_docs,  # 設置為用戶指定的數量
)

# 列出每個主題的代表性文章
for topic in topic_info['Topic'].tolist():
    if topic == -1:  # 跳過雜訊主題
        continue
    
    if topic in repr_docs:
        print(f"\n主題 {topic} 的代表性文章：")
        
        # 獲取該主題的代表性文檔ID
        doc_ids = repr_doc_ids[list(repr_docs.keys()).index(topic)]
        
        for i, (doc, doc_id) in enumerate(zip(repr_docs[topic], doc_ids)):
            # 獲取原始文章
            article_id = doc_id
            original_content = df.iloc[article_id]['content']
            
            print(f"文檔 {i+1} (行號: {article_id}):")
            print(original_content)
            print("-" * 50)
    else:
        print(f"\n主題 {topic} 沒有代表性文章")
