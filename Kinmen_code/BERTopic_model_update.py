from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from gensim.models.phrases import Phrases, Phraser
from tqdm import tqdm
import jieba
from bertopic.vectorizers import ClassTfidfTransformer


# 定義噪音字元集合
noise_chars = {'zvx','💎', '。', '▲', '△', '🔍', '？', '—', '<', '∶', '\r', '；', '✦', '\u200c', '️', '－', '℃', '‖', '!', '「', '→', '/', '║', '」', '@', '，', '?', "'", '○', '）', '『', '．', '👉', '】', '🌟', '=', '👇', '‰', '【', ';', '#', ')', '：', '\u200d', '❖', '~', ']', '%', '·', '↑', '（', '〕', '☆', '※', '&', '•', '👍', '>', '／', '▌', '–', '↓', '[', ''', ':', '《', '▎', '🤝', '©', '+', '🌊', '\xa0', '\n', '◇', ',', '◎', '…', '(', '〔', '\\', '"', '■', '｜', '─', '\u200b', '-', '●', '"', '▊', '、', '︱', ''', '*', '⭐', '》', '％', '！', '〉', '|', '▼', '👆', '🏡', '°', '\t', '』', '〈', '～', '◆', '.', '⬆', '"'}

def clean_text(text, noise_chars):
    noise_pattern = f"[{''.join(re.escape(char) for char in noise_chars)}]"
    return re.sub(noise_pattern, "", text)

# 讀取停用詞
stopwords_file_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/stop_words.txt"
with open(stopwords_file_path, encoding='utf-8') as f:
    stop_words = set([line.strip() for line in f])

# 加載自定義字典
jieba.load_userdict("/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/jieba.dict.utf8.txt")

def preprocess_text(text):
    # 1. 基礎分詞
    tokens = jieba.lcut(text)
    # 2. 處理 bigram（在停用詞過濾之前）
    bigram = Phraser(Phrases([tokens], min_count=5, threshold=10))
    tokens_bigram = bigram[tokens]
    # 3. 過濾停用詞
    tokens_filtered = [token for token in tokens_bigram if token not in stop_words and token.strip()]
    return tokens_filtered

print("開始資料處理...")

# 讀取 CSV 文件
csv_file_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/Kinmen_splitData_20250223_paragraph_new.csv"
df = pd.read_csv(csv_file_path)
print(f'總資料筆數: {len(df)}')
print(f'空字串數量: {(df["content"] == "").sum()}')

# 確保 content 欄位存在
if 'content' not in df.columns:
    raise ValueError("The CSV file does not contain a 'content' column.")

# 清理文本
print("清理文本中...")
df['cleaned_content'] = df['content'].dropna().apply(lambda x: clean_text(x, noise_chars))

# 處理空值
empty_rows = df[df["cleaned_content"] == ""]
df = df.drop(empty_rows.index)
print(f'清理後資料筆數: {len(df)}')

# 分詞和預處理
print("進行分詞和預處理...")
# 直接使用 apply 方法處理
df["tokens"] = df["cleaned_content"].apply(preprocess_text)

# 轉換為文本格式
texts = df["tokens"].apply(lambda x: " ".join(x)).tolist()

print("設置 BERTopic 模型...")
# BERTopic 模型設置
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
umap_model = UMAP(n_neighbors=10, n_components=2, metric='cosine')  # UMAP 可以使用 cosine
hdbscan_model = HDBSCAN(
    min_cluster_size=50, 
    min_samples=5,
    metric='euclidean',     # 改為 'euclidean' 或 'manhattan'
    cluster_selection_method='eom', 
    prediction_data=True,
    cluster_selection_epsilon=0.2
)

vectorizer = CountVectorizer(
    ngram_range=(1, 1),  # 因為已經用了 bigram，這裡改為 1
    stop_words=None,     # 因為已經過濾過停用詞
    max_features=10000
)

ctfidf_model = ClassTfidfTransformer(
    seed_words=["金門", "海警", "海域", "海巡"],
    bm25_weighting=True, 
    reduce_frequent_words=True
)

# 建立模型
topic_model = BERTopic(
    embedding_model=embedding_model, 
    verbose=True,
    calculate_probabilities=True,
    nr_topics="auto",
    umap_model=umap_model, 
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    top_n_words=15,
    min_topic_size=40,
    ctfidf_model=ctfidf_model
)

print("開始訓練模型...")
# 訓練模型
topics, probs = topic_model.fit_transform(texts)

# 儲存模型
model_save_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/model/bertopic_20250223Version"
topic_model.save(model_save_path)
print(f"模型已儲存至: {model_save_path}")

# 視覺化分析
print("生成視覺化結果...")
import plotly.io as pio
pio.templates.default = "presentation"

# 1. 視覺化主題（修改這部分）
fig_topics = topic_model.visualize_topics(
    topics=None,          # 顯示所有主題
    width=1000,          # 調整圖表寬度
    height=800           # 調整圖表高度
)
fig_topics.write_html("/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/visualization/topics_visualization.html")

# 2. 視覺化文檔
fig_docs = topic_model.visualize_documents(texts)
fig_docs.write_html("/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/visualization/documents_visualization.html")

# 3. 視覺化主題層次結構
fig_hierarchy = topic_model.visualize_hierarchy()
fig_hierarchy.write_html("/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/visualization/hierarchy_visualization.html")

# 4. 視覺化主題相似性熱圖
fig_heatmap = topic_model.visualize_heatmap()
fig_heatmap.write_html("/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/visualization/heatmap_visualization.html")

print("視覺化結果已保存至 visualization 資料夾")

# 5. 獲取主題詳細資訊
topic_info = topic_model.get_topic_info()
print(topic_info)

# 獲取特定主題的關鍵字（例如主題0）
print(topic_model.get_topic(0))
# # 將結果加入原始資料
# df['topic'] = topics
# df['topic_probability'] = np.max(probs, axis=1)

# # 儲存結果
# output_path = '/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/input_data_with_topics_newStopwords.csv'
# df.to_csv(output_path, index=False)
# print(f"分類結果已儲存至: {output_path}")
