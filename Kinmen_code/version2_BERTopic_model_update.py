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

# 讀取停用詞並添加自定義高頻詞
stopwords_file_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/stop_words.txt"
with open(stopwords_file_path, encoding='utf-8') as f:
    stop_words = set([line.strip() for line in f])

# 添加自定義高頻詞到停用詞
additional_stopwords = {"免责声明","文章描述","免责","删除","网络文章","旨在倡导","不良引导", "文章旨在","倡导社会","低俗","低俗不良","过程图片","图片","不良","来源于"}
stop_words.update(additional_stopwords)

# 加載自定義字典
jieba.load_userdict("/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/jieba.dict.utf8.txt")

def preprocess_text(text):
    tokens = jieba.lcut(text)
    bigram = Phraser(Phrases([tokens], min_count=5, threshold=10))
    tokens_bigram = bigram[tokens]
    tokens_filtered = [token for token in tokens_bigram if token not in stop_words and token.strip()]
    return tokens_filtered

print("開始資料處理...")

# 讀取和處理數據的部分保持不變
csv_file_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/Kinmen_splitData_20250223_paragraph_new.csv"
df = pd.read_csv(csv_file_path)
print(f'總資料筆數: {len(df)}')
print(f'空字串數量: {(df["content"] == "").sum()}')

if 'content' not in df.columns:
    raise ValueError("The CSV file does not contain a 'content' column.")

print("清理文本中...")
df['cleaned_content'] = df['content'].dropna().apply(lambda x: clean_text(x, noise_chars))

empty_rows = df[df["cleaned_content"] == ""]
df = df.drop(empty_rows.index)
print(f'清理後資料筆數: {len(df)}')

print("進行分詞和預處理...")
df["tokens"] = df["cleaned_content"].apply(preprocess_text)

texts = df["tokens"].apply(lambda x: " ".join(x)).tolist()

print("設置 BERTopic 模型...")
# 優化後的模型參數
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
umap_model = UMAP(
    n_neighbors=15,      # 增加鄰居數量以捕獲更多局部結構
    n_components=5,      # 增加維度以保留更多信息
    metric='cosine',
    min_dist=0.1        # 降低最小距離以獲得更緊密的群集
)

hdbscan_model = HDBSCAN(
    min_cluster_size=50,    # 降低最小群集大小
    min_samples=3,          # 降低最小樣本數
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
    cluster_selection_epsilon=0.1  # 降低epsilon值使群集界限更嚴格
)

# 移除這些詞從停用詞中
additional_stopwords = set()  # 清空或移除這行

vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    stop_words=None,
    max_features=15000,
    max_df=0.95,        # 在超過 95% 文檔中出現的詞將被過濾。過濾掉過於常見的詞
    min_df=2          # 至少在 2 個文檔中出現的詞才會被考慮。過濾掉過於罕見的詞
)

# 修改 ClassTfidfTransformer 的設置
ctfidf_model = ClassTfidfTransformer(
    seed_words=[
        "條約", "防禦", "執法",
        "兩岸", "事件", "協議",
        "漁權", "漁業", "經濟"
    ],
    bm25_weighting=True,
    reduce_frequent_words=True    
)

# 建立優化後的模型
topic_model = BERTopic(
    embedding_model=embedding_model,
    verbose=True,
    calculate_probabilities=True,
    nr_topics="auto",
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    top_n_words=20,
    min_topic_size=50,
    ctfidf_model=ctfidf_model,
)

print("開始訓練模型...")
topics, probs = topic_model.fit_transform(texts)

# 儲存模型
model_save_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/model/bertopic_20250223Version_v2"
topic_model.save(model_save_path)
print(f"模型已儲存至: {model_save_path}")

# 視覺化分析
print("生成視覺化結果...")
import plotly.io as pio
pio.templates.default = "presentation"

# 視覺化結果保存
visualization_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/visualization/v2"

# 1. 視覺化主題
fig_topics = topic_model.visualize_topics(width=1200, height=1000)
fig_topics.write_html(f"{visualization_path}/topics_visualization.html")

# 2. 視覺化文檔
fig_docs = topic_model.visualize_documents(texts)
fig_docs.write_html(f"{visualization_path}/documents_visualization.html")

# 3. 視覺化主題層次結構
fig_hierarchy = topic_model.visualize_hierarchy()
fig_hierarchy.write_html(f"{visualization_path}/hierarchy_visualization.html")

# 4. 視覺化主題相似性熱圖
fig_heatmap = topic_model.visualize_heatmap()
fig_heatmap.write_html(f"{visualization_path}/heatmap_visualization.html")

print("視覺化結果已保存至 visualization/v2 資料夾")

# 輸出主題資訊
topic_info = topic_model.get_topic_info()
print("\n主題分布概況：")
print(topic_info)

# 輸出主要主題的關鍵詞
print("\n主要主題的關鍵詞：")
for topic in topic_info.head()['Topic'].tolist():
    if topic != -1:  # 排除雜訊主題
        print(f"\n主題 {topic}:")
        print(topic_model.get_topic(topic))