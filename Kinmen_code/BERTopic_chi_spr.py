from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from gensim.models.phrases import Phrases, Phraser
import jieba
from bertopic.vectorizers import ClassTfidfTransformer
import numpy as np
import plotly.io as pio
import os

# 定義噪音字元集合
noise_chars = {'zvx','💎', '。', '▲', '△', '🔍', '？', '—', '<', '∶', '\r', '；', '✦', '\u200c', '️', '－', '℃', '‖', '!', '「', '→', '/', '║', '」', '@', '，', '?', "'", '○', '）', '『', '．', '👉', '】', '🌟', '=', '👇', '‰', '【', ';', '#', ')', '：', '\u200d', '❖', '~', ']', '%', '·', '↑', '（', '〕', '☆', '※', '&', '•', '👍', '>', '／', '▌', '–', '↓', '[', ''', ':', '《', '▎', '🤝', '©', '+', '🌊', '\xa0', '\n', '◇', ',', '◎', '…', '(', '〔', '\\', '"', '■', '｜', '─', '\u200b', '-', '●', '"', '▊', '、', '︱', ''', '*', '⭐', '》', '％', '！', '〉', '|', '▼', '👆', '🏡', '°', '\t', '』', '〈', '～', '◆', '.', '⬆', '"'}

def clean_text(text, noise_chars):
    noise_pattern = f"[{''.join(re.escape(char) for char in noise_chars)}]"
    return re.sub(noise_pattern, "", str(text))

# 讀取停用詞
stopwords_file_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/stop_words.txt"
with open(stopwords_file_path, encoding='utf-8') as f:
    stop_words = set([line.strip() for line in f])

# 加載自定義字典
jieba.load_userdict("/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/jieba.dict.utf8.txt")

def preprocess_text(text):
    tokens = jieba.lcut(text)
    bigram = Phraser(Phrases([tokens], min_count=5, threshold=10))
    tokens_bigram = bigram[tokens]
    tokens_filtered = [token for token in tokens_bigram if token not in stop_words and token.strip()]
    return tokens_filtered

print("開始資料處理...")

# 讀取 CSV 文件
csv_file_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/chi_sp_20240119_parag.csv"
df = pd.read_csv(csv_file_path)
print(f'總資料筆數: {len(df)}')
print(f'空字串數量: {(df["fulltext"] == "").sum()}')

# 確保 fulltext 欄位存在
if 'fulltext' not in df.columns:
    raise ValueError("The CSV file does not contain a 'fulltext' column.")

# 清理文本
print("清理文本中...")
df['cleaned_content'] = df['fulltext'].dropna().apply(lambda x: clean_text(x, noise_chars))

# 處理空值
empty_rows = df[df["cleaned_content"].str.strip() == ""]
df = df.drop(empty_rows.index)
print(f'清理後資料筆數: {len(df)}')

# 分詞和預處理
print("進行分詞和預處理...")
df["tokens"] = df["cleaned_content"].apply(preprocess_text)

# 過濫短文本
min_tokens = 3
df = df[df["tokens"].apply(len) >= min_tokens]
print(f'過濫短文本後資料筆數: {len(df)}')

# 轉換為文本格式
texts = df["tokens"].apply(lambda x: " ".join(x)).tolist()

print("設置 BERTopic 模型...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
umap_model = UMAP(n_neighbors=15, n_components=2, metric='cosine', min_dist=0.05, random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True, alpha=0.5)
vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=None, max_features=15000, max_df=0.9, min_df=3)
ctfidf_model = ClassTfidfTransformer(
    seed_words=[
        "初级制度", "关键词",
        "权力平衡", "联盟", "平衡",
        "发展", "社会进步", "人类进步", "物质进步",
        "外交", "双边会议", "外交", "外交的", "使者", "多边", "外交语言", "大会",
        "人类平等", "种族隔离", "殖民地", "性别平等", "种族的", "种族主义", "种族主义者", "性别", "前领土", "附属地", "海外领地", "殖民地",
        "国际法", "宪章", "公约", "法院", "法律", "具有法律约束力的文件", "议定书", "条约", "仲裁",
        "民族主义", "民族主义者", "自决", "民族自决", "民族主义", "人民主权",
        "主权", "独立", "不干涉", "不干预", "主权", "国家责任",
        "领土性", "边界", "疆界", "领土的", "领土",
        "市场", "经济的", "经济", "贸易", "经济一体化", "保护主义", "贸易壁垒", "关税", "市场",
        "战争", "战争", "使用武力", "进攻", "防御", "侵略", "防卫", "自卫",
        "民主", "民主", "民主的", "议会", "议会的", "少数派", "投票", "选举", "集会自由", "少数派权利",
        "环境管理", "气候变化", "生态平衡", "环境的", "保护环境", "环境保护", "排放", "全球变暖", "森林砍伐", "海平面", "温室效应",
        "人权", "人权", "酷刑", "言论自由", "奴役", "奴隶制", "监禁", "种族灭绝", "权利"
    ],
    bm25_weighting=True,
    reduce_frequent_words=True
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    verbose=True,
    calculate_probabilities=True,
    nr_topics=25,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    top_n_words=20,
    min_topic_size=20,
    ctfidf_model=ctfidf_model,
)

print("生成文檔嵌入向量...")
embeddings = embedding_model.encode(texts, show_progress_bar=True)
print(f"嵌入向量形狀: {embeddings.shape}")

print("開始訓練模型...")
topics, _ = topic_model.fit_transform(texts, embeddings)

# 儲存模型
model_save_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/model/bertopic_chi_sp_20240119"
topic_model.save(model_save_path)
print(f"模型已儲存至: {model_save_path}")

# 視覺化分析
visualization_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/visualization/chi_sp_20240119"
os.makedirs(visualization_path, exist_ok=True)

fig_topics = topic_model.visualize_topics(width=1200, height=1000)
fig_topics.write_html(f"{visualization_path}/topics_visualization.html")

fig_docs = topic_model.visualize_documents(texts)
fig_docs.write_html(f"{visualization_path}/documents_visualization.html")

fig_hierarchy = topic_model.visualize_hierarchy()
fig_hierarchy.write_html(f"{visualization_path}/hierarchy_visualization.html")

fig_heatmap = topic_model.visualize_heatmap()
fig_heatmap.write_html(f"{visualization_path}/heatmap_visualization.html")

print("視覺化結果已保存至 visualization/chi_sp_20240119 資料夾")

topic_info = topic_model.get_topic_info()
print("\n主題分布概況：")
print(topic_info)

# 將主題標籤加入原始資料集
df_with_topics = df.copy()
df_with_topics['topic'] = topics  # 新增一欄標示每篇文章的主題編號

# 新增各個主題的關鍵字
# 新增一欄標示該主題的關鍵字
def get_topic_keywords(topic_id, n_words=10):
    if topic_id in topic_model.get_topics():
        return ", ".join([word for word, _ in topic_model.get_topic(topic_id)[:n_words]])
    else:
        return ""

df_with_topics['topic_keywords'] = df_with_topics['topic'].apply(get_topic_keywords)

# 儲存含主題標籤的資料集
output_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/data/chi_sp_20240119_parag_with_topics.csv"
df_with_topics.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"已將主題標籤加入資料集並儲存至: {output_path}")