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
import plotly.express as px
import matplotlib.pyplot as plt

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
hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True, alpha=0.5)
vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=None, max_features=15000, max_df=0.9, min_df=3)
ctfidf_model = ClassTfidfTransformer(
    seed_words=[
        "权力平衡", "发展", "外交", "人类平等", "国际法", "民族主义", "主权", "领土性", "市场", "战争", "民主", "环境管理", "人权"
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

# 客製化topic labels name
custom_labels = {
    -1: "topic -1 test",
    1: "topic 1 test",
    0: "topic 0 test",
    2: "topic 2 test",
    3: "topic 3 test",
    4: "topic 4 test",
    5: "topic 5 test",
    6: "topic 6 test",
    7: "topic 7 test",
    8: "topic 8 test",
    9: "topic 9 test",
    10: "topic 10 test",
    11: "topic 11 test",
    12: "topic 12 test",
    13: "topic 13 test",
    14: "topic 14 test",
    15: "topic 15 test",
    16: "topic 16 test",
    17: "topic 17 test",
    18: "topic 18 test",
    19: "topic 19 test",
    20:"topic  20 test",
    21:"topic  21 test",
    22:"topic  22 test",
    23:"topic  23 test",
    24:"topic  24 test",
    25:"topic  25 test",
}

# 設置自定義標籤
topic_model.set_topic_labels(custom_labels)
print(f"模型已儲存至: {model_save_path}")

# 視覺化分析
visualization_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/visualization/chi_sp_20240119"
os.makedirs(visualization_path, exist_ok=True)

# 1. 文章主題分布的備用方案（降維後散點圖）
try:
    reduced_embeddings = topic_model._reduce_dimensionality(embeddings)
    # 取得每個主題的前五個關鍵詞
    def get_top5_keywords(topic_id):
        if topic_id in topic_model.get_topics():
            return ", ".join([word for word, _ in topic_model.get_topic(topic_id)[:5]])
        else:
            return ""
    keywords_list = [get_top5_keywords(t) for t in topics]
    viz_df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'topic': [custom_labels.get(t, f"Topic {t}") for t in topics],  # 使用自定義標籤
        'text': texts,
        'keywords': keywords_list
    })
    fig_backup = px.scatter(
        viz_df,
        x='x',
        y='y',
        color='topic',
        hover_data=['text'],
        title='文檔主題分布圖（備用方案）',
        opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.Set3,
        width=1200,
        height=800,
        labels={'topic': '主題分類'}
    )
    fig_backup.update_traces(marker=dict(size=7), 
                             showlegend=True)
    fig_backup.update_layout(
        legend_title_text='主題分類',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            title_font=dict(size=16),
            font=dict(size=14)
        )
    )
    fig_backup.write_html(f"{visualization_path}/documents_visualization_backup.html")
    fig_backup.write_image(f"{visualization_path}/documents_visualization_backup.png", width=1200, height=800, scale=2)
    print("備用主題分布圖已保存")
except Exception as e:
    print(f"備用主題分布圖生成失敗: {e}")

# 2. 主題分佈比例（每個主題群組中的文章數量佔總文章數量的百分比）
try:
    topic_counts = pd.Series(topics).value_counts(normalize=True).sort_index()
    topic_percent = (topic_counts * 100).round(2)
    plt.figure(figsize=(12, 6))
    topic_percent.plot(kind='bar')
    plt.title('主題分佈比例（百分比）')
    plt.xlabel('主題')
    plt.ylabel('百分比 (%)')
    plt.tight_layout()
    plt.savefig(f"{visualization_path}/topic_distribution_percentage.png")
    plt.close()
    print("主題分佈比例圖已保存")
except Exception as e:
    print(f"主題分佈比例圖生成失敗: {e}")

# 3. topic over time
timestamps = df['date'].tolist()

# 生成主題隨時間變化圖
topics_over_time = topic_model.topics_over_time(
    docs=texts,
    timestamps=timestamps,
    global_tuning=True,
    evolution_tuning=True,
    nr_bins=20,
)

# 創建視覺化
fig_topics_over_time = topic_model.visualize_topics_over_time(
    topics_over_time,
    top_n_topics=None,
    width=1200,
    height=600
)

# 保存視覺化結果
fig_topics_over_time.write_html(f"{visualization_path}/topics_over_time.html")
print(f"主題隨時間變化圖已保存至: {visualization_path}/topics_over_time.html")

# 4. 所有主題的關鍵詞分布（條形圖）
try:
    topic_info = topic_model.get_topic_info()
    valid_topics = [topic for topic in topic_info['Topic'].tolist() if topic != -1]
    fig_barchart_all = topic_model.visualize_barchart(
        topics=valid_topics,
        title="所有主題關鍵詞分布",
        n_words=20,
        autoscale=True,
    )
    fig_barchart_all.write_html(f"{visualization_path}/all_topics_keywords.html")
    print("所有主題關鍵詞分布圖已保存")
except Exception as e:
    print(f"所有主題關鍵詞分布圖生成失敗: {e}")

# 5. 所有主題詞與排名分布（ctf-idf score with log scale）
try:
    fig_term_rank_all = topic_model.visualize_term_rank(
        log_scale=True,
        title="所有主題詞語排名分布",
        width=1500,
        height=800
    )
    fig_term_rank_all.write_html(f"{visualization_path}/all_topics_term_rank.html")
    print("所有主題詞語排名分布圖已保存")
except Exception as e:
    print(f"所有主題詞語排名分布圖生成失敗: {e}")

print("所有自訂視覺化已完成並保存至 visualization/chi_sp_20240119 資料夾")

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

# 生成主題比例圖
try:
    # 獲取主題比例
    topic_proportions = topic_model.get_topic_info().sort_values("Count", ascending=True)
    
    # 過濾掉雜訊主題（-1）
    topic_proportions = topic_proportions[topic_proportions['Topic'] != -1]
    
    # 計算主題比例
    total_docs = topic_proportions['Count'].sum()
    topic_proportions['Proportion'] = topic_proportions['Count'] / total_docs
    
    # 為每個主題獲取關鍵詞
    topic_labels = []
    for topic in topic_proportions['Topic']:
        # 獲取前10個關鍵詞
        words = [word for word, _ in topic_model.get_topic(topic)[:10]]
        label = ', '.join(words)
        topic_labels.append(f"Topic {topic}: {label}")
    
    # 創建比例圖
    fig = px.bar(
        topic_proportions,
        x="Proportion",  # 改用比例作為 X 軸
        y="Topic",
        orientation='h',
        title="主題分布比例",
        labels={"Proportion": "主題比例", "Topic": "主題"},  # 更新標籤
        text="Proportion"  # 顯示比例值
    )
    
    # 更新圖表樣式
    fig.update_traces(
        textposition='outside',
        texttemplate='%{text:.1%}',  # 將比例格式化為百分比
        marker_color='lightblue'
    )
    
    fig.update_layout(
        height=800,  # 增加高度以容納所有主題
        yaxis={'ticktext': topic_labels, 'tickvals': topic_proportions['Topic']},
        showlegend=False,
        margin=dict(l=400),  # 增加左邊距以顯示完整的主題標籤
        xaxis_tickformat=',.0%'  # X 軸刻度以百分比格式顯示
    )
    
    # 保存圖表
    fig.write_html(f"{visualization_path}/topic_proportions.html")
    print(f"主題比例圖已保存至: {visualization_path}/topic_proportions.html")
    
except Exception as e:
    print(f"生成主題比例圖時出錯: {str(e)}")