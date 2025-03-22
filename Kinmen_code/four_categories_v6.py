# 在文件開頭的導入部分添加以下內容
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
import numpy as np
import plotly.express as px  # 添加這行
import plotly.io as pio      # 這行已經存在，不需要重複添加
import traceback
import random
import os
# 定義噪音字元集合
noise_chars = {'zvx','\U0001F48E', '。', '▲', '△', '\U0001F50D', '？', '—', '<', '∶', '\\r', '；', '✦', '\\u200c', '️', '－', '℃', '‖', '!', '「', '→', '/', '║', '」', '@', '，', '?', "\"", '○', '）', '『', '．', '\U0001F449', '】', '\U0001F31F', '=', '\U0001F447', '‰', '【', ';', '#', ')', '：', '\\u200d','\u200d', '❖', '~', ']', '%', '·', '↑', '（', '〕', '☆', '※', '&', '•', '\U0001F44D', '>', '／', '▌', '–', '↓', '[', ''', ':', '《', '▎', '\U0001F91D', '©', '+', '\U0001F30A', '\\xa0', '\\n', '◇', ',', '◎', '…', '(', '〔', '\\\\', '\"', '■', '｜', '─', '\\u200b', '-', '●', '\"', '▊', '、', '︱', ''', '*', '⭐', '》', '％', '！', '〉', '|', '▼', '\U0001F446', '\U0001F3E1', '°', '\\t', '』', '〈', '～', '◆', '.', '⬆', '\"'}

def clean_text(text, noise_chars):
    noise_pattern = f"[{''.join(re.escape(char) for char in noise_chars)}]"
    return re.sub(noise_pattern, "", text)

# 讀取停用詞並添加自定義高頻詞
stopwords_file_path = "./data/stop_words.txt"
with open(stopwords_file_path, encoding='utf-8') as f:
    stop_words = set([line.strip() for line in f])

# 添加自定義高頻詞到停用詞
additional_stopwords = {
    "免责声明","文章描述","免责","删除","网络文章","旨在倡导","不良引导", "文章旨在","倡导社会","低俗","低俗不良","过程图片",
    "图片","不良","来源于","口感","复盆子","阅读原文","阅读","原文","版权","文章旨在","文章","CCTV4","cctv4","上方cctv4","cctv4 关注","点击上方","上方",
    "朋友圈","一键","一键分享","朋友圈","点击","下图","分享","右侧","下方","左侧","白酒","一瓶","小编","香肠","购买","草莓","央视"
    }
stop_words.update(additional_stopwords)

# 加載自定義字典
jieba.load_userdict("./data/jieba.dict.utf8.txt")

def preprocess_text(text):
    tokens = jieba.lcut(text)
    bigram = Phraser(Phrases([tokens], min_count=5, threshold=10))
    tokens_bigram = bigram[tokens]
    tokens_filtered = [token for token in tokens_bigram if token not in stop_words and token.strip()]
    return tokens_filtered

print("開始資料處理...")

# 讀取和處理數據的部分保持不變
csv_file_path = "./data/Kinmen_splitData_20250223_paragraph_new.csv"
df = pd.read_csv(csv_file_path)
print(f'總資料筆數: {len(df)}')
print(f'空字串數量: {(df["content"] == "").sum()}')

if 'content' not in df.columns:
    raise ValueError("The CSV file does not contain a 'content' column.")

print("清理文本中...")
# 1. 先清理文本
df['cleaned_content'] = df['content'].dropna().apply(lambda x: clean_text(x, noise_chars))

# 2. 移除清理後為空的行
empty_rows = df[df["cleaned_content"].str.strip() == ""]
df = df.drop(empty_rows.index)
print(f'清理後資料筆數: {len(df)}')

print("進行分詞和預處理...")
# 3. 分詞和預處理
df["tokens"] = df["cleaned_content"].apply(preprocess_text)

# 4. 過濫token數量過少的文本
min_tokens = 3  # 設定最小token數量
df = df[df["tokens"].apply(len) >= min_tokens]
print(f'過濫短文本後資料筆數: {len(df)}')

# 5. 生成最終文本列表
texts = df["tokens"].apply(lambda x: " ".join(x)).tolist()

print("設置 BERTopic 模型...")
# 優化後的模型參數
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 調整 UMAP 參數
umap_model = UMAP(
    n_neighbors=15,      # 增加鄰居數量以捕獲更多局部結構
    n_components=2,      # 降為2維以便更好地可視化
    metric='cosine',
    min_dist=0.05,       # 適中的最小距離
    random_state=42      # 固定隨機種子以獲得可重複的結果
)

# 調整 HDBSCAN 參數以產生約20多個主題
hdbscan_model = HDBSCAN(
    min_cluster_size=35,    # 調整以獲得約20多個主題
    min_samples=5,          # 增加樣本數以獲得更穩定的群集
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
    alpha=0.5
    )

vectorizer = CountVectorizer(
    ngram_range=(1, 1),
    stop_words=None,
    max_features=15000,
    max_df=0.9,
    min_df=3
    )

# 修改 ClassTfidfTransformer 的設置
ctfidf_model = ClassTfidfTransformer(
    seed_words=[
        "条约", "防御", "执法", "金门",
        "两岸", "事件", "协议", "海域",
        "鱼权", "渔业", "经济", "台湾",
        "中国", "海巡", "大陆", "国民党",
        #"东引岛","乌丘","台海","海警"
    ],
    bm25_weighting=True,
    reduce_frequent_words=True    
)

# 建立優化後的模型
topic_model = BERTopic(
    embedding_model=embedding_model,  # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    verbose=True,
    calculate_probabilities=True,
    nr_topics=25,          # 明確指定約25個主題
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    top_n_words=20,
    min_topic_size=20,     
    ctfidf_model=ctfidf_model,
)


# 先生成嵌入向量
# 在生成嵌入向量後添加以下診斷代碼
print("生成文檔嵌入向量...")
embeddings = embedding_model.encode(texts, show_progress_bar=True)
print(f"嵌入向量形狀: {embeddings.shape}")


# 訓練模型時傳入嵌入向量
print("開始訓練模型...")
topics, _ = topic_model.fit_transform(texts, embeddings)

# 儲存模型
model_save_path = "./model/bertopic_four_categories_v6"  # 更新路徑
topic_model.save(model_save_path)
# 1. 將 topic_info 的 print 資訊儲存到指定路徑
# 定義主題標籤映射（移到視覺化之前）
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

# 在視覺化之前添加以下診斷代碼
print(f"主題數量: {len(topics)}")
print(f"嵌入向量數量: {len(embeddings)}")
print(f"文本數量: {len(texts)}")
print(f"唯一主題: {set(topics)}")
print(f"文本數量: {len(texts)}")




# 視覺化分析
print("生成視覺化結果...")
import plotly.io as pio
import os
pio.renderers.default = "browser"  # 尝试使用浏览器渲染器
pio.templates.default = "plotly"   #默認模板

# 確保視覺化目錄存在
visualization_path = "./visualization/four_categories_v6"  # 更新路徑
os.makedirs(visualization_path, exist_ok=True)

# 獲取主題資訊
topic_info = topic_model.get_topic_info()

# 使用 BERTopic 原生的視覺化功能
# 1. Intertopic Distance Map (主題間距離圖)
fig_intertopic = topic_model.visualize_topics(custom_labels=False)
fig_intertopic.write_html(f"{visualization_path}/intertopic_distance_map.html")

# 2. 生成所有主題的詞語排名圖和關鍵詞條形圖
try:
    # 獲取非雜訊主題列表
    valid_topics = [topic for topic in topic_info['Topic'].tolist() if topic != -1]
    
    # 生成所有主題的詞語排名圖 (Term Rank)
    print("生成所有主題的詞語排名圖...")
    fig_term_rank_all = topic_model.visualize_term_rank(
        log_scale=True,
        custom_labels=custom_labels,
        title="所有主題詞語排名分布",
        width=1500,
        height=800
    )
    fig_term_rank_all.write_html(f"{visualization_path}/all_topics_term_rank.html")
    
    # 生成所有主題的關鍵詞條形圖 (Keyword Barchart)
    print("生成所有主題的關鍵詞條形圖...")
    print(f"length of valid_topics: {len(valid_topics)}")
    print(f"valid_topics: {valid_topics}")
    fig_barchart_all = topic_model.visualize_barchart(
        topics=valid_topics,  # 顯示所有非雜訊主題
        title="所有主題關鍵詞分布",
        custom_labels=custom_labels,
        n_words=20,          # 每個主題顯示的關鍵詞數量
        autoscale=True,
        )
    fig_barchart_all.write_html(f"{visualization_path}/all_topics_keywords.html")
    
    print("整合主題視覺化完成")
except Exception as e:
    print(f"生成整合主題視覺化時出錯: {str(e)}")
    print(f"詳細錯誤信息:\n{traceback.format_exc()}")



print("\n===== 生成文檔視覺化 =====")
embeddings_cleaned = embedding_model.encode(df['cleaned_content'].tolist(), show_progress_bar=True)
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings_cleaned)

# 方法1：使用 BERTopic 的 visualize_documents
try:    
    # 添加更多診斷信息
    print("檢查視覺化所需數據：")
    print(f"texts 長度: {len(texts)}")
    print(f"reduced_embeddings 形狀: {reduced_embeddings.shape}")
    print(f"topics 長度: {len(topics)}")

    # 確保所有數據維度一致
    if not (len(texts) == reduced_embeddings.shape[0] == len(topics)):
        raise ValueError(f"數據維度不匹配: texts={len(texts)}, embeddings={reduced_embeddings.shape[0]}, topics={len(topics)}")

    # 使用降維後的嵌入向量進行視覺化
    fig_docs_original = topic_model.visualize_documents(
        docs=texts,
        reduced_embeddings=reduced_embeddings,
        topics=topics,  # 明確傳入主題標籤
        width=1200,
        height=800,
        title="文檔主題分布圖 (BERTopic)",
        custom_labels=custom_labels
    )
    
    # 檢查生成的圖形對象
    print(f"圖形對象類型: {type(fig_docs_original)}")
    
    # 保存前檢查目錄
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)
    
    # 使用絕對路徑保存
    save_path = os.path.abspath(f"{visualization_path}/documents_visualization_original.html")
    fig_docs_original.write_html(save_path)
    print(f"視覺化文件已保存至: {save_path}")
    

except Exception as e:    
    # 如果第一種方法失敗，記錄更多診斷信息
    print("\n=== 診斷信息 ===")
    print(f"1. texts 類型: {type(texts)}")
    print(f"2. embeddings 類型: {type(reduced_embeddings)}")
    if isinstance(reduced_embeddings, np.ndarray):
        print(f"   embeddings 數據類型: {reduced_embeddings.dtype}")
    print(f"3. topics 類型: {type(topics)}")
    print("=" * 50)

# 方法2：無論上面是否成功，都執行備用方案
try:
    
    # 獲取降維後的嵌入向量
    reduced_embeddings = topic_model._reduce_dimensionality(embeddings)
    
    # 創建數據框，使用 custom_labels 替換原始主題編號
    viz_df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'topic': [custom_labels.get(t, f"Topic {t}") for t in topics],  # 使用自定義標籤
        'text': texts
    })
    
    # 創建散點圖
    fig_backup = px.scatter(
        viz_df,
        x='x',
        y='y',
        color='topic',
        hover_data=['text'],
        title='文檔主題分布圖（備用方案）',
        opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.Set3,  # 使用更多顏色
        width=1200,
        height=800,
        labels={'topic': '主題'}  # 更新圖例標籤
    )
    
    # 更新標記大小和圖例設置
    fig_backup.update_traces(
        marker=dict(size=7),
        showlegend=True
    )
    
    # 更新布局
    fig_backup.update_layout(
        legend_title_text='主題分類',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    
    # 保存 HTML 版本
    fig_backup.write_html(f"{visualization_path}/documents_visualization_backup.html")
    
    # 保存 PNG 版本
    fig_backup.write_image(f"{visualization_path}/documents_visualization_backup.png", 
                          width=1200, height=800, scale=2)
    
except Exception as e:
    pass

print(f"所有視覺化結果已保存到 {visualization_path}")

# 建立包含主題的數據框
df_with_topics = df.copy()
df_with_topics['topic'] = topics  # 添加主題標籤

# 其他有用的視覺化
print("\n生成視覺化圖表...")

# 1. 視覺化主題相似性熱圖
try:
    fig_heatmap = topic_model.visualize_heatmap(
        custom_labels=custom_labels,
        width=1200,
        height=800
    )
    fig_heatmap.write_html(f"{visualization_path}/heatmap_visualization.html")
    print("熱圖生成成功")
except Exception as e:
    print(f"生成熱圖時出錯: {str(e)}")

# 在其他視覺化之後，添加 hierarchical clustering 視覺化
print("生成主題層次聚類圖...")
try:
    # 生成層次聚類圖
    fig_hierarchical = topic_model.visualize_hierarchy(
        width=1800,
        height=1200,
        title="主題層次結構圖"
    )
    
    # 保存視覺化結果
    fig_hierarchical.write_html(f"{visualization_path}/hierarchical_clustering.html")
    print(f"層次聚類圖已保存至: {visualization_path}/hierarchical_clustering.html")
    
except Exception as e:
    print(f"生成層次聚類圖時出錯: {str(e)}")

# 在其他視覺化之後，添加 topics over time 的視覺化
print("生成主題隨時間變化圖...")

# 假設您的 DataFrame 中有一個時間列，名為 'date'
# 如果沒有，需要先從現有數據中提取或創建時間資訊
if 'date' not in df.columns:
    print("正在從數據中提取日期信息...")
    # 如果需要從其他列提取日期，請相應修改
    df['date'] = pd.to_datetime(df['date_column'])  # 替換 'date_column' 為實際的列名

# 獲取時間序列數據
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


# 2. start visualize_documents
print("\n===== 生成文檔視覺化 =====")
embeddings_cleaned = embedding_model.encode(df['cleaned_content'].tolist(), show_progress_bar=True)
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings_cleaned)


print(f"所有視覺化結果已保存到 {visualization_path}")

# 建立包含主題的數據框
df_with_topics = df.copy()
df_with_topics['topic'] = topics  # 添加主題標籤

print("\n生成主題分布統計...")
# 計算主題分布（不含雜訊主題）
df_filtered = df_with_topics[df_with_topics['topic'] != -1]
topic_counts = df_filtered['topic'].value_counts()
total_docs = len(df_filtered)
topic_percentages = (topic_counts / total_docs * 100).round(2)

# 創建數據框，包含主題編號、百分比和關鍵字
plot_df = pd.DataFrame({
    'Topic': topic_counts.index,
    'Percentage': topic_percentages.values
})

# 添加關鍵字信息
plot_df['Keywords'] = plot_df['Topic'].apply(lambda x: 
    ' | '.join([f"{word}({score:.3f})" for word, score in topic_model.get_topic(x)[:5]])
)

# 排序
plot_df = plot_df.sort_values('Topic')

# 創建柱狀圖
fig = px.bar(
    plot_df,
    x='Topic',
    y='Percentage',
    title='主題分布比例 (不含雜訊主題)',
    labels={'Topic': '主題編號', 'Percentage': '占比 (%)'},
    text=plot_df['Percentage'].apply(lambda x: f'{x:.2f}%'),
    custom_data=['Keywords']  # 添加關鍵字數據用於hover
)

# 更新圖表樣式和hover模板
fig.update_traces(
    textposition='outside',
    marker_color='lightblue',
    hovertemplate="主題 %{x}<br>占比: %{y:.2f}%<br>關鍵字:<br>%{customdata}<extra></extra>"
)

fig.update_layout(
    width=1200,
    height=600,
    showlegend=False,
    title_x=0.5,
    title_font_size=20,
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
    )
)

# 保存圖表
fig.write_html(f"{visualization_path}/topic_distribution_without_noise.html")
print(f"主題分布圖已保存至: {visualization_path}/topic_distribution_without_noise.html")

# 打印詳細的統計信息
print("\n主題分布統計（含前五關鍵字）：")
print("=" * 100)
for _, row in plot_df.iterrows():
    print(f"主題 {row['Topic']:2d}: {row['Percentage']:5.2f}% ({topic_counts[row['Topic']]} 篇文章)")
    print(f"關鍵字: {row['Keywords']}")
    print("-" * 100)

# 生成主題比例圖
try:
    # 獲取主題比例
    topic_proportions = topic_model.get_topic_info().sort_values("Count", ascending=True)
    
    # 過濾掉�聲主題（-1）
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



