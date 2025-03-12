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


# 定義噪音字元集合
noise_chars = {'zvx','\U0001F48E', '。', '▲', '△', '\U0001F50D', '？', '—', '<', '∶', '\\r', '；', '✦', '\\u200c', '️', '－', '℃', '‖', '!', '「', '→', '/', '║', '」', '@', '，', '?', "\"", '○', '）', '『', '．', '\U0001F449', '】', '\U0001F31F', '=', '\U0001F447', '‰', '【', ';', '#', ')', '：', '\\u200d', '❖', '~', ']', '%', '·', '↑', '（', '〕', '☆', '※', '&', '•', '\U0001F44D', '>', '／', '▌', '–', '↓', '[', ''', ':', '《', '▎', '\U0001F91D', '©', '+', '\U0001F30A', '\\xa0', '\\n', '◇', ',', '◎', '…', '(', '〔', '\\\\', '\"', '■', '｜', '─', '\\u200b', '-', '●', '\"', '▊', '、', '︱', ''', '*', '⭐', '》', '％', '！', '〉', '|', '▼', '\U0001F446', '\U0001F3E1', '°', '\\t', '』', '〈', '～', '◆', '.', '⬆', '\"'}

def clean_text(text, noise_chars):
    noise_pattern = f"[{''.join(re.escape(char) for char in noise_chars)}]"
    return re.sub(noise_pattern, "", text)

# 讀取停用詞並添加自定義高頻詞
stopwords_file_path = "./data/stop_words.txt"
with open(stopwords_file_path, encoding='utf-8') as f:
    stop_words = set([line.strip() for line in f])

# 添加自定義高頻詞到停用詞
additional_stopwords = {"免责声明","文章描述","免责","删除","网络文章","旨在倡导","不良引导", "文章旨在","倡导社会","低俗","低俗不良","过程图片","图片","不良","来源于"}
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
    min_cluster_size=20,    # 調整以獲得約20多個主題
    min_samples=5,          # 增加樣本數以獲得更穩定的群集
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
    alpha=0.5               # 增加 alpha 值以產生更明顯的群集
)

vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    stop_words=None,
    max_features=15000,
    max_df=0.9,
    min_df=3
)

# 修改 ClassTfidfTransformer 的設置
ctfidf_model = ClassTfidfTransformer(
    seed_words=[
        "條約", "防禦", "執法", "金門",
        "兩岸", "事件", "協議", "海域",
        "漁權", "漁業", "經濟", "台灣",
        "中國", "海巡", "大陸", "國民黨"
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
    min_topic_size=35,     # 與 HDBSCAN 的 min_cluster_size 保持一致
    ctfidf_model=ctfidf_model,
    language= "chinese"
)


# 先生成嵌入向量
# 在生成嵌入向量後添加以下診斷代碼
print("生成文檔嵌入向量...")
embeddings = embedding_model.encode(texts, show_progress_bar=True)
print(f"嵌入向量形狀: {embeddings.shape}")


# 訓練模型時傳入嵌入向量
print("開始訓練模型...")
topics, probs = topic_model.fit_transform(texts, embeddings)



# 儲存模型
model_save_path = "./model/bertopic_four_categories_v2"
topic_model.save(model_save_path)
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
visualization_path = "./visualization/four_categories_v2"
os.makedirs(visualization_path, exist_ok=True)

# 獲取主題資訊
topic_info = topic_model.get_topic_info()

# 使用 BERTopic 原生的視覺化功能
# 1. Intertopic Distance Map (主題間距離圖) - 這就是您要求的第一項
fig_intertopic = topic_model.visualize_topics()
fig_intertopic.write_html(f"{visualization_path}/intertopic_distance_map.html")


# 先進行降維處理
print("進行文檔降維...")
reduced_embeddings = UMAP(
    n_neighbors=15,
    n_components=2,
    min_dist=0.1,        # 增加最小距離
    metric='cosine',
    random_state=42
).fit_transform(embeddings)

# 在 visualize_documents 之前添加診斷代碼
print("檢查輸入數據...")
print(f"reduced_embeddings 形狀: {reduced_embeddings.shape}")
print(f"topics 長度: {len(topics)}")
print(f"texts 長度: {len(texts)}")
print(f"topics 中的唯一值: {set(topics)}")
print(f"reduced_embeddings 是否包含 NaN: {np.isnan(reduced_embeddings).any()}")

# 檢查數據一致性
if len(topics) != len(texts) or len(topics) != reduced_embeddings.shape[0]:
    print("警告：數據長度不一致！")
    print(f"topics: {len(topics)}, texts: {len(texts)}, embeddings: {reduced_embeddings.shape[0]}")

# 2. 文檔視覺化 - 使用降維後的 embeddings
try:
    print("嘗試生成視覺化...")
    fig_docs_original = topic_model.visualize_documents(
        docs=texts,
        reduced_embeddings=reduced_embeddings,
        topics=topics,
        width=1200,
        height=800,
        hide_document_hover=False,
        hide_annotations=False
    )
    print("視覺化生成成功！")
    
    # 檢查圖表對象
    print(f"圖表類型: {type(fig_docs_original)}")
    print(f"圖表數據: {fig_docs_original.data}")

except Exception as e:
    print(f"視覺化生成失敗: {e}")
    print(f"錯誤類型: {type(e)}")
    import traceback
    print(f"詳細錯誤信息: {traceback.format_exc()}")

# 添加預覽功能
print("正在生成預覽...")
import plotly.io as pio

# 確認圖表內容
print(f"圖表點數: {len(fig_docs_original.data[0].x)}")  # 檢查數據點數量
print(f"圖表維度: {fig_docs_original.layout.width}x{fig_docs_original.layout.height}")  # 檢查圖表尺寸

# 然後再儲存 HTML
try:
    print("正在儲存 HTML...")
    fig_docs_original.write_html(f"{visualization_path}/documents_visualization_original.html")
    print("HTML 儲存成功！")
except Exception as e:
    print(f"HTML 儲存失敗: {e}")


# 3. 視覺化主題層次結構 (Hierarchical clustering) - 這是您要求的第三項
fig_hierarchy = topic_model.visualize_hierarchy()
fig_hierarchy.write_html(f"{visualization_path}/hierarchical_clustering.html")


# 其他有用的視覺化
# 5. 視覺化主題相似性熱圖
fig_heatmap = topic_model.visualize_heatmap()
fig_heatmap.write_html(f"{visualization_path}/heatmap_visualization.html")

# 6. 為每個主題生成關鍵詞條形圖
for topic in topic_info['Topic'].tolist():
    if topic != -1:  # 排除雜訊主題
        fig_barchart = topic_model.visualize_barchart(topics=[topic], n_words=20, title=f"主題 {topic} 關鍵詞")
        fig_barchart.write_html(f"{visualization_path}/topic_{topic}_keywords.html")



print(f"所有視覺化結果已保存到 {visualization_path}")


# 輸出主題資訊
print("\n主題分布概況：")
print(topic_info)

# 1. 將 topic_info 的 print 資訊儲存到指定路徑
topic_info_save_path = "./results/four_categories_v2/topic_info.txt"
with open(topic_info_save_path, "w", encoding="utf-8") as f:
    f.write("主題分布概況：\n")
    f.write(topic_info.to_string())
print(f"主題資訊已保存至: {topic_info_save_path}")

# 2. 將各主題的代表性文章原始內容列出來
representative_docs_save_path = "./results/four_categories_v2/representative_docs.txt"
with open(representative_docs_save_path, "w", encoding="utf-8") as f:
    for topic in topic_info['Topic'].tolist():
        if topic == -1:  # 跳過雜訊主題
            continue
        
        representative_docs = topic_model.get_representative_docs(topic)
        f.write(f"\n主題 {topic} 的代表性文章：\n")
        
        for i, doc in enumerate(representative_docs[:]):  
            # 找到原始文章
            original_index = texts.index(doc)  # 獲取文檔在texts中的索引
            original_row = df.iloc[original_index]  # 獲取對應的原始數據行
            original_content = original_row['content']  # 獲取原始文檔內容
            article_id = original_row.name  # 獲取文章在原始 CSV 中的行號（ID）
            
            f.write(f"文檔 {i+1} (行號: {article_id}):\n")
            f.write(original_content + "\n")
            f.write("-" * 50 + "\n")
print(f"代表性文章原始內容已保存至: {representative_docs_save_path}")

# 3. 將分類結果標示在原始的 CSV 文章列表中
# 先建立一個副本
df_with_topics = df.copy()
df_with_topics['topic'] = topics  # 添加主題標籤

# 保存到新的 CSV 文件
csv_with_topics_path = "./results/four_categories_v2/Kinmen_splitData_with_topics.csv"
df_with_topics.to_csv(csv_with_topics_path, index=False, encoding="utf-8-sig")
print(f"已將分類結果標示在原始 CSV 文件中，保存至: {csv_with_topics_path}")

# 輸出主要主題的關鍵詞
print("\n主要主題的關鍵詞：")
for topic in topic_info.head()['Topic'].tolist():
    if topic != -1:  # 排除雜訊主題
        print(f"\n主題 {topic}:")
        print(topic_model.get_topic(topic))

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
    nr_bins=20
)

# 創建視覺化
fig_topics_over_time = topic_model.visualize_topics_over_time(
    topics_over_time,
    top_n_topics=10,
    width=1200,
    height=600
)

# 保存視覺化結果
fig_topics_over_time.write_html(f"{visualization_path}/topics_over_time.html")
print(f"主題隨時間變化圖已保存至: {visualization_path}/topics_over_time.html")

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


