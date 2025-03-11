import os
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
import plotly.io as pio

# 設置輸出目錄
visualization_path = "/Users/shuyuhsu/code_workspace/Kinmen_wechat_BERTTopic/Kinmen_code/visualization/test"
os.makedirs(visualization_path, exist_ok=True)

# 準備數據（限制數量以加快測試）
print("準備數據...")
docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data'][:1000]
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=True)

# 訓練 BERTopic 模型
print("訓練模型...")
topic_model = BERTopic().fit(docs, embeddings)

print("生成視覺化結果...")

# 先進行降維
print("進行降維處理...")
reduced_embeddings = UMAP(
    n_neighbors=10,
    n_components=2,
    min_dist=0.0,
    metric='cosine'
).fit_transform(embeddings)

# 1. 使用原始 embeddings 生成視覺化
print("生成原始 embeddings 視覺化...")
fig_original = topic_model.visualize_documents(
    docs=docs,
    embeddings=embeddings,
    reduced_embeddings=reduced_embeddings,  # 加入降維後的 embeddings
    width=900,
    height=700
)
fig_original.write_html(f"{visualization_path}/doc_viz_original.html")

# 2. 使用降維後的 embeddings 生成視覺化
print("生成降維後的視覺化...")
fig_reduced = topic_model.visualize_documents(
    docs=docs,
    reduced_embeddings=reduced_embeddings,
    width=900,
    height=700
)
fig_reduced.write_html(f"{visualization_path}/doc_viz_reduced.html")

print(f"\n視覺化結果已保存至: {visualization_path}")
print("請在瀏覽器中打開 HTML 文件查看結果")
