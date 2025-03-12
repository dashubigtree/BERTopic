import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
import plotly.io as pio

# Step 1: 設置 Plotly 渲染環境（確保圖表能正常顯示）
pio.renderers.default = 'browser'  # 在瀏覽器中顯示圖表

# Step 2: 準備中文文件數據
# 假設我們有一個簡單的中文數據集
docs = [
    "我喜歡吃中國菜，因為它很美味。",
    "中國菜有很多種類，例如川菜和粵菜。",
    "我昨天去了一家中餐館，點了麻婆豆腐。",
    "川菜的特點是麻辣，粵菜則比較清淡。",
    "我最喜歡的中國菜是北京烤鴨。",
    "昨天的天氣很好，我和朋友去爬山了。",
    "爬山是一項很好的運動，可以鍛煉身體。",
    "我們爬到了山頂，風景非常美麗。",
    "下次我想去海邊玩，聽說那裡很放鬆。",
    "海邊的風景很美，還可以吃海鮮。"
]

# Step 3: 使用多語言嵌入模型生成嵌入
# 使用支援中文的多語言模型
sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=True)

# 檢查嵌入形狀
print("Embeddings shape:", embeddings.shape)  # 應為 (10, 384)

# Step 4: 訓練 BERTopic 模型
topic_model = BERTopic(
    language="multilingual",  # 設置為多語言模式
    verbose=True
)
topics, probs = topic_model.fit_transform(docs, embeddings)

# 檢查主題數量
print("Number of topics:", len(set(topics)) - (1 if -1 in topics else 0))

# Step 5: 降維嵌入以進行可視化
reduced_embeddings = UMAP(
    n_neighbors=2,  # 因為數據量小，設置較小的 n_neighbors
    n_components=2,
    min_dist=0.0,
    metric='cosine',
    random_state=42
).fit_transform(embeddings)

# 檢查降維結果
print("Reduced embeddings shape:", reduced_embeddings.shape)  # 應為 (10, 2)

# 檢查是否有 NaN 或無限值
if np.any(np.isnan(reduced_embeddings)) or np.any(np.isinf(reduced_embeddings)):
    print("Warning: Reduced embeddings contain NaN or infinite values!")
else:
    print("Reduced embeddings look normal.")

# Step 6: 可視化文件
fig = topic_model.visualize_documents(
    docs,
    reduced_embeddings=reduced_embeddings,
    title="中文文件的散點圖",
    width=1000,
    height=600
)

# Step 7: 顯示並保存圖表
fig.show()
fig.write_html("chinese_documents_visualization.html")
print("Visualization saved as 'chinese_documents_visualization.html'.")