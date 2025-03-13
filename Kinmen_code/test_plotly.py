from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP

# Prepare embeddings
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
print("\n=== 檢查文章數據 ===")
print(f"總文章數量: {len(docs)}")

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=False)
print(f"嵌入向量數量: {len(embeddings)}")
print(f"嵌入向量維度: {embeddings.shape}")

# Train BERTopic
topic_model = BERTopic().fit(docs, embeddings)

# Run the visualization with the original embeddings
topic_model.visualize_documents(docs, embeddings=embeddings)

# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
fig = topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)

visualization_path = "./visualization"

# 在儲存之前檢查圖表數據
print("\n=== 檢查視覺化數據 ===")
if fig.data:
    print(f"數據點數量: {len(fig.data[0].x)}")
    print(f"圖表類型: {fig.data[0].type}")
    print(f"圖表模式: {fig.data[0].mode}")
    
    # 檢查座標數據
    if len(fig.data[0].x) > 0 and len(fig.data[0].y) > 0:
        print("✓ 視覺化包含有效的數據點")
        print(f"X 座標範圍: [{min(fig.data[0].x):.2f}, {max(fig.data[0].x):.2f}]")
        print(f"Y 座標範圍: [{min(fig.data[0].y):.2f}, {max(fig.data[0].y):.2f}]")
        fig.write_html(f"{visualization_path}/test_documents_visualization_original.html")
    else:
        print("❌ 視覺化沒有有效的座標數據")
else:
    print("❌ 視覺化圖表沒有數據")
