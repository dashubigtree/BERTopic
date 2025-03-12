import plotly.express as px
import numpy as np
import pandas as pd
import plotly.io as pio

# Step 1: 設置 Plotly 渲染環境
# 根據你的環境選擇適當的渲染器
pio.renderers.default = 'browser'  # 在瀏覽器中顯示圖表
# 如果你在 Jupyter Notebook 中，可以使用 'notebook'
# pio.renderers.default = 'notebook'

# Step 2: 生成模擬數據
# 假設我們有 100 個點，降維到 2D，模擬 BERTopic 的 reduced_embeddings
np.random.seed(42)
x = np.random.randn(100)  # X 座標
y = np.random.randn(100)  # Y 座標
labels = np.random.choice(['Topic 1', 'Topic 2', 'Topic 3'], size=100)  # 模擬主題標籤

# 將數據整理成 DataFrame，方便 Plotly 使用
df = pd.DataFrame({
    'x': x,
    'y': y,
    'label': labels,
    'text': [f"Document {i}" for i in range(100)]  # 模擬文件標籤
})

# Step 3: 使用 Plotly Express 繪製散點圖
fig = px.scatter(
    df,
    x='x',
    y='y',
    color='label',  # 根據標籤上色，模擬主題
    hover_data=['text'],  # 懸停時顯示文件標籤
    title='測試 Plotly 散點圖',
    width=800,
    height=600
)

# Step 4: 顯示並保存圖表
fig.show()
fig.write_html("plotly_scatter_test.html")
print("Scatter plot saved as 'plotly_scatter_test.html'.")