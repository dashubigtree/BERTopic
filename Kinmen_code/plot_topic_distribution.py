import pandas as pd
import plotly.express as px
import plotly.io as pio

# 讀取CSV文件
df = pd.read_csv("./results/four_categories_v2/Kinmen_splitData_with_topics.csv")

# 過濾掉 -1 主題，並計算其他主題的文章數量和比例
df_filtered = df[df['topic'] != -1]
topic_counts = df_filtered['topic'].value_counts()
total_docs = len(df_filtered)
topic_percentages = (topic_counts / total_docs * 100).round(2)

# 創建數據框，包含主題編號和百分比
plot_df = pd.DataFrame({
    'Topic': topic_counts.index,
    'Percentage': topic_percentages.values
})

# 排序
plot_df = plot_df.sort_values('Topic')

# 創建柱狀圖
fig = px.bar(
    plot_df,
    x='Topic',
    y='Percentage',
    title='主題分布比例 (不含雜訊主題)',
    labels={'Topic': '主題編號', 'Percentage': '占比 (%)'},
    text=plot_df['Percentage'].apply(lambda x: f'{x:.2f}%')
)

# 更新圖表樣式
fig.update_traces(
    textposition='outside',
    marker_color='lightblue'
)

fig.update_layout(
    width=1200,
    height=600,
    showlegend=False,
    title_x=0.5,
    title_font_size=20
)

# 保存圖表
output_path = "./visualization/four_categories_v2/topic_distribution_without_noise.html"
fig.write_html(output_path)
print(f"主題分布圖已保存至: {output_path}")

# 打印詳細的統計信息
print("\n主題分布統計（不含雜訊主題）：")
for topic, percentage in zip(plot_df['Topic'], plot_df['Percentage']):
    print(f"主題 {topic:2d}: {percentage:5.2f}% ({topic_counts[topic]} 篇文章)")