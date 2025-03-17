# 2025/02/24
因為忘記上次使用哪個conda env來執行BERTopic，目前測試NLP3.10這個環境

# 2025/02/25
因為又出現版本是配問題，還有檔案缺失（不知道是什麼內建檔案不見）。所以重新創了一個conda env，並且也重新使用conda install 需要的套件

## 套件安裝順序

### 核心套件（會自動安裝相依套件，包含 sentence-transformers, umap-learn, hdbscan, scikit-learn）
conda install -c conda-forge bertopic

### 資料處理相關
conda install pandas
conda install -c conda-forge gensim
conda install -c conda-forge jieba
conda install tqdm

### 視覺化相關
conda install -c conda-forge plotly

## 程式碼說明
analyze_paragraphs.py: 主要用於統計經過paragraph_splitData_0223Version.py的切割結果進行「平均長度」的統計，接著最後再重新使用平均的長度切分成長度差不多的段落

## 參數調整說明
### 預處理策略
針對主題過度集中的問題，可以採取以下方法：
1. 在停用詞中加入高頻但不具分析價值的詞（如：海巡、海警、金門）
2. 調整 HDBSCAN 的 min_cluster_size 和 min_samples 參數，使群集更容易形成
3. 降低 cluster_selection_epsilon 值，使群集界限更嚴格

### Seed Words 說明
seed_words 的作用和影響：
1. 功能說明：
   - 為特定關鍵字設定較高權重
   - 影響 c-TF-IDF 計算結果
   - 引導模型關注特定主題

2. 對分析流程的影響：
   - UMAP：seed words 不影響降維過程
   - HDBSCAN：seed words 不影響群集形成
   - c-TF-IDF：
     - 提高指定詞彙的重要性
     - 影響最終主題的關鍵字排序
     - 可能導致相關詞彙更容易出現在主題表示中

3. 使用建議：
   - 避免將高頻詞設為 seed words
   - 選擇具有主題指示性的詞彙
   - 可以根據初步分析結果調整 seed words
### UMAP and HDBSCAN 參數說明
UMAP：主要用於將文本資料經過相量化的結果進行降維，以便於視覺化

HDBSCAN：主要用於將向量進行密度分群，其運作流程為：
1. 對 UMAP 降維後的向量進行密度分群
2. 依據 min_cluster_size 參數（預設100）確保每個群集至少包含指定數量的文章
3. 使用 min_samples 參數（預設5）決定形成核心點所需的最小樣本數
4. 透過 cluster_selection_epsilon 參數（預設0.2）控制群集的邊界範圍

分群完成後，BERTopic 會：
1. 使用 c-TF-IDF 對每個群集進行關鍵字分析
2. 只有達到 min_topic_size 大小的群集才會被視為有效主題
3. 根據 seed_words 對關鍵字進行權重調整
4. 最後輸出每個主題的代表性關鍵字

## 0223 version2 優化策略
### 當前問題
1. 主題1（海域相關）過度主導（7892篇）
2. 其他潛在重要主題（如條約、防禦等）被掩蓋
3. 主題分布極不均衡

### 優化建議
1. 預處理階段：
   - 將 "海域"、"海巡"、"海警"、"金門" 等高頻詞加入停用詞
   - 保留 "執法"、"事件"、"兩岸" 等具分析價值的詞

2. 參數調整：
   - 降低 min_cluster_size（如從 100 降至 50）
   - 降低 min_samples（如從 5 降至 3）
   - 調整 cluster_selection_epsilon（如從 0.2 降至 0.1）

3. Seed Words 策略：
   - 移除現有的海巡相關 seed words
   - 新增其他感興趣的主題詞，如：
     ```python
     seed_words = [
         "條約", "防禦", "執法",
         "兩岸", "事件", "協議",
         "漁權", "漁業", "經濟"
     ]
     ```

### 預期效果
1. 降低海巡相關主題的主導性
2. 提升其他主題的可見度
3. 獲得更均衡的主題分布
### 問題
會有一個主題是超級沒有用：
[('文章', 1.0269343205778318), ('免责 声明', 0.6371709728328179), ('文章 描述', 0.6306349752359934), ('免责', 0.6295520228654853), ('删除', 0.6235244826370268), ('网络 文章', 0.6150808659871279), ('旨在 倡导', 0.6150808659871279), ('不良 引导', 0.6150808659871279), ('文章 旨在', 0.6150808659871279), ('倡导 社会', 0.6150808659871279), ('社会 能量', 0.6150808659871279), ('低俗', 0.6150808659871279), ('低俗 不良', 0.6150808659871279), ('描述 过程', 0.6150808659871279), ('能量 低俗', 0.6150808659871279), ('过程 图片', 0.6150808659871279), ('图片', 0.6107905314165205), ('不良', 0.6106096891562324), ('倡导', 0.6063739491540482), ('来源于', 0.5976931732338989)]

## 0223 version2-2 調整
加回金門、海警等關鍵字，讓文字的發生背景更為明確。但是也因此在移除背景關鍵字之後發現，還有其他不相關的詞需要刪除。所以將一些額外需要刪除的stopwords加入。

# 0226 任務
1. 視覺化要有時間得變化
2. 使用專有名詞的dict 去做jieba
3. m503, 航線, 菲律賓 (也是重要的關鍵字)
4. 有新的stopwords要加：下圖, 圖片
5. 指定特有主題的關鍵字

# 0304 任務
我發現Model2的模型檔案被我覆蓋掉了，目前的分類結果主題數量都只有六七個。但是我的視覺化結果有一次分類的結果非常好，大約有二十多個，且主要有四大面向。所以我現在正在嘗試將這個模型復刻
模型復刻結果叫做optimized modle2: 但是這個模型的結果雖然主題數量變多了，結果有點分散
-> 針對分散的狀況，我重新下prompt來進行修正。讓分類結果集中，原本預設希望會有四個主要大類別，最終得到的結果是four_categories_v1模型，此模型得到三大類型
(I changed the topic into four_categories_v1)
目標：
1. 要將模型分類結果為四大類型
2. 分類結果中的representitive article要能夠顯示來細讀其中的內容
3. 將模型針對各文章分類的主題結果編號標回原本文章列表

### 新問題
目前從representative articles當中發現好像有些一樣的分段結果。可能是因為有些貼文有公版，所以在訓練的時候會顯示一樣的段落結果
- 新增了一個python file: callOUt_representativeArticle.py，用來將representative article的結果輸出成txt檔案，方便我後續的分析。在這邊也多加入了自定義「顯示文章數量」，讓每個主題顯示的文章內容數量可以自行調整。


# 0310 missions
- 要畫出docs的文章散點圖
- 要研究具有時間變化的主題變化折線圖
- 去統整出每個主題的文章數量占據總文章數量的比例
### four_categories_v1 模型特色
- 在分類上有指定主題文章數量

在 `four_categories_v1.py` 中，有幾個關鍵的設定來控制和克制主題分類的結果，主要體現在以下幾個方面：

1. **HDBSCAN 參數設定**：
   ```python
   hdbscan_model = HDBSCAN(
       min_cluster_size=35,    # 調整以獲得約20多個主題
       min_samples=5,          # 增加樣本數以獲得更穩定的群集
       metric='euclidean',
       cluster_selection_method='eom',
       prediction_data=True,
       alpha=1.0               # 增加 alpha 值以產生更明顯的群集
   )
   ```
   - `min_cluster_size=35`：控制每個主題至少需要包含 35 個文檔，這避免了過小的主題產生
   - `min_samples=5`：增加樣本數以獲得更穩定的群集，減少噪音影響
   - `alpha=1.0`：增加 alpha 值使群集更加明顯，有助於產生更清晰的主題邊界

2. **BERTopic 模型參數**：
   ```python
   topic_model = BERTopic(
       embedding_model=embedding_model,
       verbose=True,
       calculate_probabilities=True,
       nr_topics=25,          # 明確指定約25個主題
       umap_model=umap_model,
       hdbscan_model=hdbscan_model,
       vectorizer_model=vectorizer,
       top_n_words=20,
       min_topic_size=35,     # 與 HDBSCAN 的 min_cluster_size 保持一致
       ctfidf_model=ctfidf_model,
   )
   ```
   - `nr_topics=25`：明確限制主題數量約為 25 個
   - `min_topic_size=35`：與 HDBSCAN 的 `min_cluster_size` 保持一致，確保每個主題有足夠的文檔

3. **自定義層次化主題分類**：
   ```python
   def custom_hierarchical_topics(embeddings, topics):
       # 排除雜訊主題 (-1)
       mask = topics != -1
       filtered_embeddings = embeddings[mask]
       filtered_topics = topics[mask]
       
       # 使用 K-means 將主題分為4大類
       kmeans = KMeans(n_clusters=4, random_state=42)
       super_topics = kmeans.fit_predict(filtered_embeddings)
       
       # 創建主題到超主題的映射
       topic_to_super_topic = {}
       for topic, super_topic in zip(filtered_topics, super_topics):
           topic_to_super_topic[topic] = super_topic
       
       # 將雜訊主題映射到 -1
       topic_to_super_topic[-1] = -1
       
       return topic_to_super_topic
   ```
   - 這個函數強制將所有主題（除了雜訊主題）分為 4 個超主題
   - 使用 K-means 算法確保這 4 個超主題的劃分是基於主題嵌入向量的相似性

4. **ClassTfidfTransformer 設定**：
   ```python
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
   ```
   - `seed_words`：提供種子詞彙引導主題形成，這些詞彙會影響主題的關鍵詞提取
   - `bm25_weighting=True`：使用 BM25 加權方式，更好地識別主題特徵詞
   - `reduce_frequent_words=True`：減少高頻詞的影響，避免常見詞主導主題

5. **UMAP 降維參數**：
   ```python
   umap_model = UMAP(
       n_neighbors=15,      # 增加鄰居數量以捕獲更多局部結構
       n_components=2,      # 降為2維以便更好地可視化四大類
       metric='cosine',
       min_dist=0.05,       # 適中的最小距離
       random_state=42      # 固定隨機種子以獲得可重複的結果
   )
   ```
   - `n_neighbors=15`：增加鄰居數量以捕獲更多局部結構，有助於形成更穩定的主題
   - `n_components=2`：降為 2 維有助於更好地可視化四大類
   - `min_dist=0.05`：適中的最小距離，控制點之間的間隔

6. **停用詞和噪音字元處理**：
   - 定義了大量噪音字元和停用詞，確保這些不會影響主題分類
   - 添加了自定義高頻詞到停用詞，如 `"免责声明","文章描述"` 等

這些設定共同作用，確保主題分類結果具有一定的克制性和可解釋性，避免過度細分或過度合併主題，同時通過超主題的設定，將所有主題強制歸類為四大類。

### 處理BERTopic當中的visualize document問題
1. 在 BERTopic 模型中設置 embedding_model ：
```python
topic_model = BERTopic(
    embedding_model=embedding_model,
    # ... 其他參數 ...
)
 ```

這一步是告訴 BERTopic 使用哪個模型來生成嵌入向量。這只是設置了模型的配置，還沒有實際執行嵌入向量的生成。

2. 預先生成嵌入向量：
```python
embeddings = embedding_model.encode(texts, show_progress_bar=True)
topics, probs = topic_model.fit_transform(texts, embeddings)
```

這一步是實際執行嵌入向量的生成，並將生成好的向量傳給模型使用。這樣做的好處是：

- 可以重複使用這些嵌入向量，而不需要重複計算
- 可以更好地控制嵌入過程（例如顯示進度條）
- 提高了代碼的效率，避免了模型內部重複計算嵌入向量
如果不預先生成嵌入向量，而是直接使用：

```python
topics, probs = topic_model.fit_transform(texts)
 ```


BERTopic 會在內部使用設置的 embedding_model 來生成嵌入向量，但這樣就無法重複使用這些向量，也無法顯示進度條。

所以，目前的寫法是更好的實踐方式，不會造成重複計算的問題。

# 0311 任務紀錄
目前正在解決visualize_documents無法顯示文章的散點圖問題。
已經確認在visualize_documents要提供的參數：docs= 在輸入主題模型的文字（在這邊提供texts就是最終輸入到BERTopic進行主題分類的文字，是經過切詞的文字；topics= 選擇要顯示哪幾個編號的主題；embedding= 就是提供docs經過同樣embedding_model轉換後的結果），接著後續要在視覺化的時候就會直接使用topic_model(也就是我們前面設定好的BERTopic model)的UMAP降維；reduced_embedding= 自己進行UMAP來降成2D。

但是現在我就算重新確認這樣參數之後，我還是無法顯示視覺化的結果。本來以為和文章的數量有關，但是8789篇的文章數量依該也是沒問題。就算有問題，我也已經將需要顯示的topic數量降低到2的時候也還是無法正常顯示出文章的散點圖。
就算我自己增加reduced_embedding，也還是無法正常顯示文章的散點圖。

在另一台電腦執行此指令可以安裝yml檔案中的套件
```bash
conda env create -f environment.yml
```

# 0312 progress
我嘗試在大一點的ram去執行看看，可是最終圖片還是出不來。
目前已經完成各主題的比例以及topic over time

# 0313 progress
看起來散點圖還是會失敗，就算我直接用reduced_embeddings的結果去畫scatter，他顯示的效果也不太好。

# 0316 progress
four_catrgories_v3 只是嘗試一些不同的畫法。最後visualize documents 就用v2 的就好
今天發現在8789篇文章當中，還有幾篇文章出現空字串的問題。所以修改一下之後，最終有效文本只會有8776篇
這樣處理之後還是畫不出來。

# 0317 record
four_categories_v4 是和four_categories_v2一模壹樣，只是為了不要讓老師那邊搞混

新增一個BERTopic_KeyBertopic.py檔案，用來測試KeyBert的效果。   