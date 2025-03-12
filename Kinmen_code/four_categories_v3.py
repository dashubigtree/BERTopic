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
additional_stopwords = {
    "免责声明","文章描述","免责","删除","网络文章","旨在倡导","不良引导", "文章旨在","倡导社会","低俗","低俗不良","过程图片",
    "图片","不良","来源于","口感","复盆子","阅读原文","阅读","原文","版权","文章旨在","文章","cctv4","上方cctv4","cctv4 关注","点击上方","上方",
    "朋友圈","一键","一键分享","朋友圈","点击","下图","分享","右侧","下方","左侧"
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
model_save_path = "./model/bertopic_four_categories_v3"  # 從 v2 改為 v3
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
visualization_path = "./visualization/four_categories_v3"  # 從 v2 改為 v3
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

# =====================================================================
# BERTopic visualize_documents 增強診斷工具
# =====================================================================
import numpy as np
import pandas as pd
import os
import traceback
import time
import json
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import logging
import sys

# 創建診斷日誌目錄
diagnostic_path = f"{visualization_path}/diagnostics"
os.makedirs(diagnostic_path, exist_ok=True)

# 設置詳細日誌
diagnostic_log = f"{diagnostic_path}/visualize_documents_diagnosis.log"
with open(diagnostic_log, 'w', encoding='utf-8') as f:
    f.write("BERTopic visualize_documents 增強診斷日誌\n")
    f.write("=" * 60 + "\n\n")

def log_diagnostic(message, level="INFO"):
    """記錄診斷信息到日誌文件和控制台"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{level}] {message}"
    with open(diagnostic_log, 'a', encoding='utf-8') as f:
        f.write(f"{formatted_message}\n")
    
    if level == "ERROR":
        logging.error(message)
    elif level == "WARNING":
        logging.warning(message)
    else:
        logging.info(message)

def save_error_details(error, test_name):
    """保存詳細錯誤信息到單獨的文件"""
    error_file = f"{diagnostic_path}/error_{test_name}.txt"
    with open(error_file, 'w', encoding='utf-8') as f:
        f.write(f"錯誤類型: {type(error).__name__}\n")
        f.write(f"錯誤信息: {str(error)}\n\n")
        f.write("詳細錯誤堆疊:\n")
        f.write(traceback.format_exc())
    log_diagnostic(f"詳細錯誤信息已保存到: {error_file}", "INFO")

def save_test_result(test_name, success, fig=None, details=None):
    """保存測試結果和可選的圖表"""
    # 記錄測試結果
    result_file = f"{diagnostic_path}/result_{test_name}.json"
    result = {
        "test_name": test_name,
        "success": success,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "details": details or {}
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    # 如果有圖表且測試成功，保存圖表
    if fig is not None and success:
        html_file = f"{diagnostic_path}/{test_name}.html"
        fig.write_html(html_file)
        log_diagnostic(f"測試 '{test_name}' 成功，圖表已保存到: {html_file}")
        return html_file
    elif success:
        log_diagnostic(f"測試 '{test_name}' 成功")
    else:
        log_diagnostic(f"測試 '{test_name}' 失敗", "ERROR")
    
    return None

def diagnose_visualize_documents(topic_model, texts, reduced_embeddings, topics, visualization_path):
    """系統性診斷 visualize_documents 方法的問題"""
    
    log_diagnostic("\n=== 開始系統性診斷 BERTopic visualize_documents ===\n")
    
    # 記錄系統環境信息
    log_diagnostic("=== 系統環境信息 ===")
    import platform
    log_diagnostic(f"Python 版本: {platform.python_version()}")
    log_diagnostic(f"操作系統: {platform.system()} {platform.version()}")
    
    # 記錄相關庫版本
    log_diagnostic("\n=== 相關庫版本 ===")
    import bertopic
    import plotly
    import numpy
    import pandas
    import sklearn
    
    log_diagnostic(f"BERTopic 版本: {bertopic.__version__}")
    log_diagnostic(f"Plotly 版本: {plotly.__version__}")
    log_diagnostic(f"NumPy 版本: {numpy.__version__}")
    log_diagnostic(f"Pandas 版本: {pandas.__version__}")
    log_diagnostic(f"Scikit-learn 版本: {sklearn.__version__}")
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        log_diagnostic(f"當前 Python 進程記憶體使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        log_diagnostic(f"系統可用記憶體: {psutil.virtual_memory().available / 1024 / 1024:.2f} MB")
    except ImportError:
        log_diagnostic("無法獲取記憶體使用信息 (psutil 未安裝)", "WARNING")
    
    # 1. 基本數據檢查
    log_diagnostic("\n=== 1. 基本數據檢查 ===")
    data_ok = True
    
    # 檢查數據長度
    log_diagnostic(f"文本數量: {len(texts)}")
    log_diagnostic(f"主題數量: {len(topics)}")
    log_diagnostic(f"嵌入向量形狀: {reduced_embeddings.shape}")
    
    if len(texts) != len(topics) or len(texts) != reduced_embeddings.shape[0]:
        log_diagnostic("❌ 數據長度不匹配", "ERROR")
        data_ok = False
    else:
        log_diagnostic("✓ 數據長度匹配")
    
    # 檢查嵌入向量
    if np.isnan(reduced_embeddings).any():
        log_diagnostic("❌ 嵌入向量包含 NaN", "ERROR")
        nan_count = np.isnan(reduced_embeddings).sum()
        nan_indices = np.where(np.isnan(reduced_embeddings).any(axis=1))[0]
        log_diagnostic(f"  NaN 值數量: {nan_count}")
        log_diagnostic(f"  包含 NaN 的索引: {nan_indices[:10]}...")
        data_ok = False
    else:
        log_diagnostic("✓ 嵌入向量不包含 NaN")
    
    if np.isinf(reduced_embeddings).any():
        log_diagnostic("❌ 嵌入向量包含無限值", "ERROR")
        inf_count = np.isinf(reduced_embeddings).sum()
        inf_indices = np.where(np.isinf(reduced_embeddings).any(axis=1))[0]
        log_diagnostic(f"  無限值數量: {inf_count}")
        log_diagnostic(f"  包含無限值的索引: {inf_indices[:10]}...")
        data_ok = False
    else:
        log_diagnostic("✓ 嵌入向量不包含無限值")
    
    # 檢查嵌入向量範圍
    embedding_min = np.min(reduced_embeddings)
    embedding_max = np.max(reduced_embeddings)
    log_diagnostic(f"嵌入向量範圍: [{embedding_min:.4f}, {embedding_max:.4f}]")
    
    if embedding_min < -100 or embedding_max > 100:
        log_diagnostic("⚠️ 嵌入向量範圍較大，可能需要標準化", "WARNING")
    
    # 檢查主題標籤
    unique_topics = sorted(set(topics))
    log_diagnostic(f"唯一主題數量: {len(unique_topics)}")
    log_diagnostic(f"唯一主題值: {unique_topics}")
    
    # 檢查主題標籤類型
    topic_types = set(type(t).__name__ for t in topics)
    log_diagnostic(f"主題標籤類型: {topic_types}")
    
    if len(topic_types) > 1:
        log_diagnostic("⚠️ 主題標籤類型不一致", "WARNING")
    
    # 檢查文本內容
    empty_texts = sum(1 for t in texts if not t)
    if empty_texts > 0:
        log_diagnostic(f"❌ 發現 {empty_texts} 個空文本", "ERROR")
        data_ok = False
    else:
        log_diagnostic("✓ 所有文本非空")
    
    long_texts = sum(1 for t in texts if len(t) > 1000)
    if long_texts > 0:
        log_diagnostic(f"⚠️ 發現 {long_texts} 個長度超過1000字符的文本", "WARNING")
    
    # 2. 逐步測試不同參數組合
    if not data_ok:
        log_diagnostic("\n❌ 數據存在問題，請先修復數據問題再繼續", "ERROR")
        return False
    
    # 2.1 準備測試數據集
    log_diagnostic("\n=== 2. 準備測試數據集 ===")
    # 完整數據集
    full_dataset = {
        "texts": texts,
        "reduced_embeddings": reduced_embeddings,
        "topics": topics
    }
    
    # 小數據集 (100個樣本)
    sample_size = min(100, len(texts))
    small_dataset = {
        "texts": texts[:sample_size],
        "reduced_embeddings": reduced_embeddings[:sample_size],
        "topics": topics[:sample_size]
    }
    log_diagnostic(f"準備了完整數據集 ({len(texts)} 項) 和小數據集 ({sample_size} 項)")
    
    # 2.2 定義參數組合進行測試
    log_diagnostic("\n=== 3. 參數組合測試 ===")
    
    # 測試 1: 最簡化版本 (小數據集)
    log_diagnostic("\n3.1 測試最簡化版本 (小數據集)")
    try:
        simple_fig = topic_model.visualize_documents(
            docs=small_dataset["texts"],
            reduced_embeddings=small_dataset["reduced_embeddings"],
            topics=small_dataset["topics"]
        )
        save_test_result("01_simple_small", True, simple_fig)
    except Exception as e:
        log_diagnostic(f"❌ 最簡化版本 (小數據集) 失敗: {str(e)}", "ERROR")
        save_error_details(e, "01_simple_small")
        save_test_result("01_simple_small", False, details={"error": str(e)})
    
    # 測試 2: 最簡化版本 (完整數據集)
    log_diagnostic("\n3.2 測試最簡化版本 (完整數據集)")
    try:
        full_simple_fig = topic_model.visualize_documents(
            docs=full_dataset["texts"],
            reduced_embeddings=full_dataset["reduced_embeddings"],
            topics=full_dataset["topics"]
        )
        save_test_result("02_simple_full", True, full_simple_fig)
    except Exception as e:
        log_diagnostic(f"❌ 最簡化版本 (完整數據集) 失敗: {str(e)}", "ERROR")
        save_error_details(e, "02_simple_full")
        save_test_result("02_simple_full", False, details={"error": str(e)})
    
    # 測試 3-7: 逐步添加參數 (小數據集)
    parameter_tests = [
        {
            "name": "03_width_height",
            "title": "添加寬度和高度參數",
            "params": {"width": 1200, "height": 800}
        },
        {
            "name": "04_hover_annotations",
            "title": "添加懸停和註釋參數",
            "params": {"hide_document_hover": False, "hide_annotations": False}
        },
        {
            "name": "05_title",
            "title": "添加標題參數",
            "params": {"title": "文檔主題分布圖"}
        },
        {
            "name": "06_sample",
            "title": "添加樣本參數",
            "params": {"sample": 1.0}
        },
        {
            "name": "07_full_params",
            "title": "添加所有參數",
            "params": {
                "width": 1200, 
                "height": 800,
                "hide_document_hover": False,
                "hide_annotations": False,
                "title": "文檔主題分布圖",
                "sample": 1.0
            }
        }
    ]
    
    for test in parameter_tests:
        log_diagnostic(f"\n3.{int(test['name'][:2])} 測試{test['title']} (小數據集)")
        try:
            fig = topic_model.visualize_documents(
                docs=small_dataset["texts"],
                reduced_embeddings=small_dataset["reduced_embeddings"],
                topics=small_dataset["topics"],
                **test["params"]
            )
            save_test_result(test["name"], True, fig)
        except Exception as e:
            log_diagnostic(f"❌ {test['title']} (小數據集) 失敗: {str(e)}", "ERROR")
            save_error_details(e, test["name"])
            save_test_result(test["name"], False, details={"error": str(e)})
    
    # 測試 8: 完整參數 (完整數據集)
    log_diagnostic("\n3.8 測試完整參數 (完整數據集)")
    try:
        full_params_fig = topic_model.visualize_documents(
            docs=full_dataset["texts"],
            reduced_embeddings=full_dataset["reduced_embeddings"],
            topics=full_dataset["topics"],
            width=1200, 
            height=800,
            hide_document_hover=False,
            hide_annotations=False,
            title="文檔主題分布圖 (所有文檔)",
            sample=1.0
        )
        save_test_result("08_full_params_full_data", True, full_params_fig)
    except Exception as e:
        log_diagnostic(f"❌ 完整參數 (完整數據集) 失敗: {str(e)}", "ERROR")
        save_error_details(e, "08_full_params_full_data")
        save_test_result("08_full_params_full_data", False, details={"error": str(e)})
    
    # 3. 嘗試數據轉換
    log_diagnostic("\n=== 4. 數據轉換測試 ===")
    
    # 測試 9: 轉換主題標籤為整數
    log_diagnostic("\n4.1 測試轉換主題標籤為整數")
    try:
        topics_int = [int(t) if isinstance(t, (int, float, str)) and str(t).strip('-').isdigit() else -1 for t in small_dataset["topics"]]
        int_fig = topic_model.visualize_documents(
            docs=small_dataset["texts"],
            reduced_embeddings=small_dataset["reduced_embeddings"],
            topics=topics_int,
            width=1200,
            height=800,
            title="主題標籤轉整數"
        )
        save_test_result("09_int_topics", True, int_fig)
    except Exception as e:
        log_diagnostic(f"❌ 轉換主題標籤為整數失敗: {str(e)}", "ERROR")
        save_error_details(e, "09_int_topics")
        save_test_result("09_int_topics", False, details={"error": str(e)})
    
    # 測試 10: 簡化文本
    log_diagnostic("\n4.2 測試簡化文本")
    try:
        simplified_texts = [
            (t[:100] + "...") if t and len(t) > 100 else (t if t else "Empty text")
            for t in small_dataset["texts"]
        ]
        text_fig = topic_model.visualize_documents(
            docs=simplified_texts,
            reduced_embeddings=small_dataset["reduced_embeddings"],
            topics=small_dataset["topics"],
            width=1200,
            height=800,
            title="簡化文本"
        )
        save_test_result("10_simplified_text", True, text_fig)
    except Exception as e:
        log_diagnostic(f"❌ 簡化文本失敗: {str(e)}", "ERROR")
        save_error_details(e, "10_simplified_text")
        save_test_result("10_simplified_text", False, details={"error": str(e)})
    
    # 測試 11: 標準化嵌入向量
    log_diagnostic("\n4.3 測試標準化嵌入向量")
    try:
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(small_dataset["reduced_embeddings"])
        norm_fig = topic_model.visualize_documents(
            docs=small_dataset["texts"],
            reduced_embeddings=normalized_embeddings,
            topics=small_dataset["topics"],
            width=1200,
            height=800,
            title="標準化嵌入向量"
        )
        save_test_result("11_normalized_embeddings", True, norm_fig)
    except Exception as e:
        log_diagnostic(f"❌ 標準化嵌入向量失敗: {str(e)}", "ERROR")
        save_error_details(e, "11_normalized_embeddings")
        save_test_result("11_normalized_embeddings", False, details={"error": str(e)})
    
    # 測試 12: 組合所有修復 (小數據集)
    log_diagnostic("\n4.4 測試組合所有修復 (小數據集)")
    try:
        # 應用所有修復
        topics_int = [int(t) if isinstance(t, (int, float, str)) and str(t).strip('-').isdigit() else -1 for t in small_dataset["topics"]]
        simplified_texts = [(t[:100] + "...") if t and len(t) > 100 else (t if t else "Empty text") for t in small_dataset["texts"]]
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(small_dataset["reduced_embeddings"])
        
        combined_fig = topic_model.visualize_documents(
            docs=simplified_texts,
            reduced_embeddings=normalized_embeddings,
            topics=topics_int,
            width=1200,
            height=800,
            title="組合所有修復"
        )
        save_test_result("12_combined_fixes_small", True, combined_fig)
    except Exception as e:
        log_diagnostic(f"❌ 組合所有修復 (小數據集) 失敗: {str(e)}", "ERROR")
        save_error_details(e, "12_combined_fixes_small")
        save_test_result("12_combined_fixes_small", False, details={"error": str(e)})
    
    # 測試 13: 組合所有修復 (完整數據集)
    log_diagnostic("\n4.5 測試組合所有修復 (完整數據集)")
    try:
        # 應用所有修復到完整數據集
        topics_int_full = [int(t) if isinstance(t, (int, float, str)) and str(t).strip('-').isdigit() else -1 for t in full_dataset["topics"]]
        simplified_texts_full = [(t[:100] + "...") if t and len(t) > 100 else (t if t else "Empty text") for t in full_dataset["texts"]]
        scaler_full = StandardScaler()
        normalized_embeddings_full = scaler_full.fit_transform(full_dataset["reduced_embeddings"])
        
        combined_full_fig = topic_model.visualize_documents(
            docs=simplified_texts_full,
            reduced_embeddings=normalized_embeddings_full,
            topics=topics_int_full,
            width=1200,
            height=800,
            title="組合所有修復 (完整數據集)"
        )
        save_test_result("13_combined_fixes_full", True, combined_full_fig)
    except Exception as e:
        log_diagnostic(f"❌ 組合所有修復 (完整數據集) 失敗: {str(e)}", "ERROR")
        save_error_details(e, "13_combined_fixes_full")
        save_test_result("13_combined_fixes_full", False, details={"error": str(e)})
    
    # 4. 創建備用視覺化
    log_diagnostic("\n=== 5. 創建備用視覺化 ===")
    try:
        # 創建數據框
        viz_df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'topic': [str(t) for t in topics],
            'text': [t[:50] + "..." if t and len(t) > 50 else t for t in texts]
        })
        
        # 使用 plotly express 創建散點圖
        fig_backup = px.scatter(
            viz_df,
            x='x',
            y='y',
            color='topic',
            hover_data=['text'],
            title='文檔主題分布圖（備用方案）',
            opacity=0.8,
            width=1200,
            height=800
        )
        
        save_test_result("14_backup_visualization", True, fig_backup)
        log_diagnostic("✓ 備用視覺化方案成功")
    except Exception as e:
        log_diagnostic(f"❌ 備用視覺化方案失敗: {str(e)}", "ERROR")
        save_error_details(e, "14_backup_visualization")
        save_test_result("14_backup_visualization", False, details={"error": str(e)})
    
    # 5. 檢查測試結果並提供建議
    log_diagnostic("\n=== 6. 診斷結論與建議 ===")
    
    # 收集所有測試結果
    results = {}
    for test_name in ["01_simple_small", "02_simple_full", "03_width_height", "04_hover_annotations", 
                     "05_title", "06_sample", "07_full_params", "08_full_params_full_data", 
                     "09_int_topics", "10_simplified_text", "11_normalized_embeddings", 
                     "12_combined_fixes_small", "13_combined_fixes_full", "14_backup_visualization"]:
        result_file = f"{diagnostic_path}/result_{test_name}.json"
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                results[test_name] = json.load(f)
        else:
            results[test_name] = {"success": False, "details": {"error": "測試未執行"}}
    
    # 分析成功和失敗的測試
    successful_tests = [name for name, result in results.items() if result.get("success")]
    failed_tests = [name for name, result in results.items() if not result.get("success")]
    
    log_diagnostic(f"成功的測試: {len(successful_tests)}/{len(results)}")
    for test in successful_tests:
        log_diagnostic(f"  ✓ {test}")
    
    log_diagnostic(f"失敗的測試: {len(failed_tests)}/{len(results)}")
    for test in failed_tests:
        log_diagnostic(f"  ❌ {test}")
    
    # 提供具體建議
    log_diagnostic("\n基於測試結果的建議:")
    
    if "02_simple_full" in successful_tests:
        log_diagnostic("1. ✓ 基本功能正常: 最簡化版本在完整數據集上可以工作")
        log_diagnostic("   建議: 使用最簡化參數調用 visualize_documents")
        
        # 檢查哪些參數可能導致問題
        if "02_simple_full" in successful_tests and "08_full_params_full_data" not in successful_tests:
            problematic_params = []
            if "03_width_height" not in successful_tests:
                problematic_params.append("width/height")
            if "04_hover_annotations" not in successful_tests:
                problematic_params.append("hide_document_hover/hide_annotations")
            if "05_title" not in successful_tests:
                problematic_params.append("title")
            if "06_sample" not in successful_tests:
                problematic_params.append("sample")
            
            if problematic_params:
                log_diagnostic(f"   避免使用以下可能導致問題的參數: {', '.join(problematic_params)}")
    
    elif "01_simple_small" in successful_tests:
        log_diagnostic("1. ⚠️ 數據量限制: 只有小數據集可以工作")
        log_diagnostic("   建議: 考慮分批處理數據或減少數據量")
        
        if "13_combined_fixes_full" in successful_tests:
            log_diagnostic("2. ✓ 組合修復方案在完整數據集上可以工作")
            log_diagnostic("   建議: 使用組合修復方案處理完整數據集")
    
    else:
        log_diagnostic("1. ❌ 基本功能異常: 即使最簡化版本也無法工作")
        
        if "09_int_topics" in successful_tests:
            log_diagnostic("2. ✓ 整數主題標籤有效")
            log_diagnostic("   建議: 將主題標籤轉換為整數")
        
        if "10_simplified_text" in successful_tests:
            log_diagnostic("3. ✓ 簡化文本有效")
            log_diagnostic("   建議: 簡化文本內容，限制長度")
        
        if "11_normalized_embeddings" in successful_tests:
            log_diagnostic("4. ✓ 標準化嵌入向量有效")
            log_diagnostic("   建議: 標準化嵌入向量")
        
        if "12_combined_fixes_small" in successful_tests or "13_combined_fixes_full" in successful_tests:
            log_diagnostic("5. ✓ 組合修復方案有效")
            log_diagnostic("   建議: 同時應用多個修復方案")
    
    if "14_backup_visualization" in successful_tests:
        log_diagnostic("6. ✓ 備用視覺化方案可用")
        log_diagnostic("   建議: 如果其他方法都失敗，可以使用備用視覺化方案")
    
    # 返回最佳解決方案
    best_solution = None
    
    if "08_full_params_full_data" in successful_tests:
        best_solution = "08_full_params_full_data"
    elif "02_simple_full" in successful_tests:
        best_solution = "02_simple_full"
    elif "13_combined_fixes_full" in successful_tests:
        best_solution = "13_combined_fixes_full"
    elif "12_combined_fixes_small" in successful_tests:
        best_solution = "12_combined_fixes_small"
    elif "01_simple_small" in successful_tests:
        best_solution = "01_simple_small"
    elif "14_backup_visualization" in successful_tests:
        best_solution = "14_backup_visualization"
    
    if best_solution:
        log_diagnostic(f"\n最佳解決方案: {best_solution}")
        best_solution_file = f"{diagnostic_path}/{best_solution}.html"
        if os.path.exists(best_solution_file):
            log_diagnostic(f"最佳解決方案視覺化文件: {best_solution_file}")
    else:
        log_diagnostic("\n❌ 未找到可行的解決方案", "ERROR")
    
        # 生成診斷摘要報告
        # 生成診斷摘要報告
    summary_file = f"{diagnostic_path}/diagnosis_summary.html"
    with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
            <html>
            <head>
                <title>BERTopic visualize_documents 診斷摘要</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .success {{ color: green; }}
                    .failure {{ color: red; }}
                    .warning {{ color: orange; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .solution {{ background-color: #e6f7ff; padding: 15px; border-left: 5px solid #1890ff; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>BERTopic visualize_documents 診斷摘要</h1>
                <p>診斷時間: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h2>數據概況</h2>
                <ul>
                    <li>文本數量: {len(texts)}</li>
                    <li>主題數量: {len(topics)}</li>
                    <li>唯一主題數量: {len(set(topics))}</li>
                    <li>嵌入向量形狀: {reduced_embeddings.shape}</li>
                    <li>嵌入向量範圍: [{embedding_min:.4f}, {embedding_max:.4f}]</li>
                </ul>
                
                <h2>測試結果摘要</h2>
                <table>
                    <tr>
                        <th>測試名稱</th>
                        <th>結果</th>
                        <th>視覺化</th>
                    </tr>
        """)
            
            # 添加測試結果行
            for test_name, result in sorted(results.items()):
                success = result.get("success", False)
                status = "成功" if success else "失敗"
                status_class = "success" if success else "failure"
                
                viz_link = ""
                html_file = f"{test_name}.html"
                if os.path.exists(f"{diagnostic_path}/{html_file}"):
                    viz_link = f'<a href="{html_file}" target="_blank">查看視覺化</a>'
                
                f.write(f"""
            <tr>
                <td>{test_name}</td>
                <td class="{status_class}">{status}</td>
                <td>{viz_link}</td>
            </tr>""")
            
            # 添加建議解決方案
            f.write("""
        </table>
        
        <h2>建議解決方案</h2>
        <div class="solution">
    """)
            
            if best_solution:
                best_solution_file = f"{best_solution}.html"
                if os.path.exists(f"{diagnostic_path}/{best_solution_file}"):
                    f.write(f"""
            <p><strong>最佳解決方案:</strong> {best_solution}</p>
            <p><a href="{best_solution_file}" target="_blank">查看最佳解決方案視覺化</a></p>
    """)
                    
                    # 添加具體代碼建議
                    if best_solution == "02_simple_full":
                        f.write("""
            <h3>建議代碼:</h3>
            <pre>
    fig_docs_original = topic_model.visualize_documents(
        docs=texts,
        reduced_embeddings=reduced_embeddings,
        topics=topics
    )
    fig_docs_original.write_html(f"{visualization_path}/documents_visualization_original.html")
            </pre>
    """)
                    elif best_solution == "13_combined_fixes_full":
                        f.write("""
            <h3>建議代碼:</h3>
            <pre>
    # 應用所有修復
    topics_int = [int(t) if isinstance(t, (int, float, str)) and str(t).strip('-').isdigit() else -1 for t in topics]
    simplified_texts = [(t[:100] + "...") if t and len(t) > 100 else (t if t else "Empty text") for t in texts]
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(reduced_embeddings)

    fig_docs_original = topic_model.visualize_documents(
        docs=simplified_texts,
        reduced_embeddings=normalized_embeddings,
        topics=topics_int,
        width=1200,
        height=800,
        title="文檔主題分布圖 (所有文檔)"
    )
    fig_docs_original.write_html(f"{visualization_path}/documents_visualization_original.html")
            </pre>
    """)
                    elif best_solution == "14_backup_visualization":
                        f.write("""
            <h3>建議代碼 (備用視覺化方案):</h3>
            <pre>
    # 創建備用視覺化
    import plotly.express as px
    import pandas as pd

    # 創建數據框
    viz_df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'topic': [str(t) for t in topics],
        'text': [t[:50] + "..." if t and len(t) > 50 else t for t in texts]
    })

    # 使用 plotly express 創建散點圖
    fig_backup = px.scatter(
        viz_df,
        x='x',
        y='y',
        color='topic',
        hover_data=['text'],
        title='文檔主題分布圖（備用方案）',
        opacity=0.8,
        width=1200,
        height=800
    )

    fig_backup.write_html(f"{visualization_path}/documents_visualization_backup.html")
            </pre>
    """)
            else:
                f.write("""
            <p class="failure"><strong>未找到可行的解決方案</strong></p>
            <p>建議檢查 BERTopic 版本，考慮更新或降級。</p>
            <p>如果問題持續，請考慮使用備用視覺化方法或減少數據量。</p>
    """)
            
            f.write("""
                </div>
                
                <h2>診斷日誌</h2>
                <p><a href="visualize_documents_diagnosis.log" target="_blank">查看完整診斷日誌</a></p>
                
            </body>
            </html>
            """)
        
            log_diagnostic(f"\n診斷摘要報告已生成: {summary_file}")
        
        # 返回是否找到解決方案
    return best_solution is not None

# 執行診斷
best_solution = diagnose_visualize_documents(
    topic_model=topic_model,
    texts=texts,
    reduced_embeddings=reduced_embeddings,
    topics=topics,
    visualization_path=visualization_path
)

# 根據診斷結果決定如何繼續
if best_solution:
    logging.info("診斷完成，已找到可行的解決方案")
    
    # 檢查最佳解決方案並應用
    best_solution_file = f"{diagnostic_path}/{best_solution}.html"
    if os.path.exists(best_solution_file):
        logging.info(f"最佳解決方案視覺化文件: {best_solution_file}")
        
        # 複製最佳解決方案到主視覺化目錄
        import shutil
        target_file = f"{visualization_path}/documents_visualization_original.html"
        shutil.copy2(best_solution_file, target_file)
        logging.info(f"已將最佳解決方案複製到: {target_file}")
        
        # 提供成功信息
        print(f"\n✅ 視覺化生成成功！文件位置: {target_file}")
        print(f"診斷摘要報告: {diagnostic_path}/diagnosis_summary.html")
else:
    logging.error("診斷未找到可行的解決方案，將嘗試備用視覺化方案")
    
    # 嘗試備用視覺化方案
    try:
        logging.info("嘗試備用視覺化方案...")
        
        # 創建備用視覺化
        import plotly.express as px
        import pandas as pd

        # 創建數據框
        viz_df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'topic': [str(t) for t in topics],
            'text': [t[:50] + "..." if t and len(t) > 50 else t for t in texts]
        })

        # 使用 plotly express 創建散點圖
        fig_backup = px.scatter(
            viz_df,
            x='x',
            y='y',
            color='topic',
            hover_data=['text'],
            title='文檔主題分布圖（備用方案）',
            opacity=0.8,
            width=1200,
            height=800
        )

        backup_file = f"{visualization_path}/documents_visualization_backup.html"
        fig_backup.write_html(backup_file)
        logging.info(f"備用視覺化方案成功，文件位置: {backup_file}")
        
        print(f"\n⚠️ 原生視覺化失敗，但備用視覺化生成成功！文件位置: {backup_file}")
        print(f"診斷摘要報告: {diagnostic_path}/diagnosis_summary.html")
    except Exception as e:
        logging.error(f"備用視覺化方案也失敗: {e}")
        logging.error(f"詳細錯誤信息: {traceback.format_exc()}")
        print("\n❌ 所有視覺化嘗試都失敗了。請查看診斷報告了解詳情。")
        print(f"診斷摘要報告: {diagnostic_path}/diagnosis_summary.html")
# =====================================================================
# 診斷代碼結束
# =====================================================================


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
topic_info_save_path = "./results/four_categories_v3/topic_info.txt"  # 從 v2 改為 v3
with open(topic_info_save_path, "w", encoding="utf-8") as f:
    f.write("主題分布概況：\n")
    f.write(topic_info.to_string())
print(f"主題資訊已保存至: {topic_info_save_path}")

# 2. 將各主題的代表性文章原始內容列出來
representative_docs_save_path = "./results/four_categories_v3/representative_docs.txt"  # 從 v2 改為 v3
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
csv_with_topics_path = "./results/four_categories_v3/Kinmen_splitData_with_topics.csv"  # 從 v2 改為 v3
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


